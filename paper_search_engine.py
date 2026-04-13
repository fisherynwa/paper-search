import os, textwrap, urllib.request
import fitz, faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import io
import warnings
import ollama

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

class PaperSearchEngine:
    """
    A search engine for research/ news papers that uses a transformer model to encode chunks of text from the paper and FAISS for similarity search.
    
    Args: model_name (str): the name of the transformer model to use for encoding text
        pdf_path (str): local path to the PDF file; 
        pdf_url (str): URL to read the PDF file directly into memory without saving to disk
        chunk_size (int): number of words in each chunk;
        overlap (int): number of overlapping words between consecutive chunks; 
        max_length (int): maximum number of tokens for the transformer model input.
    """
    def __init__(
        self, 
        model_name: str, 
        pdf_path: str = None, 
        pdf_url: str = None, 
        chunk_size: int = 300, 
        overlap: int = 50, 
        max_length: int = 128
    ):
        if not pdf_path and not pdf_url:
            raise ValueError("Either pdf_path or pdf_url must be provided.")
 
        # configurable parameters
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_length = max_length
        self.pdf_url  = pdf_url
        self.pdf_path = pdf_path
        # class attributes
        self.chunks = self._chunk() 
        components = self._load_model()
        self.model = components['model']
        self.tokenizer = components['tokenizer']

        self.index = self._build_index()
        
    ##################################################################
    # BEGINS: helper utilities; not part of the public class interface
    ###################################################################
    
    def _chunk(self):
        # Splits each page into overlapping chunks
        # e.g. a dummy page of text (15 words (one, \ldots, fifteen)) along with chunk_size=5 and overlap=2 produces the following chunks:
        # Chunk 1: one two three four five
        # Chunk 2: four five six seven eight
        # Chunk 3: seven eight nine ten eleven
        # Chunk 4: ten eleven twelve thirteen fourteen
        # Chunk 5: thirteen fourteen fifteen
        if self.pdf_path:
           doc = fitz.open(self.pdf_path)
        else:
           response = urllib.request.urlopen(self.pdf_url)
           doc = fitz.open(stream=io.BytesIO(response.read()), filetype='pdf')
           
        chunks = []
        for i, page in enumerate(doc):
            words = page.get_text().split()
            for start in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[start : start + self.chunk_size]
                chunks.append({
                    'text': ' '.join(chunk_words),
                    'page': i + 1
                })
                if start + self.chunk_size >= len(words):
                    break
        return chunks

    def _load_model(self):
        """Loads the tokenizer and the model specified by model_name for embedding generation."""
        # First, the tokenizer converts raw text into token IDs that the model at hand can understand
        # e.g. "I am a student." -> [101, 1045, 2572, 1037, 3231, 1012, 102]
        # Second, the model at hand takes each token ID and produces embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model     = AutoModel.from_pretrained(self.model_name)
        model.eval()
        return {'tokenizer': tokenizer, 'model': model}

    def _embed_text(self, text):
        """Converts a text string into a normalised embedding vector."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt', # return PyTorch tensors
            truncation=True, # truncation = True + max_length = 128 if a text has more than 128 tokens, cut it off at 128
            max_length=self.max_length, # the maximum number of tokens allowed; if a text has more than max_length tokens, it will be truncated to fit this length
            padding=True # pads shorter sequences to the longest in the batch using token ID '0'
            # when TRUE; "Hello"  → [101, 7592,  102,    0,   0] (given) "Hello world today"  → [101, 7592, 2088, 2651, 102]
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        token_emb     = outputs.last_hidden_state # shape: (batch, num_tokens, hidden_dim) — one vector per token
        mask_expanded = inputs['attention_mask'].unsqueeze(-1).float() # attention_mask is a binary tensor that indicates which tokens are actual text (1) and which are padding (0)
        embedding     = (token_emb * mask_expanded).sum(1) / mask_expanded.sum(1) 
        embedding     = F.normalize(embedding, p=2, dim=1) # Scales the vector so its length equals exactly 1
        return embedding.squeeze().numpy().astype('float32')

    def _build_index(self):
        """Encodes all chunks and builds a FAISS index for similarity search."""
        # encode every chunk into an embedding vector
        embeddings = np.array(
            [self._embed_text(c['text']) for c in tqdm(self.chunks, desc='Encoding')],
            dtype='float32'
        )
        
        # build FAISS index; cosine similarity
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        print(f'Index built — {index.ntotal} vectors stored')
        return index
    
    ##################################################################
    # public interface
    ##################################################################

    def query(self, query, top_k=3):
        """Takes a query string, encodes it, and retrieves the top_k most similar chunks from the paper."""
        query_embedding = self._embed_text(query) # encode the query into an embedding vector using the same model and tokenizer as for the chunks
        xq = np.array([query_embedding], dtype='float32') # FAISS expects a 2D array of shape (n_queries, embedding_dim); here we have just one query, so we wrap it in an extra list to make it 2D
        scores, indices = self.index.search(xq, top_k)
        return [
            {'text': self.chunks[i]['text'], 'page': self.chunks[i]['page'], 'score': float(scores[0][r])}
            for r, i in enumerate(indices[0])
        ]

    def answer(self, query, top_k=3):
        """Retrieves top_k chunks and generates a direct answer using a local LLM."""
    
        # retrieve relevant chunks
        results = self.query(query, top_k)
        context = '\n\n'.join([
            f'[Page {r["page"]}]: {r["text"]}' for r in results
        ])
        
        # generate its answer
        response = ollama.chat(
            model='llama3.2:1b',
            messages=[{
                'role': 'user',
                'content': (
                    f'You are a scientific assistant. Answer the question based only on the context provided.\n\n'
                    f'Context:\n{context}\n\n'
                    f'Question: {query}'
                )
            }]
        )
        return response['message']['content']

    def answer_with_validation(self, query, top_k=3):
        """Generates an answer and validates it with a second LLM pass."""
        
        # retrieve relevant chunks
        results = self.query(query, top_k)
        context = '\n\n'.join([
            f'[Page {r["page"]}]: {r["text"]}' for r in results
        ])
        
        # generate a answer
        answer_response = ollama.chat(
            model='llama3.2:1b',
            messages=[{
                'role': 'user',
                'content': (
                    f'You are a scientific assistant. Answer as precise as possilbe the question based only on the context provided. Do NOT use any outside knowledge. If the answer is not in the context, say so.\n\n'
                    f'Context:\n{context}\n\n'
                    f'Question: {query}'
                )
            }]
        )
        answer = answer_response['message']['content']
        
        # validate the generated answer
        validation_response = ollama.chat(
            model='qwen2.5:3b',
            messages=[{
                'role': 'user', 
                'content': f'''Rate this answer on accuracy and completeness (on a scale from 1 to 5):

    Context from research paper:
    {context}

    Question: {query}

    Generated answer: {answer}

    Provide:
    - Score (1-5): where 5 = fully accurate and complete, 1 = inaccurate or missing key info
    - Issues: any factual errors or important missing details
    - Confidence: high/medium/low based on context quality

    Format: Score: X/5 | Issues: ... | Confidence: ...'''
            }]
        )
        
        return {
            'answer': answer,
            'validation': validation_response['message']['content'],
            'chunks_used': len(results),
            'chunks': results
        }

## This function is not part of the PaperSearchEngine class; it is a standalone utility to compare results from two different engines on the same query.
def compare_engines(engine1, engine2, query, top_k=3):
    """Compares retrieval results between two PaperSearchEngine instances on the same query."""
    results1 = engine1.query(query, top_k)
    results2 = engine2.query(query, top_k)

    print(f'\nQuery: "{query}"')
    print('=' * 70)

    for r, (res1, res2) in enumerate(zip(results1, results2), 1):
        print(f'\nRank {r}')
        print(f'[{engine1.model_name}]')
        print(f'  Score : {res1["score"]:.4f}  |  Page: {res1["page"]}')
        print(f'  Text  : {textwrap.fill(res1["text"][:500], width=65)}')
        print(f'[{engine2.model_name}]')
        print(f'  Score : {res2["score"]:.4f}  |  Page: {res2["page"]}')
        print(f'  Text  : {textwrap.fill(res2["text"][:500], width=65)}')
        print('-' * 70)
        

