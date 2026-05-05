import warnings; warnings.filterwarnings('ignore')
import os, urllib.request
import fitz, faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import ollama
from rank_bm25 import BM25Okapi


class PaperSearchEngine:

    """Purpose: Load a research-lead article in PDF, index its content, and answer questions about it using a hybrid retrieval + LLM pipeline.
    """
    
    def __init__(self, model_name: str, pdf_path=None, pdf_url=None,
                 chunk_size=150, overlap=30, max_length=128,
                 use_contextual=True, ollama_model='qwen2.5:7b',
                 ollama_validator='llama3.2:3b', prompts=None):

        self.model_name       = model_name
        self.chunk_size       = chunk_size
        self.overlap          = overlap
        self.max_length       = max_length
        self.use_contextual   = use_contextual
        self.ollama_model     = ollama_model
        self.ollama_validator = ollama_validator
        self.pdf_url          = pdf_url
        self.pdf_path         = self._download(pdf_url) if pdf_url else pdf_path

        # prompts — use defaults if not provided via Hydra
        self.prompts = prompts or {
            'qa_system':      'You are a scientific assistant. Answer questions based only on the context provided.',
            'contextualize':  'Here is a document excerpt:\n{doc_preview}\n\nHere is a specific chunk:\n{chunk_text}\n\nWrite a single short sentence (max 30 words) situating this chunk within the document. Answer only with that sentence.',
            'describe_image': 'This is a figure from a scientific paper. Describe what it shows, including axis labels, trends, key values, and what conclusion it supports.',
            'validate':       'Rate this answer on accuracy and completeness (1-5 scale):\n\nContext:\n{context}\n\nQuestion: {query}\n\nGenerated answer: {answer}\n\nFormat: Score: X/5 | Issues: ... | Confidence: ...',
            
        }

        # extract and chunk
        raw_chunks = self._chunk()

        # optionally enrich chunks with Qwen-generated context
        if self.use_contextual:
            print("Generating contextual descriptions for chunks...")
            self.chunks = self._contextualize(raw_chunks)
        else:
            self.chunks = raw_chunks

        # load embedding model
        components     = self._load_model()
        self.tokenizer = components['tokenizer']
        self.model     = components['model']

        # build FAISS index (semantic)
        self.index = self._build_index()

        # build BM25 index (lexical)
        self.bm25 = self._build_bm25()

    def __repr__(self):
        mode = "contextual" if self.use_contextual else "standard"
        return f"PaperSearchEngine(model='{self.model_name}', chunks={len(self.chunks)}, mode={mode})"

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _download(self, url):
        local_path = url.split('/')[-1] + '.pdf' if not url.endswith('.pdf') else url.split('/')[-1]
        if not os.path.exists(local_path):
            print(f'Downloading paper from {url}...')
            urllib.request.urlretrieve(url, local_path)
        else:
            print(f'Already exists — size: {os.path.getsize(local_path) / 1e6:.1f} MB')
        return local_path

    def _chunk(self):
       # Splits each page into overlapping chunks
        # e.g. a dummy page of text (15 words (one, \ldots, fifteen)) along with chunk_size=5 and overlap=2 produces the following chunks:
        # Chunk 1: one two three four five
        # Chunk 2: four five six seven eight
        # Chunk 3: seven eight nine ten eleven
        # Chunk 4: ten eleven twelve thirteen fourteen
        # Chunk 5: thirteen fourteen fifteen

        doc    = fitz.open(self.pdf_path)
        words  = []
        for page_num, page in enumerate(doc, start=1):
            for word in page.get_text("words"):
                words.append((word[4], page_num))

        chunks = []
        # step = 150 - 30 = 120 means we start a new chunk every 120 words, creating an overlap of 30 words between consecutive chunks
        step   = self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            chunk_words = words[i: i + self.chunk_size]
            if not chunk_words:
                continue
            text      = ' '.join(w[0] for w in chunk_words)
            page      = chunk_words[len(chunk_words) // 2][1]
            chunks.append({'text': text, 'page': page})

        print(f'Extracted {len(chunks)} chunks.')
        return chunks
    
    def _contextualize(self, chunks):
        """Prepend a short Qwen-generated context to each chunk."""
        # Build a short document summary from first ~500 words
        # This comes from: https://www.anthropic.com/engineering/contextual-retrieval
        # "This chunk discusses XGBoost's sparsity-aware algorithm performance. the algorithm achieved 50x speedup"
        doc_preview = ' '.join(c['text'] for c in chunks[:5]) # first 5 chunks as preview
        
        contextualized = []
        for chunk in tqdm(chunks, desc="Contextualizing"):
            prompt = self.prompts['contextualize'].format(
                doc_preview=doc_preview,
                chunk_text=chunk['text']
            )
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.0}
            )
            context_sentence = response['message']['content'].strip()
            enriched_text    = f"{context_sentence} {chunk['text']}"
            contextualized.append({'text': enriched_text, 'page': chunk['page']})

        return contextualized

    def _load_model(self):
        # First, the tokenizer converts raw text into token IDs that the model at hand can understand
        # e.g. "I am a student." -> [101, 1045, 2572, 1037, 3231, 1012, 102]
        # Second, the model at hand takes each token ID and produces embeddings
        print(f'Loading model: {self.model_name}')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model     = AutoModel.from_pretrained(self.model_name)
        model.eval()
        return {'tokenizer': tokenizer, 'model': model}

    def _embed_text(self, texts):
        encoded = self.tokenizer(
            texts, 
            padding=True, # pads shorter sequences to the longest in the batch using token ID '0'
            # when TRUE; "Hello"  → [101, 7592,  102,    0,   0] (given) "Hello world today"  → [101, 7592, 2088, 2651, 102]
            truncation=True,  # truncation = True + max_length = 128 if a text has more than 128 tokens, cut it off at 128
            max_length=self.max_length,
            return_tensors='pt' # return PyTorch tensors
        )
        with torch.no_grad():
            output    = self.model(**encoded)
            embedding = output.last_hidden_state[:, 0, :] # extracts the [CLS] token embedding; [CLS] Hello world today [SEP]
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.numpy()

    def _build_index(self):
        print('Building FAISS index...')
        texts      = [c['text'] for c in self.chunks]
        batch_size = 32
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch      = texts[i: i + batch_size]
            all_embeds.append(self._embed_text(batch))
        embeddings = np.vstack(all_embeds).astype('float32')
        index      = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        print(f'FAISS index built with {index.ntotal} vectors.')
        return index

    def _build_bm25(self):
        """Build a BM25 index over chunk texts for lexical retrieval."""
        print('Building BM25 index...')
        tokenized = [c['text'].lower().split() for c in self.chunks]
        return BM25Okapi(tokenized)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def _semantic_search(self, query, top_k):
        """FAISS semantic search — returns list of (chunk_idx, score)."""
        q_embed = self._embed_text([query]).astype('float32')
        scores, indices = self.index.search(q_embed, top_k)
        return list(zip(indices[0], scores[0]))

    def _bm25_search(self, query, top_k):
        """BM25 lexical search — returns list of (chunk_idx, score)."""
        tokenized_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenized_query)
        top_indices     = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]

    def _rank_fusion(self, semantic_hits, bm25_hits, top_k, k=60):
        """Reciprocal Rank Fusion to merge semantic and BM25 results."""
        scores = {}
        for rank, (idx, _) in enumerate(semantic_hits):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        for rank, (idx, _) in enumerate(bm25_hits):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def query(self, query_text, top_k=3):
        """Hybrid retrieval: semantic + BM25 with rank fusion."""
        semantic_hits = self._semantic_search(query_text, top_k * 2)
        bm25_hits     = self._bm25_search(query_text, top_k * 2)
        fused         = self._rank_fusion(semantic_hits, bm25_hits, top_k)
        results = []
        for idx, score in fused:
            results.append({
                'text':  self.chunks[idx]['text'],
                'page':  self.chunks[idx]['page'],
                'score': round(float(score), 4)
            })
        return results

    def chat(self, query, history, top_k=3, temperature=0.1):
        """Multi-turn conversational Q&A with retrieval context."""
        results = self.query(query, top_k)
        context = '\n\n'.join([
            f'[Page {r["page"]}]: {r["text"]}' for r in results
        ])
        messages = [
            {
                'role': 'system',
                'content': (
                    self.prompts['qa_system'] + f'\n\nContext:\n{context}'
                )
            }
        ] + history + [{'role': 'user', 'content': query}]

        response = ollama.chat(
            model=self.ollama_model,
            messages=messages,
            options={"temperature": temperature}
        )
        return response['message']['content'], results

    def answer_with_validation(self, query, top_k=3, temperature=0.1):
        """Single-turn Q&A with a second LLM validation pass."""
        results = self.query(query, top_k)
        context = '\n\n'.join([
            f'[Page {r["page"]}]: {r["text"]}' for r in results
        ])

        # generation
        answer_response = ollama.chat(
            model=self.ollama_model,
            messages=[{
                'role': 'user',
                'content': self.prompts['qa_system'] + f'\n\nContext:\n{context}\n\nQuestion: {query}'
            }],
            options={"temperature": temperature}
        )
        answer = answer_response['message']['content']

        # validation
        validation_response = ollama.chat(
            model=self.ollama_validator,
            messages=[{
                'role': 'user',
                'content': self.prompts['validate'].format(
                    context=context,
                    query=query,
                    answer=answer
                )
            }],
            options={"temperature": 0.0}
        )

        return {
            'answer':      answer,
            'validation':  validation_response['message']['content'],
            'chunks_used': len(results),
            'chunks':      results
        }