# Semantic Search & RAG Pipeline

A semantic search and question-answering system for research papers using Transformers, FAISS, and two LLMs
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

![PaperSearch App](./images/app_view.png)

## Architecture
```
PDF → Chunks → Embeddings → FAISS Index → Search Results → LLM Answer → Validation
 ↓       ↓          ↓            ↓             ↓               ↓             ↓
Research  p-word   384-dim    Vector       Top-k          llama3.2:1b   qwen2.5:3b
Paper     segments  vectors    similarity   chunks         + context     Score +
URL                            search                                    feedback
```
Note that the user can select each of the above-given values on the Streamlit page.

## Step-by-step process
1) **PDF Processing**: PDF (URL) → overlapping word chunks with page references
2) **Embedding**: Chunks → 384-dim vectors via `BAAI/bge-large-en-v1.5`
3) **Indexing**: Vectors stored in FAISS `IndexFlatIP` for cosine similarity search
4) **Query Processing**: User question → embedded using the same model
5) **Retrieval**: Top-k most similar chunks returned
6) **Generation**: Retrieved chunks + query → `llama3.2:1b` (a light model)
7) **Validation**: Generated answer → `qwen2.5:3b` → quality score and feedback

## Key components
- **Pre-processing**: PyMuPDF, text chunking, overlap handling
- **LLM Models**: Transformers (`BAAI/bge-large-en-v1.5`), Ollama (`llama3.2:1b`, `qwen2.5:3b`)
- **Vector Search**: FAISS inner product index
- **Interfaces**: Streamlit web app, command-line tool

## Setup

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies
```bash
uv venv
source .venv/bin/activate
uv sync
```

### 3. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4. Pull models
```bash
ollama pull llama3.2:1b
ollama pull qwen2.5:3b
```

### 5. Start Ollama server
```bash
ollama serve
```

## Usage

### Web interface
```bash
streamlit run app.py
```

### Terminal interface
```bash
python main.py --model BAAI/bge-large-en-v1.5 --chunk_size 150
```

## Requirements
- Python 3.12+
- 8GB RAM minimum (given the two LLMs used)
- macOS, Linux, or Windows


## Acknowledgements
- [Meta FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [Ollama](https://ollama.com)
- [Streamlit](https://streamlit.io)
- [Claude AI](https://claude.ai)

## License
This project is licensed under the MIT License — see the LICENSE file for details (in short: anyone can use, modify, distribute freely).