import torch
torch.set_num_threads(1)

import streamlit as st
import base64
import fitz
import ollama
from omegaconf import OmegaConf
from paper_search_engine import PaperSearchEngine
from pathlib import Path

##################################################################
# page config
##################################################################
st.set_page_config(
    page_title='Paper Search',
    page_icon='🔍',
    layout='wide'
)

##################################################################
# styling
##################################################################
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('./app_style/styles.css')

##################################################################
# session state initialisation
##################################################################
if 'history' not in st.session_state:
    st.session_state.history = []

##################################################################
# load config directly via OmegaConf (no Hydra needed in Streamlit)
##################################################################
_base = Path(__file__).parent / "conf"
cfg = OmegaConf.merge(
    OmegaConf.load(_base / "model"      / "default.yaml"),
    OmegaConf.load(_base / "retrieval"  / "default.yaml"),
    OmegaConf.load(_base / "generation" / "default.yaml"),
)
# default PDF URL
if "pdf_url" not in cfg:
    cfg.pdf_url = "https://arxiv.org/pdf/1603.02754"

##################################################################
# cached engine loader — only reloads when parameters change
##################################################################
@st.cache_resource(show_spinner='Loading model and building index...')
def load_engine(model_name, pdf_url, chunk_size, overlap, max_length,
                use_contextual, ollama_model, ollama_validator, prompts_key):
    return PaperSearchEngine(
        model_name       = model_name,
        pdf_url          = pdf_url,
        chunk_size       = chunk_size,
        overlap          = overlap,
        max_length       = max_length,
        use_contextual   = use_contextual,
        ollama_model     = ollama_model,
        ollama_validator = ollama_validator,
        prompts          = OmegaConf.to_container(cfg.prompts, resolve=True),
    )


##################################################################
# sidebar — configuration
##################################################################
with st.sidebar:
    st.markdown('## ⚙️ Configuration')

    pdf_url = st.text_input(
        'PDF URL',
        value=cfg.pdf_url,
        help='URL of the research paper PDF'
    )

    model_name = st.selectbox(
        'Embedding model',
        options=[
            'BAAI/bge-large-en-v1.5',
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/allenai-specter'
        ],
        index=0
    )

    ollama_model = st.selectbox(
        'Ollama model',
        options=['qwen2.5:7b', 'qwen2.5:3b', 'llama3.2:3b'],
        index=0
    )

    chunk_size      = st.slider('Chunk size (words)',  50,  500, cfg.chunk_size,  step=25)
    overlap         = st.slider('Overlap (words)',     10,  100, cfg.overlap,      step=5)
    max_length      = st.slider('Max token length',    64,  512, cfg.max_length,   step=32)
    top_k           = st.slider('Top-k results',        1,   10, cfg.top_k)
    temperature     = st.slider('Generation temperature', 0.0, 1.0, cfg.temperature, step=0.05)

    use_contextual  = st.toggle(
        'Contextual embeddings',
        value=cfg.use_contextual,
        help='Enriches each chunk with Qwen-generated context before embedding. Slower on first load.'
    )

    use_rag = st.toggle(
        'Generate answer (RAG)', value=True,
        help='Uses Ollama for answer generation. Make sure `ollama serve` is running.'
    )

    show_validation = st.toggle(
        'Show answer validation',
        value=False,
        help='Runs a second LLM call to validate each answer. Slower.'
    )

    if st.button('🗑️ Clear chat history'):
        st.session_state.history = []
        st.rerun()

    st.markdown('---')
    st.markdown('<span class="tag">FAISS + BM25 + Transformers + Ollama</span>', unsafe_allow_html=True)

##################################################################
# main area
##################################################################
st.markdown('# 🔍 Paper Search')
st.markdown('Semantic search and question answering over research papers.')
st.markdown('---')

# load engine
engine = load_engine(
    model_name       = model_name,
    pdf_url          = pdf_url,
    chunk_size       = chunk_size,
    overlap          = overlap,
    max_length       = max_length,
    use_contextual   = use_contextual,
    ollama_model     = ollama_model,
    ollama_validator = cfg.validator,
    prompts_key      = str(OmegaConf.to_container(cfg.prompts)),
)

##################################################################
# tabs
##################################################################
tab1, tab2 = st.tabs(['💬 Q&A', '🖼️ Figures'])

# ----------------------------------------------------------------
# Tab 1 — conversational Q&A
# ----------------------------------------------------------------
with tab1:

    # render chat history
    for msg in st.session_state.history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # chat input
    if query := st.chat_input('Ask a question about the paper...'):

        # show user message
        with st.chat_message('user'):
            st.markdown(query)

        with st.spinner('Searching and generating...'):
            if use_rag:
                try:
                    answer, results = engine.chat(
                        query,
                        history     = st.session_state.history,
                        top_k       = top_k,
                        temperature = temperature,
                    )

                    # show assistant answer
                    with st.chat_message('assistant'):
                        st.markdown(answer)

                    # optional validation
                    if show_validation:
                        with st.spinner('Validating...'):
                            validation = engine.answer_with_validation(
                                query,
                                top_k       = top_k,
                                temperature = temperature,
                            )
                        with st.expander('🔍 Answer validation', expanded=False):
                            st.text(validation['validation'])
                            st.markdown(f'**Chunks used:** {validation["chunks_used"]}')

                    # show retrieved chunks
                    with st.expander('📚 Retrieved chunks', expanded=False):
                        for r in results:
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="score">Score: {r['score']:.4f} &nbsp;|&nbsp; Page {r['page']}</div>
                                <div class="text">{r['text'][:500]}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # update history
                    st.session_state.history.append({'role': 'user',      'content': query})
                    st.session_state.history.append({'role': 'assistant', 'content': answer})

                except Exception as e:
                    st.error(f'Ollama error: {e} — make sure `ollama serve` is running.')
            else:
                # retrieval only — no generation
                results = engine.query(query, top_k=top_k)
                with st.chat_message('assistant'):
                    st.markdown('**Retrieved chunks** (RAG disabled):')
                    for r in results:
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="score">Score: {r['score']:.4f} &nbsp;|&nbsp; Page {r['page']}</div>
                            <div class="text">{r['text'][:500]}</div>
                        </div>
                        """, unsafe_allow_html=True)

# ----------------------------------------------------------------
# Tab 2 — figures
# ----------------------------------------------------------------
with tab2:
    st.markdown('### 🖼️ Paper Figures')
    st.markdown('Extracts figures from the PDF and describes them using a vision model.')

    vision_model = st.selectbox(
        'Vision model',
        options=['llava:7b'],
        help='Make sure the selected model is pulled in Ollama.'
    )

    if st.button('Extract & describe figures'):
        with st.spinner('Extracting figures from PDF...'):
            try:
                doc    = fitz.open(engine.pdf_path)
                images = []
                for page_num, page in enumerate(doc, start=1):
                    for img in page.get_images():
                        xref = img[0]
                        pix  = fitz.Pixmap(doc, xref)
                        # skip tiny images (icons, bullets)
                        if pix.width < 100 or pix.height < 100:
                            continue
                        # convert CMYK to RGB if needed
                        if pix.n > 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        images.append({
                            'bytes':    pix.tobytes('png'),
                            'b64':      base64.b64encode(pix.tobytes('png')).decode(),
                            'page':     page_num,
                            'width':    pix.width,
                            'height':   pix.height,
                        })

                if not images:
                    st.warning('No figures found in this PDF.')
                else:
                    st.success(f'Found {len(images)} figure(s). Describing with {vision_model}...')
                    descriptions = []
                    for i, img in enumerate(images):
                        with st.spinner(f'Describing figure {i+1}/{len(images)}...'):
                            response = ollama.chat(
                                model=vision_model,
                                messages=[{
                                    'role':    'user',
                                    'content': cfg.prompts.describe_image,
                                    'images':  [img['b64']]
                                }],
                                options={"temperature": 0.0}
                            )
                            description = response['message']['content']
                            descriptions.append(description)

                        # display image + description side by side
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(img['bytes'], caption=f'Page {img["page"]}')
                        with col2:
                            st.markdown(f'**Figure {i+1} — Page {img["page"]}**')
                            st.markdown(description)
                        st.markdown('---')

            except Exception as e:
                st.error(f'Error: {e} — make sure `ollama serve` is running and {vision_model} is pulled.')