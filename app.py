import streamlit as st
from paper_search_engine import PaperSearchEngine

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
if 'query' not in st.session_state:
    st.session_state.query = ''

##################################################################
# cached engine loader — only reloads when parameters change
##################################################################
@st.cache_resource(show_spinner='Loading model and building index...')
def load_engine(model_name, pdf_url, chunk_size, overlap, max_length):
    return PaperSearchEngine(
        model_name=model_name,
        pdf_url=pdf_url,
        chunk_size=chunk_size,
        overlap=overlap,
        max_length=max_length
    )

##################################################################
# sidebar — configuration
##################################################################
with st.sidebar:
    st.markdown('## ⚙️ Configuration')

    pdf_url = st.text_input(
        'PDF URL',
        value='https://arxiv.org/pdf/1603.02754',
        help='URL of the research paper PDF'
    )

    model_name = st.selectbox(
        'Embedding model',
        options=[
            'BAAI/bge-large-en-v1.5',
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/allenai-specter'
        ]
    )

    chunk_size = st.slider('Chunk size (words)', 50,  500, 150, step=25)
    overlap    = st.slider('Overlap (words)',     10,  100,  30, step=5)
    max_length = st.slider('Max token length',    64,  512, 128, step=32)
    top_k      = st.slider('Top-k results',        1,   10,   3)

    use_rag = st.toggle(
        'Generate answer (RAG)', value=True,
        help='Uses the selected Ollama model for answer generation and validation. Make sure `ollama serve` is running in a separate terminal.'
    )

    st.markdown('---')
    st.markdown('<span class="tag">FAISS + Transformers + Ollama</span>', unsafe_allow_html=True)

##################################################################
# main area
##################################################################
st.markdown('# 🔍 PaperSearch')
st.markdown('Semantic search and question answering over research papers.')
st.markdown('---')

# load engine — always before query input so input is never blocked
engine = load_engine(model_name, pdf_url, chunk_size, overlap, max_length)

# query input — key ties it to session state so it persists across rerenders
query = st.text_input(
    'Enter your query',
    placeholder='e.g. how does XGBoost handle missing values?',
    key='query'
)

# search and answer
if query: 
    with st.spinner('Searching...'):
        results = engine.query(query, top_k=top_k)

    # retrieval results
    st.markdown('### Retrieved chunks')
    for r in results:
        st.markdown(f"""
        <div class="result-card">
            <div class="score">Score: {r['score']:.4f} &nbsp;|&nbsp; Page {r['page']}</div>
            <div class="text">{r['text'][:500]}</div>
        </div>
        """, unsafe_allow_html=True)

    # RAG answer with validation
    if use_rag:
        st.markdown('### Generated answer')
        with st.spinner(f'Generating answer with ...'):
            try:
                result = engine.answer_with_validation(query, top_k=top_k)
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # validation results
                with st.expander('🔍 Answer validation', expanded=False):
                    st.markdown('**Validation result:**')
                    st.text(result['validation'])
                    st.markdown(f'**Chunks used:** {result["chunks_used"]}')
                    
            except Exception as e:
                st.error(f'Ollama error: {e} — make sure `ollama serve` is running in a separate terminal.')