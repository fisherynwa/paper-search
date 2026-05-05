import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
 
import numpy as np
import pytest
from unittest.mock import MagicMock
 
 
@pytest.fixture
def engine():
    """Minimal engine with fake FAISS and BM25 — no real I/O."""
    from paper_search_engine import PaperSearchEngine
 
    e = PaperSearchEngine.__new__(PaperSearchEngine)
    e.chunks = [
        {'text': 'XGBoost handles missing values using a sparsity-aware algorithm.', 'page': 4},
        {'text': 'The regularized objective combines a loss function with a penalty term.', 'page': 2},
        {'text': 'Column subsampling prevents overfitting and speeds up split evaluation.', 'page': 5},
    ]
    e.max_length = 128
 
    # fake embeddings matrix (3 chunks, 4-dim for simplicity)
    e.embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype='float32')
 
    # fake FAISS index
    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.array([[0, 1]], dtype='int64'),
        np.array([[0.9, 0.8]], dtype='float32')
    )
    e.index = mock_index
 
    # fake BM25
    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = np.array([0.5, 0.1, 0.3])
    e.bm25 = mock_bm25
 
    # fake tokenizer and model for _embed_text
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}
    e.tokenizer = mock_tokenizer
 
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.last_hidden_state = MagicMock()
    mock_output.last_hidden_state.__getitem__ = MagicMock(
        return_value=MagicMock(return_value=np.random.rand(1, 4).astype('float32'))
    )
    mock_model.return_value = mock_output
    e.model = mock_model
 
    return e
 
 
class TestSemanticSearch:
 
    def test_returns_correct_number_of_results(self, engine):
        results = engine._semantic_search('missing values', top_k=2)
        assert len(results) == 2
 
    def test_returns_list_of_tuples(self, engine):
        results = engine._semantic_search('missing values', top_k=3)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
 
    def test_calls_faiss_index_search(self, engine):
        engine._semantic_search('missing values', top_k=2)
        assert engine.index.search.called
 
 
class TestBM25Search:
 
    def test_returns_correct_number_of_results(self, engine):
        results = engine._bm25_search('missing values', top_k=2)
        assert len(results) == 2
 
    def test_results_sorted_by_score_descending(self, engine):
        results = engine._bm25_search('missing values', top_k=3)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
 
    def test_query_lowercased(self, engine):
        engine._bm25_search('Missing Values XGBoost', top_k=2)
        called_with = engine.bm25.get_scores.call_args[0][0]
        assert called_with == ['missing', 'values', 'xgboost']
 
    def test_returns_list_of_tuples(self, engine):
        results = engine._bm25_search('missing values', top_k=2)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)