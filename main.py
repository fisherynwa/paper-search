import argparse
import textwrap
from paper_search_engine import PaperSearchEngine

parser = argparse.ArgumentParser(description='Semantic search over a research paper.')
parser.add_argument('--pdf_url',    type=str, default='https://arxiv.org/pdf/1603.02754')
parser.add_argument('--model',      type=str, default='BAAI/bge-large-en-v1.5')
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--chunk_size', type=int, default=150)
parser.add_argument('--overlap',    type=int, default=30)
parser.add_argument('--top_k',      type=int, default=3)
args = parser.parse_args()

if __name__ == '__main__':

    print(f'Loading engine — {args.model}...')
    engine = PaperSearchEngine(
        model_name=args.model,
        pdf_url=args.pdf_url,
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )

    queries = [
        'how does XGBoost handle missing values?',
        'what is the regularized objective function in XGBoost?',
        'how does XGBoost achieve parallelism?'
    ]

    for query in queries:
        print(f'\n{"=" * 70}')
        print(f'Query: "{query}"')
        print(f'{"=" * 70}')

        # retrieval results
        for r in engine.query(query, top_k=args.top_k):
            print(f'  Score: {r["score"]:.4f}  |  Page: {r["page"]}')
            print(f'  {textwrap.fill(r["text"][:400], width=65)}')
            print('  ' + '-' * 65)

        # RAG answer
        print(f'\n  Answer: {textwrap.fill(engine.answer(query, top_k=args.top_k), width=65)}')