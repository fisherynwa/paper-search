import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)


import json
import hashlib
from pathlib import Path
from datetime import datetime

import hydra
import ollama
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from paper_search_engine import PaperSearchEngine


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def retrieval_hit(results, relevant_pages):
    """Check if any relevant page appears in retrieved chunks."""
    retrieved_pages = {r['page'] for r in results}
    hits = retrieved_pages & set(relevant_pages)
    return len(hits) / len(relevant_pages) if relevant_pages else 0.0


def score_answer(engine, question, answer, ground_truth, context):
    prompt = (
        f"### Role\n"
        f"You are an expert auditor for Retrieval-Augmented Generation (RAG) systems.\n\n"
        f"### Input Data\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Generated Answer: {answer}\n\n"
        f"### Evaluation Rubric\n"
        f"1. Faithfulness: 5 = All claims in answer are backed by Context. 1 = Answer contains claims not in Context.\n"
        f"2. Completeness: 5 = Answer covers all key points in Ground Truth. 1 = Answer misses the main point.\n"
        f"3. Clarity: 5 = Logical, easy to read. 1 = Unintelligible.\n\n"
        f"### Instructions\n"
        f"Analyze the answer step-by-step. First, identify any discrepancies. "
        f"Then, provide the final scores in the following JSON format:\n"
        f'{{"reasoning": "...", "faithfulness": X, "completeness": X, "clarity": X}}'
    )
    response = ollama.chat(
        model=engine.ollama_validator,
        messages=[{'role': 'user', 'content': prompt}],
        options={"temperature": 0.0}
    )
    try:
        text = response['message']['content'].strip()
        text = text.replace('```json', '').replace('```', '').strip()
        start = text.find('{')
        end   = text.rfind('}') + 1
        if start != -1 and end > start:
            text = text[start:end]
        return json.loads(text)
    except Exception as e:
        print(f"Parse error: {e}\n  Raw: {text[:200]}")
        return {"faithfulness": 0, "completeness": 0, "clarity": 0, "issues": "parse error"}

def consistency_score(engine, question, top_k, temperature, n_runs=3):
    """Run the same question n times and measure answer variance."""
    answers = []
    for _ in range(n_runs):
        answer, _ = engine.chat(question, history=[], top_k=top_k, temperature=temperature)
        answers.append(answer.strip())
    # simple consistency: ratio of identical answers
    most_common = max(set(answers), key=answers.count)
    return answers.count(most_common) / n_runs, answers


# ------------------------------------------------------------------ #
# Auto-generation
# ------------------------------------------------------------------ #

def auto_generate_benchmark(engine, n_questions=10):
    """Generate Q&A pairs from the paper using the LLM."""
    # sample chunks evenly across the paper
    step   = max(1, len(engine.chunks) // n_questions)
    sample = [engine.chunks[i] for i in range(0, len(engine.chunks), step)][:n_questions]

    questions = []
    for i, chunk in enumerate(tqdm(sample, desc="Generating Q&A pairs")):
        prompt = (
                f"Excerpt: {chunk['text']}\n\n"
                f"Task: Create a high-quality QA pair for a benchmark.\n"
                f"Requirements:\n"
                f"1. The question must be 'decontextualized' (do not use phrases like 'in this excerpt' or 'the author mentions').\n"
                f"2. The question must be answerable solely using the excerpt provided.\n"
                f"3. The answer should be concise and factually dense.\n"
                f"Respond ONLY with JSON: {{\"question\": \"...\", \"answer\": \"...\"}}"
            )
        response = ollama.chat(
            model=engine.ollama_model,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.0}
        )
        try:
            text = response['message']['content'].strip()
            text = text.replace('```json', '').replace('```', '').strip()
            qa   = json.loads(text)
            qa['id'] = f'auto_{i+1}'
            questions.append(qa)
        except Exception:
            continue

    return questions


# ------------------------------------------------------------------ #
# Preference data collection
# ------------------------------------------------------------------ #

def save_preference(question, chosen, rejected, context, out_dir):
    """Save a preference pair for future DPO training."""
    record = {
        "prompt":   question,
        "context":  context,
        "chosen":   chosen,
        "rejected": rejected,
        "timestamp": datetime.now().isoformat()
    }
    out_path = Path(out_dir) / "preferences.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


# ------------------------------------------------------------------ #
# Main benchmark runner
# ------------------------------------------------------------------ #

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    # build engine
    engine = PaperSearchEngine(
        model_name       = cfg.model.embedding,
        pdf_url          = cfg.pdf_url,
        chunk_size       = cfg.retrieval.chunk_size,
        overlap          = cfg.retrieval.overlap,
        max_length       = cfg.retrieval.max_length,
        use_contextual   = cfg.retrieval.use_contextual,
        ollama_model     = cfg.model.ollama,
        ollama_validator = cfg.model.validator,
        prompts          = OmegaConf.to_container(cfg.generation.prompts, resolve=True),
    )

    # ---- load or generate benchmark  #
    pdf_id     = cfg.pdf_url.split('/')[-1].replace('.pdf', '')
    bench_cfg  = OmegaConf.to_container(cfg.get("benchmark", {}), resolve=True)
    bench_path = Path(bench_cfg.get("path") or f"benchmarks/{pdf_id}.yaml")

    auto_generate = cfg.get("benchmark", {}).get("auto_generate", False)

    if auto_generate or not bench_path.exists():
        print(f"Auto-generating benchmark for {pdf_id}...")
        questions = auto_generate_benchmark(engine, n_questions=cfg.get("benchmark", {}).get("n_questions", 10))
        # save for reuse and manual review
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bench_path, 'w') as f:
            yaml.dump({"pdf_url": cfg.pdf_url, "questions": questions}, f)
        print(f"Saved benchmark to {bench_path} — review and correct answers before trusting results.")
    else:
        print(f"Loading static benchmark from {bench_path}...")
        with open(bench_path) as f:
            bench = yaml.safe_load(f)
        questions = bench['questions']

    # ---- run evaluation  #
    results_log = []
    pref_dir    = "preference_data"

    for q in tqdm(questions, desc="Evaluating"):
        question       = q['question']
        ground_truth   = q.get('answer', '')
        relevant_pages = q.get('relevant_pages', [])

        # retrieve + answer
        answer, chunks = engine.chat(
            question, history=[],
            top_k       = cfg.retrieval.top_k,
            temperature = cfg.generation.temperature,
        )
        context = '\n\n'.join([f'[Page {r["page"]}]: {r["text"]}' for r in chunks])

        # metrics
        ret_hit    = retrieval_hit(chunks, relevant_pages)
        scores     = score_answer(engine, question, answer, ground_truth, context)
        cons_score, cons_answers = consistency_score(
            engine, question,
            top_k=cfg.retrieval.top_k,
            temperature=cfg.generation.temperature,
            n_runs=3
        )

        result = {
            "id":            q.get('id', '?'),
            "question":      question,
            "ground_truth":  ground_truth,
            "answer":        answer,
            "retrieval_hit": round(ret_hit, 3),
            "faithfulness":  scores.get('faithfulness', 0),
            "completeness":  scores.get('completeness', 0),
            "clarity":       scores.get('clarity', 0),
            "consistency":   round(cons_score, 3),
            "issues":        scores.get('issues', ''),
        }
        results_log.append(result)

        # collect preference pairs where consistency < 1.0
        # (inconsistent answers = potential chosen/rejected pairs)
        if cons_score < 1.0 and len(set(cons_answers)) >= 2:
            save_preference(
                question  = question,
                chosen    = cons_answers[0],   # first answer as proxy for chosen
                rejected  = cons_answers[-1],  # last divergent answer as rejected
                context   = context,
                out_dir   = pref_dir
            )

        # print per-question result
        print(f"\n[{result['id']}] {question}")
        print(f"  Retrieval hit : {result['retrieval_hit']}")
        print(f"  Faithfulness  : {result['faithfulness']}/5")
        print(f"  Completeness  : {result['completeness']}/5")
        print(f"  Clarity       : {result['clarity']}/5")
        print(f"  Consistency   : {result['consistency']}")
        print(f"  Issues        : {result['issues']}")

    # ---- aggregate  #
    n = len(results_log)
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Questions evaluated : {n}")
    print(f"Avg retrieval hit   : {sum(r['retrieval_hit']  for r in results_log) / n:.3f}")
    print(f"Avg faithfulness    : {sum(r['faithfulness']   for r in results_log) / n:.2f}/5")
    print(f"Avg completeness    : {sum(r['completeness']   for r in results_log) / n:.2f}/5")
    print(f"Avg clarity         : {sum(r['clarity']        for r in results_log) / n:.2f}/5")
    print(f"Avg consistency     : {sum(r['consistency']    for r in results_log) / n:.3f}")

    # save full results
    run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()[:6]
    out_path    = Path("benchmark_results") / f"{run_id}_{config_hash}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            "config":  OmegaConf.to_container(cfg, resolve=True),
            "results": results_log,
            "summary": {
                "avg_retrieval_hit": sum(r['retrieval_hit'] for r in results_log) / n,
                "avg_faithfulness":  sum(r['faithfulness']  for r in results_log) / n,
                "avg_completeness":  sum(r['completeness']  for r in results_log) / n,
                "avg_clarity":       sum(r['clarity']       for r in results_log) / n,
                "avg_consistency":   sum(r['consistency']   for r in results_log) / n,
            }
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Preference data saved to {pref_dir}/preferences.jsonl")


if __name__ == "__main__":
    main()