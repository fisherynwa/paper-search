import hydra
from omegaconf import DictConfig, OmegaConf
from paper_search_engine import PaperSearchEngine
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # build engine
    engine = PaperSearchEngine(
        model_name      = cfg.model.embedding,
        pdf_url         = cfg.pdf_url,
        chunk_size      = cfg.retrieval.chunk_size,
        overlap         = cfg.retrieval.overlap,
        max_length      = cfg.retrieval.max_length,
        use_contextual  = cfg.retrieval.use_contextual,
        ollama_model    = cfg.model.ollama,
        ollama_validator= cfg.model.validator,
        prompts         = OmegaConf.to_container(cfg.generation.prompts, resolve=True),
    )

    gen_options = {
        "temperature":    cfg.generation.temperature,
        "top_p":          cfg.generation.top_p,
        "repeat_penalty": cfg.generation.repeat_penalty,
        "seed":           cfg.generation.seed,
    }

    # example queries
    queries = [
        "How does XGBoost handle missing values?",
        "What is the regularized objective function in XGBoost?",
        "How does XGBoost achieve parallelism?",
    ]

    for query in queries:
        log.info(f"\nQuery: {query}")
        result = engine.answer_with_validation(
            query,
            top_k      = cfg.retrieval.top_k,
            temperature= cfg.generation.temperature,
        )
        log.info(f"Answer:\n{result['answer']}")
        log.info(f"Validation: {result['validation']}")
        log.info(f"Chunks used: {result['chunks_used']}")
        print("-" * 70)


if __name__ == "__main__":
    main()