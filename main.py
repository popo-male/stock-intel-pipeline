from src.db.connection import close_pool
from src.core.config import load_config
from src.core.logger import logger
from src.db.repository import setup_database
from src.nlp.analyzer import Analyzer
from src.nlp.llm_processor import LLMProcessor
from src.scraper.fetcher import run_scraper


def run() -> None:
    logger.info("News Ingestion Pipeline Starting...")
    config = load_config()

    try:
        setup_database()
        run_scraper(config)

        # nlp processing (sentiment)
        analyzer = Analyzer()
        analyzer.process_unscored_articles()

        # llm summaries
        llm_processor = LLMProcessor()
        llm_processor.process_unsummarized_articles()

        logger.info("Pipeline execution complete!")
    except Exception as exc:
        logger.error(f"Unhandled pipeline error: {exc}", exc_info=True)
    finally:
        close_pool()


if __name__ == "__main__":
    run()
