from core.config import load_config
from db.repository import setup_database
from nlp.analyzer import Analyzer
from nlp.llm_processor import LLMProcessor
from scraper.fetcher import run_scraper


def run() -> None:
    config = load_config()

    # setup db
    setup_database()

    print("Stock Intelligence Platform Initializing...")

    # ingest data
    run_scraper(config)

    # nlp processing (sentiment)
    analyzer = Analyzer()
    analyzer.process_unscored_articles()

    # llm summaries
    llm_processor = LLMProcessor()
    llm_processor.process_unsummarized_articles()

    print("Pipeline execution complete!")


if __name__ == "__main__":
    run()
