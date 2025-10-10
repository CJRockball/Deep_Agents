# scripts/test_download_process.py
"""
Simple test for the download_and_process_papers wrapper function
"""
import logging
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from academic_paper_tool.wrappers.search_and_select import search_and_select_papers
from academic_paper_tool.wrappers.download_and_process import download_and_process_papers, get_pipeline_summary

def main():
    """Test the complete download-and-process pipeline and print evaluation metrics"""
    # Example topic
    topic = "langgraph"
    max_papers = 3

    logger.info(f"Starting test for topic: '{topic}' (max_papers={max_papers})")
    start_time = datetime.now()

    # Step 1: Search and select
    papers = search_and_select_papers(topic, max_papers=max_papers)
    if not papers:
        logger.error("No papers returned by search_and_select_papers.")
        return
    logger.info(f"Selected papers: {[p.id for p in papers]}")

    # Step 2: Download and process through full pipeline
    result = download_and_process_papers(papers)

    # Print overall success
    status = "SUCCESS" if result.success else "FAILURE"
    logger.info(f"Pipeline execution status: {status}")

    # Print summary
    summary_text = get_pipeline_summary(result)
    print("\n" + summary_text + "\n")

    # Detailed per-paper evaluation
    for paper_id, details in result.paper_results.items():
        print(f"Paper ID: {paper_id}")
        print(f"  Success: {details['success']}")
        print(f"  Processing Time: {details['processing_time']:.1f}s")
        print(f"  Errors: {details['errors']}" if details['errors'] else "  No errors")
        print("  Stages:")
        for stage, info in details['stages'].items():
            print(f"    {stage}: success={info['success']}, time={info['execution_time']:.1f}s, "
                  f"data_created={info['data_created']}, quality_score={info['quality_score']:.1f}")
        print()

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Test completed in {duration:.1f} seconds")

if __name__ == "__main__":
    main()
