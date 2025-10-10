from academic_paper_tool.wrappers.search_and_select import search_and_select_papers
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    papers = search_and_select_papers("langgraph")
    for p in papers:
        print(p.id, "-", p.title)