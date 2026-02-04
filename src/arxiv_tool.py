import arxiv
from llama_index.core.tools import FunctionTool

def search_arxiv_papers(query: str, max_results: int = 3):
    """
    Searches Arxiv for research papers based on a query.
    Returns the title, authors, summary, and PDF URL of the papers.
    
    Args:
        query (str): The search topic or description.
        max_results (int): Number of papers to return (default 3).
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for result in search.results():
        paper_info = (
            f"Title: {result.title}\n"
            f"Note: Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}\n"
            f"PDF URL: {result.pdf_url}\n"
        )
        results.append(paper_info)
    
    return "\n\n".join(results)

arxiv_tool = FunctionTool.from_defaults(
    fn=search_arxiv_papers,
    name="arxiv_search",
    description="Useful for searching for research papers on Arxiv based on a topic or description."
)
