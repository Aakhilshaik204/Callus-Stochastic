import google.generativeai as genai
import arxiv
from src.config import configure_genai
from src.ingestion import get_chroma_collection

configure_genai()

# --- Tools ---
def search_arxiv(query: str):
    """
    Searches Arxiv for research papers based on a query.
    Returns the title, authors, summary, and PDF URL.
    """
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for result in search.results():
        paper_info = (
            f"Title: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}\n"
            f"PDF URL: {result.pdf_url}\n"
        )
        results.append(paper_info)
    
    return "\n\n".join(results)

tools = [search_arxiv]

# --- RAG Logic ---

def query_chromadb(query, n_results=15):
    """
    Embeds the query and retrieves N nearest chunks from ChromaDB.
    """
    collection = get_chroma_collection()
    
    # embed_content for query
    # task_type="retrieval_query" is important for asymmetric models
    model = "models/text-embedding-004"
    embedding_resp = genai.embed_content(
        model=model,
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = embedding_resp['embedding']
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Results structure is dict of lists.
    # documents[0] is the list of chunks for the first query.
    retrieved_chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    return retrieved_chunks, metadatas

def generate_rag_answer(query, retrieved_chunks):
    """
    Generates an answer using Gemini 1.5 Flash based on retrieved context.
    """
    context_str = "\n\n".join(retrieved_chunks)
    
    system_instruction = (
        "You are an expert AI Research Assistant. "
        "Answer the user's question based strictly on the provided Context below. "
        "If the answer is not in the context, say you don't know (or check Arxiv if relevant). "
        "Always cite the source if possible (from context metadata)."
    )
    
    prompt = (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    
    model = genai.GenerativeModel(
        model_name='gemini-3-flash-preview',
        tools=tools, # Tools can be used if the model decides to ignore context or needs external info
        system_instruction=system_instruction
    )
    
    # Use chat session to enable automatic function calling (execution loop)
    # The SDK handles the: Model -> Function Call -> Execute -> Model Response loop.
    chat = model.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(prompt)
    return response.text

# --- Interface ---

def get_rag_response(query):
    """
    Full RAG Pipeline: Query -> Retrieve -> Generate
    """
    # 1. Retrieve
    chunks, metadatas = query_chromadb(query)
    
    # 2. Generate
    if not chunks:
        # Fallback if DB is empty
        return "I don't have any documents indexed yet. Please upload some PDFs."
        
    answer = generate_rag_answer(query, chunks)
    return answer
