import os
import time
import chromadb
import google.generativeai as genai
from pypdf import PdfReader
from chromadb import Documents, EmbeddingFunction, Embeddings
from src.config import configure_genai

# Ensure configuration is loaded
configure_genai()

# ChromaDB Setup
CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "document_qa_collection"

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom Embedding Function for ChromaDB using Google Gemini.
    """
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/text-embedding-004"
        # Gemini expects a list of strings
        # Note: Depending on the list size, we might need batching.
        # The free tier has limits, but batching 10-20 usually works well.
        
        embeddings = []
        # Batching prevents hitting payload limits
        # Reduced batch size to 5 and added sleep to avoid 429
        batch_size = 5
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            
            # Retry mechanism
            for attempt in range(3):
                try:
                    response = genai.embed_content(
                        model=model,
                        content=batch,
                        task_type="retrieval_document"
                    )
                    embeddings.extend(response['embedding'])
                    time.sleep(1) # Rate limit buffer
                    break
                except Exception as e:
                    if "429" in str(e) or "ResourceExhausted" in str(e):
                        print(f"Rate limit hit. Waiting {2**attempt}s...")
                        time.sleep(2**attempt)
                    else:
                        raise e
            else:
                 # If we exhausted retries
                 raise Exception("Failed to embed batch after retries due to Rate Limits.")
            
        return embeddings

def get_chroma_collection():
    """
    Returns the ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_function = GeminiEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    return collection

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Splits text into chunks.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks

def process_uploaded_file_rag(uploaded_file):
    """
    Saves file, chunks it, and adds to ChromaDB.
    """
    # Save locally
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 1. Extract Text
    print(f"Extracting text from {uploaded_file.name}...")
    full_text = extract_text_from_pdf(file_path)
    print(f"Extracted {len(full_text)} characters.")
    
    # 2. Chunk Text
    # Smaller chunks (500 chars) for better granularity
    chunks = chunk_text(full_text, chunk_size=500, overlap=50)
    print(f"Generated {len(chunks)} chunks.")
    
    # 3. Add to ChromaDB
    collection = get_chroma_collection()
    
    # Generate IDs
    ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": uploaded_file.name, "chunk_index": i} for i in range(len(chunks))]
    
    print("Embedding and adding to ChromaDB...")
    collection.upsert(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    print("Done.")
    return len(chunks)

def reset_knowledge_base():
    """
    Deletes and recreates the collection to plain slate.
    """
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        pass # Collection might not exist
    
    # Re-create (get_chroma_collection handles creation)
    get_chroma_collection()
    print("Knowledge base reset.")
