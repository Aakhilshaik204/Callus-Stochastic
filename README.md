# Enterprise Document Q&A AI Agent (Custom RAG)

An enterprise-grade AI agent that implements a **Custom RAG Pipeline** using **Google Gemini 1.5** and **ChromaDB**.

## Features
- **RAG Architecture**: Uses Retrieval-Augmented Generation to answer questions grounded in your documents.
- **Custom Pipeline**: 
    - **Ingestion**: PDF -> Text -> Chunks -> Embeddings (`text-embedding-004`).
    - **Storage**: Local Vector Store (`ChromaDB`).
    - **Retrieval**: Semantic Search.
    - **Generation**: Gemini 1.5 Flash.
- **Tools**: Integrated Arxiv search.
- **Streamlit UI**: Clean interface.

## Setup

1.  **Clone the repository** (if applicable) or navigate to the project folder.
    ```bash
    cd Callus
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    - Open `.env` file.
    - Add your Google Gemini API Key:
      ```
      GOOGLE_API_KEY=your_key_here
      ```

## Usage

1.  **Run the Application**
    ```bash
    streamlit run src/app.py
    ```

2.  **Interact**
    - **Upload**: Upload PDF research papers. The app will chunk and index them into `chroma_db` folder.
    - **Chat**: Ask questions. The system will retrieve relevant chunks and generate an answer.

## Directory Structure
- `src/`: Source code.
    - `ingestion.py`: RAG Ingestion (Chunking + ChromaDB).
    - `engine.py`: RAG Retrieval & Generation.
- `chroma_db/`: Local vector store persistence.
