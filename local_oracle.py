import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from settings import settings

def get_retriever(force_reload=False):
    """
    Returns a retriever with Incremental Sync capabilities.
    Only processes files that aren't already in the database.
    """
    embeddings = OllamaEmbeddings(model=settings.config['indexing']['embedding_model'])
    persist_dir = "./chroma_db"
    data_dir = settings.DATA_PATH

    # --- 1. EMERGENCY RESET (Optional) ---
    if force_reload and os.path.exists(persist_dir):
        print(f"--- QA ALERT: force_reload=True. Wiping entire database... ---")
        shutil.rmtree(persist_dir)

    # --- 2. LOAD OR CREATE DATABASE ---
    if os.path.exists(persist_dir):
        print("--- Checking existing database for updates... ---")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

        # Get list of all files currently in the database metadata
        existing_data = vectorstore.get()
        # Extract unique 'source' paths from metadata
        indexed_sources = {meta['source'] for meta in existing_data['metadatas']}
    else:
        print("--- No database found. Initializing fresh store... ---")
        vectorstore = None
        indexed_sources = set()

    # --- 3. IDENTIFY NEW FILES ONLY ---
    all_pdfs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    # Compare absolute paths to find what's missing
    new_files = [f for f in all_pdfs if f not in indexed_sources]

    if not new_files:
        print("--- Everything is up to date. No new files to process. ---")
    else:
        print(f"--- Found {len(new_files)} new files. Starting incremental sync... ---")
        new_docs = []

        for file_path in new_files:
            try:
                print(f"  -> Loading: {os.path.basename(file_path)}")
                loader = PyMuPDFLoader(file_path)
                new_docs.extend(loader.load())
            except Exception as e:
                print(f"  [!] SKIP CULPRIT: {os.path.basename(file_path)} | Error: {str(e)[:50]}")

        # --- 4. CHUNKING & UPSERTING ---
        if new_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            new_splits = text_splitter.split_documents(new_docs)

            print(f"--- Embedding {len(new_splits)} new chunks... (Wait) ---")

            if vectorstore is not None:
                # Add to existing database
                vectorstore.add_documents(new_splits)
                print(f"--- Sync Complete! Added {len(new_splits)} chunks. ---")
            else:
                # Create for the first time
                vectorstore = Chroma.from_documents(
                    documents=new_splits,
                    embedding=embeddings,
                    persist_directory=persist_dir
                )
                print("--- Initial Database Created. ---")

    # This avoids the 'fetch_k' error we saw earlier
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 15, 'lambda_mult': 0.25}  # Lower lambda = more diversity
    )

def get_production_retriever(vectorstore):
    # 1. The Base Librarian (MMR for diversity)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 50,  # Fetch a wide net
            "fetch_k": 100,  # Initial pool for MMR
            "lambda_mult": 0.5  # Balanced diversity
        }
    )

    # 2. The Logic Filter (Reranker)
    compressor = FlashrankRerank(top_n=15)

    # 3. The Unified Pipeline
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )