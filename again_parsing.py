import logging
import os
import json
from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- 2. Manual JSON Loading Function (The Correct One) ---
def load():
    all_docs = []
    
    try:
        with open("product_details.json", "r") as f:
            reviews_data = json.load(f)
            
        for item in reviews_data:
            content = (
                f"Album Title: {item.get('Title', 'N/A')}\n"
                f"Reviewer: {item.get('Reviewer', 'N/A')}\n"
                f"Rating: {item.get('Rating', 'N/A')}/5\n"
                f"Review Text: {item.get('ReviewText', '')}"
            )
            metadata = {
                "AlbumId": item.get("AlbumId"),
                "Title": item.get("Title"),
                "Reviewer": item.get("Reviewer"),
                "Rating": item.get("Rating"),
                "source_file": "product_details.json"
            }
            doc = Document(page_content=content, metadata=metadata)
            all_docs.append(doc)
            
        log.info(f"Loaded {len(reviews_data)} documents from product_details.json")

    except Exception as e:
        log.error(f"Failed to load 'product_details.json': {e}")
        return []

    # --- Load Customer Feedback ---
    try:
        with open("customer_feedback.json", "r") as f:
            feedback_data = json.load(f)
            
        for item in feedback_data:
            content = (
                f"Customer Name: {item.get('FirstName', '')} {item.get('LastName', '')}\n"
                f"Customer ID: {item.get('CustomerId', 'N/A')}\n"
                f"Rating: {item.get('Rating', 'N/A')}/5\n"
                f"Comment: {item.get('Comment', '')}"
            )
            metadata = {
                "CustomerId": item.get("CustomerId"),
                "FirstName": item.get("FirstName"),
                "LastName": item.get("LastName"),
                "Rating": item.get("Rating"),
                "source_file": "customer_feedback.json"
            }
            doc = Document(page_content=content, metadata=metadata)
            all_docs.append(doc)

        log.info(f"Loaded {len(feedback_data)} documents from customer_feedback.json")

    except Exception as e:
        log.error(f"Failed to load 'customer_feedback.json': {e}")
        return []
        
    return all_docs

def main():
    log.info("--- STARTING KNOWLEDGE BASE CONSTRUCTION ---")
    
    all_docs = load()

    if not all_docs:
        log.error("No documents were loaded. Exiting.")
        return

    log.info(f"Loaded {len(all_docs)} total documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    log.info(f"Split into {len(splits)} text chunks.")

    log.info("Initializing Ollama embedding model...")
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        embeddings_model.embed_query("test query") 
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        return

    log.info("Embedding documents and building FAISS index...")
    vector_store = FAISS.from_documents(splits, embeddings_model)
    
    vector_store.save_local("json_faiss_index")
    
    log.info("\n--- KNOWLEDGE BASE CONSTRUCTION COMPLETE  ---")
    log.info(" FAISS index 'json_faiss_index' has been saved.")

if __name__ == "__main__":
    main()