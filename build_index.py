import os
import json
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
CORPUS_DIR = "data/corpus"       # Where you unzipped corpus.zip
DB_DIR = "data/chroma_db"        # Where the database will be saved
CHUNK_SIZE = 500                 # Number of characters per text chunk
OVERLAP = 50                     # Overlap to prevent cutting sentences in half

def chunk_text(text, size, overlap):
    """Splits long protocol text into smaller, overlapping chunks."""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

def build_database():
    print("Initializing local ChromaDB...")
    # 1. Set up the local database and the embedding model
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # We use a fast, lightweight local embedding model from HuggingFace
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create a collection (like a table in a database)
    collection = client.get_or_create_collection(
        name="protocols", 
        embedding_function=sentence_transformer_ef
    )

    print("Reading and chunking clinical protocols...")
    documents = []
    metadatas = []
    ids = []
    
    # 2. Process each JSON file in the corpus
    for filename in os.listdir(CORPUS_DIR):
        if not filename.endswith(".json"):
            continue
            
        filepath = os.path.join(CORPUS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            protocol = json.load(f)
            
            # Extract the necessary fields based on the challenge data format
            p_id = protocol.get("protocol_id", filename)
            text = protocol.get("text", "")
            icd_codes = protocol.get("icd_codes", [])
            title = protocol.get("title", "")
            
            # 3. Chunk the text
            chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                # Save the ICD codes and Title as metadata so we can return them later!
                metadatas.append({
                    "protocol_id": p_id,
                    "title": title,
                    "icd_codes": json.dumps(icd_codes) # ChromaDB requires metadata to be strings or ints
                })
                ids.append(f"{p_id}_chunk_{i}")

    # 4. Save to the database
    print(f"Adding {len(documents)} chunks to the database. This might take a few minutes...")
    
    # ChromaDB has a batch limit, so we add in batches of 5000
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"Processed batch {i} to {i+batch_size}")

    print(f"Database successfully built at {DB_DIR}!")

if __name__ == "__main__":
    build_database()