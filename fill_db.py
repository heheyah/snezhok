import json
import chromadb
from chromadb.utils import embedding_functions

# 1. Connect to your database
client = chromadb.PersistentClient(path="data/chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 2. Recreate the collection to start fresh
try:
    client.delete_collection(name="protocols")
except Exception:
    pass
collection = client.create_collection(name="protocols", embedding_function=ef)

# 3. POINT THIS TO YOUR CORPUS FILE
# Make sure this path is exactly where your .jsonl file is located!
CORPUS_FILE = "data/corpus/protocols_corpus.jsonl" 

documents = []
ids = []
metadatas = []

print(f"Reading {CORPUS_FILE}...")

try:
    # 4. Read the JSONL file line by line
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Skip empty lines
            if not line.strip(): 
                continue
                
            data = json.loads(line)
            
            # Extract the medical text. (Adjust the key "text" if your JSON uses "content" or something else)
            doc_text = data.get("text", str(data))
            
            # Create a unique ID for each protocol
            doc_id = str(data.get("id", f"protocol_{i}"))
            
            # Optional: Save the ground truth if the corpus provides it, otherwise save the source ID
            gt_code = data.get("gt", data.get("icd_code", "unknown"))
            
            documents.append(doc_text)
            ids.append(doc_id)
            metadatas.append({"id": doc_id, "gt": gt_code})
            
    # 5. Save everything to ChromaDB
    if documents:
        print(f"Loaded {len(documents)} protocols from the file.")
        print("Adding to the database... (This might take a minute or two depending on the file size)")
        
        # We add them to the database
        collection.add(
            documents=documents, 
            ids=ids,
            metadatas=metadatas
        )
        print("✅ Success! Your database is now full of medical protocols.")
    else:
        print("❌ The file was read, but no documents were found inside.")

except FileNotFoundError:
    print(f"❌ Could not find the file: {CORPUS_FILE}. Please check the path and try again!")