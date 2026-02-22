import chromadb
from chromadb.utils import embedding_functions

# 1. Connect to your database
# (Make sure this path matches exactly where your database folder is)
client = chromadb.PersistentClient(path="data/chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    collection = client.get_collection(name="protocols", embedding_function=ef)
    
    # 2. Run a fake patient symptom query
    test_symptoms = "hard to pee"
    results = collection.query(query_texts=[test_symptoms], n_results=3)
    
    print("\n✅ DATABASE WORKS! Here is what it found:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n--- Document {i+1} ---")
        print(f"{doc[:300]}...") # Print the first 300 characters
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Your database might still be empty or the collection name is wrong.")