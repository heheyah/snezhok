# Create a quick script test_rag.py
import chroma_db
from chroma_db.utils import embedding_functions

client = chroma_db.PersistentClient(path="data/chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name="protocols", embedding_function=ef)

# Test a symptom manually
query = "High fever, stiff neck, and light sensitivity"
results = collection.query(query_texts=[query], n_results=2)

print("TOP MATCHES:")
for doc in results['documents'][0]:
    print(f"- {doc[:200]}...") # Print first 200 chars