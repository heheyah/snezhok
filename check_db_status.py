import chromadb

# 1. Connect
client = chromadb.PersistentClient(path="data/chroma_db")

# 2. List all collections found
collections = client.list_collections()
print(f"Collections found: {[c.name for c in collections]}")

if collections:
    # 3. Check the count of the first collection
    col_name = collections[0].name
    collection = client.get_collection(name=col_name)
    print(f"Collection '{col_name}' has {collection.count()} items.")
    
    # 4. Show a sample
    if collection.count() > 0:
        sample = collection.get(limit=1)
        print(f"Sample Document: {sample['documents'][0][:100]}...")
else:
    print("❌ No collections found! Your path 'data/chroma_db' might be empty.")