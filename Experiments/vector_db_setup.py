import json
import sys
import time
from pathlib import Path
sys.path.append('/content/ConvFinQA')
from retrieve import chromadb_client, sentence_transformer_ef
from utils import format_document
from config import DATA_PATH, COLLECTION_NAME
def parse_convfinqa_dataset(filepath, limit=None):
    if not Path(filepath).exists():
        raise FileNotFoundError(f"The file at {filepath} was not found.")
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    if not isinstance(data, list):
        raise TypeError("Parsed data must be a list of entries.")
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Limit must be a positive integer.")
        data = data[:limit]
    docs = []
    for entry in data:
        try:
            doc = format_document(entry)
            docs.append(doc)
        except Exception as e:
            print(f"Skipping entry due to formatting error: {e}")
    return docs
def main():
    start_time = time.time()

    try:
        chromadb_client.delete_collection(name=COLLECTION_NAME)
        chromadb_client.clear_system_cache()
    except ValueError:
        pass
    db = chromadb_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    docs = parse_convfinqa_dataset(DATA_PATH, limit=None)
    batch_size = 500
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        ids = [doc.id for doc in batch]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        db.add(ids=ids, documents=texts, metadatas=metadatas)
        print(f"Inserted {i + len(batch)} / {len(docs)} documents")
    print(f"\nTotal documents inserted: {len(docs)}")
    try:
        sample = db.get(ids=[docs[0].id])
        print(json.dumps(sample, indent=2))
    except Exception as e:
        print(f"Failed to fetch sample document: {e}")
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")
  if __name__ == "__main__":
    main()
