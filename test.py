from qdrant_client import QdrantClient
from config import QDRANT_HOST, QDRANT_PORT

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
collections = client.get_collections()
print(f"Connecting to: {QDRANT_HOST}:{QDRANT_PORT}")
print(f"Collections found: {collections}")
