import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load free embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Load document
with open("sample_document.txt", "r") as f:
    document = f.read()

# Step 2: Chunk document
chunks = [chunk.strip() for chunk in document.split("\n") if chunk.strip()]

# Step 3: Generate embeddings locally
embeddings = model.encode(chunks)

# Convert to numpy float32
embedding_matrix = np.array(embeddings).astype("float32")

# Step 4: Store in FAISS
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Step 5: Ask user question
query = input("Ask your question: ")

# Convert question to embedding
query_vector = model.encode([query])
query_vector = np.array(query_vector).astype("float32")

# Step 6: Retrieve most similar chunk
_, I = index.search(query_vector, k=1)
retrieved_chunk = chunks[I[0][0]]

# Step 7: Print answer
print("\nMost Relevant Answer:")
print(retrieved_chunk)