from rag.document_loader import load_and_split_documents
from rag.vector_store import build_vector_store

if __name__ == "__main__":
    docs = load_and_split_documents("data/Help-Guide.pdf")
    build_vector_store(docs)
    print("✅ Vector DB built and uploaded to Pinecone.")
