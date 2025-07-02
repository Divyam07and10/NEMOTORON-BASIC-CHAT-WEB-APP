import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LangchainPinecone
import open_clip
import torch

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone SDK
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize OpenCLIP model for embeddings
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Custom embedding function for OpenCLIP
def embed_text(texts):
    with torch.no_grad():
        # Tokenize and compute embeddings
        tokens = tokenizer(texts)
        embeddings = model.encode_text(tokens).cpu().numpy()
    return embeddings

# Wrapper for LangChain compatibility
class OpenCLIPEmbeddings:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

    def embed_documents(self, texts):
        return embed_text(texts).tolist()

    def embed_query(self, text):
        return embed_text([text])[0].tolist()

# Initialize embedding model
embedding_model = OpenCLIPEmbeddings()

def ensure_index():
    if PINECONE_INDEX not in pc.list_indexes().names():
        print("🧠 Creating index in aws us-east-1...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=512,  # Matches ViT-B-32 embedding size
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    else:
        print(f"✅ Index '{PINECONE_INDEX}' already exists.")

def build_vector_store(documents):
    ensure_index()
    # Initialize vector store with LangchainPinecone
    vectorstore = LangchainPinecone.from_texts(
        texts=[doc.page_content for doc in documents],
        embedding=embedding_model,
        index_name=PINECONE_INDEX,
        text_key="text",
        metadatas=[{"text": doc.page_content} for doc in documents]
    )
    print("✅ Documents embedded and added to Pinecone.")

def get_context_relevant_to(query, k=3):
    vectorstore = LangchainPinecone.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embedding_model,
        text_key="text"
    )
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])