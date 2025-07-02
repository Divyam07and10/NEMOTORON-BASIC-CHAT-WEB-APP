from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(file_path="data/Help-Guide.pdf"):
    # Load PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(documents)
