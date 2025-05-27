import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Qdrant, Pinecone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB = os.getenv("VECTOR_DB", "faiss")  # Options: faiss, qdrant, pinecone

# Load and split document
loader = PyPDFLoader("../internal_docs/strategic_roadmap.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Generate embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if VECTOR_DB == "faiss":
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    print("Saved FAISS index to ./faiss_index")

elif VECTOR_DB == "qdrant":
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    db = Qdrant.from_documents(
        chunks,
        embeddings,
        url=qdrant_url,
        collection_name="rag-exfil-poc"
    )
    print("Uploaded documents to Qdrant at", qdrant_url)

elif VECTOR_DB == "pinecone":
    import pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
    index_name = "rag-exfil-poc"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    db = Pinecone.from_documents(
        chunks,
        embeddings,
        index_name=index_name
    )
    print("Uploaded documents to Pinecone index:", index_name)

else:
    print("Unsupported VECTOR_DB type.")
