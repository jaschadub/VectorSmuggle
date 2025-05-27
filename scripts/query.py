import os
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS, Qdrant, Pinecone
from langchain.chains import RetrievalQA

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB = os.getenv("VECTOR_DB", "faiss")

llm = OpenAI(openai_api_key=OPENAI_API_KEY)

if VECTOR_DB == "faiss":
    db = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
elif VECTOR_DB == "qdrant":
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    db = Qdrant(
        collection_name="rag-exfil-poc",
        embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        url=qdrant_url
    )
elif VECTOR_DB == "pinecone":
    import pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
    db = Pinecone.from_existing_index(
        index_name="rag-exfil-poc",
        embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    )
else:
    raise ValueError("Unsupported VECTOR_DB type")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

while True:
    query = input("Ask your question: ")
    if query.lower() in ("exit", "quit"): break
    print("→", qa.run(query))
