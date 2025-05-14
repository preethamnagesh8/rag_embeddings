from langchain.vectorstores import FAISS
from langchain.schema import Document


def create_vectorstore(chunks, embeddings):
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
