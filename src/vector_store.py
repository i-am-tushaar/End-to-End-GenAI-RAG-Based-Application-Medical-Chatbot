# src/vector_store.py

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_file,
    text_split,
    download_hugging_face_embeddings,
)

from src.config import INDEX_NAME, PINECONE_API_KEY


# -------------------------------
# Create Pinecone Index
# -------------------------------
def create_pinecone_index():

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 embedding size
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Index '{INDEX_NAME}' created")
    else:
        print(f"✅ Index '{INDEX_NAME}' already exists")


# -------------------------------
# Ingest Documents
# -------------------------------
def ingest_documents(data_path: str):

    print("Loading documents...")
    documents = load_pdf_file(data_path)

    print("Splitting text...")
    text_chunks = text_split(documents)

    print("Loading embeddings...")
    embeddings = download_hugging_face_embeddings()

    print("Uploading to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print("✅ Documents successfully indexed")


if __name__ == "__main__":
    create_pinecone_index()
    ingest_documents("data/")
