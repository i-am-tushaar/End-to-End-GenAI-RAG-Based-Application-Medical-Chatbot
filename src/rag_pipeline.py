from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.config import INDEX_NAME, MODEL_NAME


def create_rag_chain():

    # embeddings
    embeddings = download_hugging_face_embeddings()

    # vector store
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # ✅ Gemini LLM
    chat_model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.3
    )

    # prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # chains
    qa_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain
