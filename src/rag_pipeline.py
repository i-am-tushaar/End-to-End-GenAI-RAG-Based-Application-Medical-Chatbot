from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ✅ Memory imports
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.config import INDEX_NAME, MODEL_NAME


# -------------------------------
# Session-based memory storage
# -------------------------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# -------------------------------
# Create RAG Chain
# -------------------------------
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

    # Gemini LLM
    chat_model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.3,
        max_retries=1   # prevents long retry delays
    )

    # ✅ Buffer Memory (FAST)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Prompt (history included)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    # QA chain
    qa_chain = create_stuff_documents_chain(
        chat_model,
        prompt,
        document_variable_name="context"
    )

    # RAG chain
    rag_chain = create_retrieval_chain(
        retriever,
        qa_chain
    )

    # ✅ Attach memory
    rag_chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return rag_chain_with_memory