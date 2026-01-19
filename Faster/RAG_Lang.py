import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.vectorstores import FAISS

# --- Load HuggingFace token ---
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Example documents ---
texts = [
    "LangChain helps developers build LLM applications.",
    "FAISS is used for vector similarity search.",
    "Chat history must be manually maintained in LangChain 1.1.",
    "Retrievers are used in RAG pipelines.",
    "HuggingFace embeddings create vector representations."
]

# --- Embeddings + Vector DB ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(texts, embeddings)
retriever = db.as_retriever()

# --- Prompt with memory placeholder ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the retrieved context to answer the user."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{question}\n\nContext:\n{context}")
])

# --- HuggingFace LLM ---
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",  # ✅ pick a chat-capable HF model
        temperature=0.2,
        max_new_tokens=512
    )
)

# --- Chain ---
chain = prompt | llm

# --- Chat history ---
history = InMemoryChatMessageHistory()

def ask(question: str) -> str:
    # Retrieve context documents
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])

    # Add human message to history
    history.add_message(HumanMessage(content=question))

    # Run the chain with history
    response = chain.invoke({
        "context": context,
        "question": question,
        "messages": history.messages
    })

    # Add AI reply to history
    history.add_message(AIMessage(content=response.content))

    return response.content

# --- Demo ---
print("User: What is FAISS?")
print("AI:", ask("What is FAISS?"))

print("\nUser: What did I ask earlier?")
print("AI:", ask("What did I ask earlier?"))

print("\nUser: How does LangChain handle memory?")
print("AI:", ask("How does LangChain handle memory?"))


