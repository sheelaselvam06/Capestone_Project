import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

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


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc_embeddings = [embeddings.embed_query(t) for t in texts]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{question}")
])


llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.2,
        max_new_tokens=512
    )
)

# --- Chain ---
chain = prompt | llm

# --- Chat history ---
history = InMemoryChatMessageHistory()

def ask(question: str) -> str:
    # Compute embedding for the question
    q_embedding = embeddings.embed_query(question)

    # (Optional) do a simple similarity check with documents
    sims = [sum(qe * de for qe, de in zip(q_embedding, doc)) for doc in doc_embeddings]
    best_idx = sims.index(max(sims))
    context = texts[best_idx]  # pick the most similar doc

    # Add human message
    history.add_message(HumanMessage(content=question))

    # Run chain with context
    response = chain.invoke({
        "question": f"{question}\n\nRelevant context: {context}",
        "messages": history.messages
    })

    # Add AI reply
    history.add_message(AIMessage(content=response.content))

    return response.content

# --- Demo ---
print("User: What is FAISS?")
print("AI:", ask("What is FAISS?"))

print("\nUser: What did I ask earlier?")
print("AI:", ask("What did I ask earlier?"))

print("\nUser: How does LangChain handle memory?")
print("AI:", ask("How does LangChain handle memory?"))


  
