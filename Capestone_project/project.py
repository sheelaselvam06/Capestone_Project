import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


pdf_path = "/Capestone_Project/projectData.pdf"   # ensure this file is in your project folder
loader = PyPDFLoader(pdf_path)
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=50,chunk_overlap=5)
docs = splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(docs, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",  # chat-capable HF model
        temperature=0.2,
        max_new_tokens=512
    )
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the retrieved context to answer the user."),
    ("human", "{question}\n\nContext:\n{context}")
])

chain = prompt | llm

st.title("Capstone Project Q&A")
st.write("Ask questions about ProjectData.pdf")

question = st.text_input("Enter your question:")
if st.button("Submit"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Retrieve context
        docs = retriever.invoke(question)
        if not docs:
            st.error("Data not found in ProjectData.pdf")
        else:
            context = "\n".join([d.page_content for d in docs])
            response = chain.invoke({"question": question, "context": context})
            st.success(response.content)
