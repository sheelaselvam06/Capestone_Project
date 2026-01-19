import os
from dotenv import load_dotenv

load_dotenv()
os.getenv('HF_TOKEN')
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate


documents=["A christmas carol is a novella by Charles Dickens, first published in 1843. The story follows Ebenezer Scrooge, a miserly old",
           " THe story tells of sourdough miner Santiago's journey to the Egyptian pyramids after having a recurring dream of finding treasure there. Along the way",
           "He meets a series of characters who help him understand the importance of following his dreams and listening to his heart.",
           " The Great Gatsby is a novel by F. Scott Fitzgerald, published in 1925"
           ]    

query = "Who is the author of A christmas carol?"   
context = "\n\n".join(documents)
prompt = ChatPromptTemplate.from_template(
    """ You are an assistant that answers questions strictly using the provided context.

    Context: {context}

    Question: {question}

    if the answer is not in the context, say:
        "I don't know based on provided context."
    """,
)

llm = ChatHuggingFace(llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b'))

chain = prompt |llm
response = chain.invoke({
    "context":context,
    "question":"What is the significance of Christmas Eve in A Christmas Carol?"
})
print(response.content)


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document   
documents_for_spillters=[Document(page_content=doc) for doc in documents]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, 
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""]
)
chunks=text_splitter.split_documents(documents_for_spillters)
context_chunks = "\n\n".join([chunk.page_content for chunk in chunks])
print(context)
response2 = chain.invoke({"context": context_chunks, "question": "what is the christmas carol?"}
                         
)
















