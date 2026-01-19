import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API token from environment variables
HUGGING_FACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_TOKEN

# Define a file path for storing chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Custom class for managing chat history
class CustomChatMessageHistory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, content):
        self.messages.append({"type": "user", "content": content})

    def add_ai_message(self, content):
        self.messages.append({"type": "ai", "content": content})

# Function to load chat history from a file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            data = json.load(file)
            history = CustomChatMessageHistory()
            history.messages = data
            return history
    return CustomChatMessageHistory()

# Function to save chat history to a file
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history.messages, file)

llmhg = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id='openai/gpt-oss-120b'
    )
)

# Load existing chat history
chat_history = load_chat_history()

# Add a new user message
chat_history.add_user_message("What is the capital of Russia?")

# Invoke the model with the chat history
response2 = llmhg.invoke([HumanMessage(content=msg["content"]) if msg["type"] == "user" else AIMessage(content=msg["content"]) for msg in chat_history.messages])

# Add the AI's response to the chat history
chat_history.add_ai_message(response2.content)

# Save the updated chat history
save_chat_history(chat_history)

# Print the response
print("Response from HuggingFaceEndpoint:", response2.content)