import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.prebuilt import create_react_agent   # use LangGraph's version
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.tools import StructuredTool
from langgraph_supervisor import create_supervisor

# Load Hugging Face token from .env
load_dotenv()

# LLM setup (no token argument, it uses HUGGINGFACEHUB_API_TOKEN from env)
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b"
    )
)

# Tools
def search_ddgo(query: str):
    """Wikipedia search"""
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    return wiki_tool.run({"query": query})

def add(a: float, b: float):
    """Adding"""
    return a + b

def multiply(a: float, b: float):
    """Multiplying"""
    return a * b

# Wrap tools
add_tool = StructuredTool.from_function(add)
multiply_tool = StructuredTool.from_function(multiply)
search_tool = StructuredTool.from_function(search_ddgo)

# Agents
math_agent = create_react_agent(
    model=llm,
    tools=[add_tool, multiply_tool],
    name="math_agent",
    prompt="You are a math agent. Always use one tool at a time"
)

research_agent = create_react_agent(
    model=llm,
    tools=[search_tool],
    name="res_agent",
    prompt="You are a world-class researcher with access to web search. Do not do any math"
)

# Supervisor
workflow = create_supervisor(
    [research_agent, math_agent],
    model=llm,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use res_agent. For math tasks, use math_agent."
    )
)

app = workflow.compile()

# Run workflow
result = app.invoke({
    "messages": [{"role": "user", "content": "what is quantum computing ?"}]
})

print(result)

for m in result['messages']:
    m.pretty_print()
