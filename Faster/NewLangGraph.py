def manger_node(state):
    task_input = state.get("task","")
    input = state.get("input","")
    prompt = f""" You are an expert code generator.
    Your are a task router, Based on the user request below, decide whether it is a:
    -translate
    -summarize
    -calculate

    Respond with only one word (translate, summarize, calculate).

    Task:(task_input)
    """
    decision = llm.invoke(prompt),content.strip().lower()
    return {"agents":decision, "input":input}

def translate_node(state):
    text = state.get("input","")
    prompt = f"Act like you a translator. only respond with English translation of the next below:\n\n{text}"
    result = llm.invoke(prompt).content
    return {"result":result}

def summarizer_node(state):
    text = state.get("input","")
    prompt = f"Summarize the following in 1-2 lines:\n\n{text}"
    result = llm.invoke(prompt).content
    return {"result":result}
      
def calculator_node(state):
    text = state.get("input","")
    prompt = f"Please calculate and return of:\n\n{text}"
    result = llm.invoke(prompt).content
    return {"result":result}

def default_node(state):
    return {"result":"Sorry, I could not understand the task."}

g=StateGraph(dict)

g.add_node("manager",manger_node)
g.add_node("translator",translate_node)
g.add_node("summarizer",summarizer_node)
g.add_node("calculator",calculator_node)
g.add_node("default",default_node)

g.set_entry_point("manager")
g.add_conditional_edge("manager",route_by_agent)

g.set_finish_point("Translator")
g.set_finish_point("Summarizer")
g.set_finish_point("Calculator")
g.set_finish_point("Default")

graph = g.compile()

def route_by_agent(state):
    return{
        "translate":"translator",
        "summarize":"summarizer",
        "calculate":"calculator",
        "default":"default",
        "input": state.get("input","")
    }.get(state.get("agents",""),"default")

def default_node(state):
    return {"result":"Sorry, I could not understand the task."}

from langgraph.graph import StateGraph

g = StateGraph(dict)
g.add_node("Manager", manger_node)
g.add_node("Translator", translate_node)
g.add_node("Summarizer", summarizer_node)
g.add_node("Calculator", calculator_node)
g.add_node("Default", default_node)
g.set_entry_point("Manager")
g.add_conditional_edge("Manager", route_by_agent)
g.set_finish_point("Translator")
g.set_finish_point("Summarizer")
g.set_finish_point("Calculator")
g.set_finish_point("Default")
graph = g.compile()

raspcal = graph.invoke({
    "task": "what is 12 * 8 + 5",
    "input": "12 * 8 + 5"
})

print(respcal['result'])

print(graph.invoke({  
    
    "task": "Please translate to English: Bonjour tout le monde",
    "input":"foo",
    }))
    

