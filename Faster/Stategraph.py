from typing import TypedDict
from langgraph.graph import StateGraph, END
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

# Use a non-interactive backend (avoids Tcl/Tk errors on Windows)
matplotlib.use("Agg")

class MyState(TypedDict):
    count: int

# Simple increment function
def increment(st: MyState) -> MyState:
    return {"count": st["count"] + 1}

# Build the graph
graph = StateGraph(MyState)
graph.add_node("increment", increment)
graph.set_entry_point("increment")
graph.add_edge("increment", END)

# Compile and run
app = graph.compile()
result = app.invoke({"count": 3})
print("Result:", result)

# --- Visualization using networkx ---
G = nx.DiGraph()
for node in graph.nodes:
    G.add_node(node)
for edge in graph.edges:
    G.add_edge(edge[0], edge[1])

# Draw the graph and save to file
plt.figure(figsize=(6, 4))
nx.draw(
    G,
    with_labels=True,
    node_color="lightblue",
    node_size=2000,
    font_size=12,
    arrows=True
)
plt.title("LangGraph StateGraph")
plt.savefig("graph.png")   # saves the graph as an image
print("Graph saved as graph.png — open it to view.")

from typing import TypedDict
from langgraph.graph import StateGraph, END
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Use a non-interactive backend (avoids Tcl/Tk errors on Windows)
matplotlib.use("Agg")

class MyState(TypedDict):
    count: int

# Node functions
def increment(st: MyState) -> MyState:
    return {"count": st["count"] + 1}

def decrement(st: MyState) -> MyState:
    return {"count": st["count"] - 1}

def double(st: MyState) -> MyState:
    return {"count": st["count"] * 2}

# Build the graph
graph = StateGraph(MyState)
graph.add_node("increment", increment)
graph.add_node("decrement", decrement)
graph.add_node("double", double)

# Entry point and edges
graph.set_entry_point("increment")
graph.add_edge("increment", "double")
graph.add_edge("double", "decrement")
graph.add_edge("decrement", END)

# Compile and run
app = graph.compile()
result = app.invoke({"count": 3})
print("Result:", result)

# --- Visualization using networkx ---
G = nx.DiGraph()
for node in graph.nodes:
    G.add_node(node)
for edge in graph.edges:
    G.add_edge(edge[0], edge[1])

plt.figure(figsize=(6, 4))
nx.draw(
    G,
    with_labels=True,
    node_color="lightblue",
    node_size=2000, 
    font_size=12,
    arrows=True
)
plt.title("LangGraph StateGraph")

# Save the graph as PNG
output_file = "graph.png"
plt.savefig(output_file)
print(f"Graph saved as {output_file}")

# --- Display with Pillow ---
img = Image.open(output_file)
img.show()   # opens in your default image viewer

