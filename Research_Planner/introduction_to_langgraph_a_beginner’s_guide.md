# Introduction to LangGraph: A Beginner’s Guide

## What Is LangGraph and Why It Matters  

LangGraph is a lightweight, graph‑oriented framework designed to orchestrate large language model (LLM) components into coherent, production‑ready applications. Instead of writing linear scripts that chain prompts, API calls, and post‑processing steps, developers define **nodes** (individual LLM calls, data fetches, or custom logic) and **edges** (the flow of information between nodes). The resulting graph visualizes the entire reasoning or workflow, making it easy to:

* **Compose** reusable LLM primitives (e.g., retrieval, summarisation, tool use) without duplicating boilerplate.  
* **Branch** and **loop** based on model output, enabling dynamic decision‑making such as tool selection, re‑asking, or error recovery.  
* **Monitor** execution paths, debug failures, and collect metrics per node, which is far harder in monolithic pipelines.  

### Role in Building LLM‑Driven Applications  

1. **Modular Architecture** – Each node is an isolated, testable unit. Teams can swap a retrieval node for a newer vector store or replace a summarisation model without touching the rest of the graph.  
2. **State Management** – LangGraph maintains a shared state object that travels with the graph, so context (e.g., conversation history, retrieved documents) is automatically available to downstream nodes.  
3. **Tool Integration** – Nodes can invoke external tools (APIs, databases, code execution environments) and feed the results back into the LLM, supporting “agentic” behaviours.  
4. **Scalability** – Because the graph is declarative, it can be executed locally for prototyping or dispatched to distributed runtimes (e.g., serverless workers, orchestration platforms) for production.  

### Problems Solved Compared to Traditional Pipelines  

| Traditional Linear Pipelines | LangGraph Approach |
|------------------------------|--------------------|
| **Hard‑coded sequence** – Adding a new step often requires rewriting the whole script. | **Declarative graph** – New nodes or branches are added by updating the graph definition, not the control flow code. |
| **Limited branching** – Conditional logic is tangled with prompt code, leading to spaghetti code. | **Explicit edges** – Branches and loops are first‑class, making conditional paths clear and visualizable. |
| **State leakage** – Global variables or manual passing of context cause bugs and make debugging difficult. | **Central state object** – State is passed automatically, ensuring consistency and simplifying debugging. |
| **Opaque debugging** – Errors surface as generic API failures; tracing which node caused them is painful. | **Node‑level logging & metrics** – Each node can be instrumented individually, giving fine‑grained visibility. |
| **Reusability pain** – Prompt templates and post‑processing code are duplicated across projects. | **Reusable nodes** – A node encapsulating a prompt or tool can be imported across multiple graphs. |

In short, LangGraph transforms the ad‑hoc, script‑driven way of building LLM applications into a structured, maintainable, and observable system—allowing developers to focus on *what* they want the model to do rather than *how* to stitch together a fragile pipeline.

## Key Building Blocks and First‑Step Example

### 1. Core Concepts  

| Concept | What it does | Typical use |
|---------|--------------|-------------|
| **Node** | A reusable unit that contains a prompt (or any callable) and the logic to process its output. | Generate a response, call an API, transform data, etc. |
| **Edge** | The directed link that determines the next node(s) based on the current node’s output or a condition. | Branching logic, loops, or simple linear flow. |
| **State** | A mutable dictionary that travels with the graph execution, holding inputs, intermediate results, and any metadata you need later. | Pass user inputs, store LLM responses, keep track of counters, etc. |
| **Graph** | The orchestrator that wires nodes together via edges and drives the execution loop until a terminal node is reached. | End‑to‑end workflow such as “question → retrieval → answer”. |

### 2. Minimal Two‑Node Graph  

Below is a **complete, runnable** example that creates a tiny LangGraph with:

1. **`PromptNode`** – asks the LLM to re‑phrase a user query.  
2. **`EchoNode`** – simply returns the re‑phrased text (acts as a placeholder for any downstream logic).  

The graph demonstrates:

* **Prompt composition** – the first node builds a prompt string from the incoming state.  
* **Response handling** – each node receives the LLM output and stores it back into the shared state.  
* **Execution** – `graph.run(state)` drives the flow from the start node to the terminal node.

```python
# -------------------------------------------------
# 1️⃣  Install the required packages (once)
# -------------------------------------------------
# pip install langgraph openai   # or `pip install "langgraph[all]"`

# -------------------------------------------------
# 2️⃣  Imports & basic LLM client
# -------------------------------------------------
from langgraph.graph import Graph
from langgraph.nodes import PromptNode, FunctionNode
from openai import OpenAI  # using the official OpenAI SDK

# Replace with your own key or rely on OPENAI_API_KEY env var
client = OpenAI()

# -------------------------------------------------
# 3️⃣  Define the two nodes
# -------------------------------------------------
def rephrase_prompt(state: dict) -> str:
    """Build the prompt that asks the LLM to re‑phrase the user query."""
    user_q = state.get("user_query", "")
    return f"Please rewrite the following question in a clear, concise way:\n\n{user_q}"

prompt_node = PromptNode(
    name="rephrase",
    prompt_builder=rephrase_prompt,
    client=client,
    model="gpt-4o-mini",          # any model supported by your client
    temperature=0.2,
)

def echo(state: dict) -> dict:
    """A trivial downstream node that just passes the result forward."""
    # `state["rephrase_output"]` was stored by the PromptNode automatically.
    state["final_answer"] = f"✅ Re‑phrased query: {state['rephrase_output']}"
    return state

echo_node = FunctionNode(name="echo", fn=echo)

# -------------------------------------------------
# 4️⃣  Wire the graph (edges)
# -------------------------------------------------
graph = Graph()
graph.add_node(prompt_node)   # becomes the start node by default
graph.add_node(echo_node)

# Linear flow: rephrase → echo → end
graph.add_edge("rephrase", "echo")
graph.set_terminal("echo")   # tells LangGraph when to stop

# -------------------------------------------------
# 5️⃣  Run the graph with an initial state
# -------------------------------------------------
initial_state = {"user_query": "How does LangGraph handle state across nodes?"}
result_state = graph.run(initial_state)

# -------------------------------------------------
# 6️⃣  Inspect the outcome
# -------------------------------------------------
print(result_state["final_answer"])
```

#### What the snippet does, step‑by‑step  

1. **Builds a prompt** from the incoming `user_query`.  
2. **Calls the LLM** (`gpt-4o-mini`) and stores the raw response under `rephrase_output`.  
3. **Passes control** to `echo_node` via the defined edge.  
4. `echo_node` **writes a friendly message** into the state (`final_answer`).  
5. The graph stops because `echo_node` is marked as *terminal*.  

You can replace `echo_node` with any custom function—e.g., a retrieval step, a knowledge‑base lookup, or another LLM call—without changing the surrounding wiring. This illustrates the **modular, composable** nature of LangGraph: each node is a black‑box that only talks to the shared state, while edges dictate the execution order.
