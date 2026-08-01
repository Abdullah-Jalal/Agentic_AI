# Agentic AI: Stateful Workflows with LangGraph & Model Context Protocol (MCP)

Welcome to the **Agentic AI** repository! This project is a comprehensive sandbox and development workspace containing notebooks, custom chatbot backends, Model Context Protocol (MCP) servers, and client applications. It is designed to demonstrate stateful agent orchestration using [LangGraph](https://github.com/langchain-ai/langgraph) and tool-use extension using [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

---

## 📂 Repository Structure

The project is organized as follows:

*   **`chat_bot/`** — Modular chatbot backends using LangGraph:
    *   [backend.py](file:///C:/LangGraph/chat_bot/backend.py) — Basic in-memory stateful chatbot.
    *   [langgraph_backend.py](file:///C:/LangGraph/chat_bot/langgraph_backend.py) — Stateful chatbot with OpenAI model integration.
    *   [langgraph_database_backend.py](file:///C:/LangGraph/chat_bot/langgraph_database_backend.py) — Stateful chatbot utilizing [SqliteSaver](file:///C:/LangGraph/chat_bot/langgraph_database_backend.py#L5) database persistence.
    *   [langraph_rag_backend.py](file:///C:/LangGraph/chat_bot/langraph_rag_backend.py) — Complete retrieval-augmented chatbot featuring PDF ingestion via [PyPDFLoader](file:///C:/LangGraph/chat_bot/langraph_rag_backend.py#L10), [FAISS](file:///C:/LangGraph/chat_bot/langraph_rag_backend.py#L12) vector storage, thread-safe session retriever retrieval, and custom tools.
*   **`mcp/`** — Expense Tracker MCP Server:
    *   [main.py](file:///C:/LangGraph/mcp/main.py) — FastMCP server running async sqlite queries ([aiosqlite](file:///C:/LangGraph/mcp/main.py#L3)) to log expenses.
    *   [categories.json](file:///C:/LangGraph/mcp/categories.json) — Dynamic categories catalog.
    *   [pyproject.toml](file:///C:/LangGraph/mcp/pyproject.toml) — Server dependency definition.
*   **`remote-add-server/`** — Calculator MCP Server:
    *   [main.py](file:///C:/LangGraph/remote-add-server/main.py) — FastMCP server exposing basic math tools run over HTTP transport.
    *   [pyproject.toml](file:///C:/LangGraph/remote-add-server/pyproject.toml) — Calculator server dependency definition.
*   **`yt-mcp-client/`** — Multi-Server MCP Clients & Streamlit UIs:
    *   [client1.py](file:///C:/LangGraph/yt-mcp-client/client1.py) — CLI client leveraging LangChain's [MultiServerMCPClient](file:///C:/LangGraph/yt-mcp-client/client1.py#L2) to coordinate multiple servers.
    *   [client2.py](file:///C:/LangGraph/yt-mcp-client/client2.py) — A complete interactive chat dashboard powered by [Streamlit](file:///C:/LangGraph/yt-mcp-client/client2.py#L7) that binds multi-server tools dynamically.
    *   [pyproject.toml](file:///C:/LangGraph/yt-mcp-client/pyproject.toml) — Client application dependencies.
*   **Notebooks (Root Directory)** — Learning notebooks explaining step-by-step principles of agentic AI:
    *   [9_basic_chatbot.ipynb](file:///C:/LangGraph/9_basic_chatbot.ipynb) — Building stateful chats with LangGraph state.
    *   [Fault_Tolerance.ipynb](file:///C:/LangGraph/Fault_Tolerance.ipynb) — Designing recovery logic & retry strategies.
    *   [_hitl.ipynb](file:///C:/LangGraph/_hitl.ipynb) — Human-in-the-Loop workflows (interrupts & reviews).
    *   [_rag.ipynb](file:///C:/LangGraph/_rag.ipynb) — RAG graph integration with LangGraph.
    *   [base_review_reply.ipynb](file:///C:/LangGraph/base_review_reply.ipynb) — Automated customer review sentiment responder.
    *   [cric_project_workflow.ipynb](file:///C:/LangGraph/cric_project_workflow.ipynb) — Custom sports query/analytics workflow graph.
    *   [essay_workflow.ipynb](file:///C:/LangGraph/essay_workflow.ipynb) — Iterative essay writing assistant (Draft -> Critique -> Revise).
    *   [non_llm_workflow.ipynb](file:///C:/LangGraph/non_llm_workflow.ipynb) — Graph execution using traditional Python functions.
    *   [prompt_chaining.ipynb](file:///C:/LangGraph/prompt_chaining.ipynb) — Sequential LLM pipeline orchestration.
    *   [quadratic.ipynb](file:///C:/LangGraph/quadratic.ipynb) — Equation-solving mathematical graph workflow.
    *   [simple_llm_workflow.ipynb](file:///C:/LangGraph/simple_llm_workflow.ipynb) — Minimal agent connection and state updates.
    *   [X_post_gen.ipynb](file:///C:/LangGraph/X_post_gen.ipynb) — Social media agent pipeline (Idea -> Critique -> Draft).

---

## 🧠 Key Architectural Concepts Covered

### 1. Stateful Graph Orchestration (LangGraph)
*   **State & Nodes**: Defining python functions as nodes that process inputs, update a shared state dictionary, and return modified values.
*   **Edges**: Controlling routing dynamically through traditional edges or `conditional_edges` (e.g. routing based on LLM output or tool calls).
*   **Reducers**: Using state annotators like `add_messages` to handle state accumulation (e.g. appending new messages to message history).

### 2. Advanced Workflow Patterns
*   **Prompt Chaining**: Passing outputs sequentially from one node to another.
*   **Evaluator-Optimizer**: Multi-node feedback loops where one LLM critiques the output of another, driving iterative refinement (seen in [essay_workflow.ipynb](file:///C:/LangGraph/essay_workflow.ipynb) and [X_post_gen.ipynb](file:///C:/LangGraph/X_post_gen.ipynb)).
*   **Parallel Execution**: Running multiple nodes concurrently and merging state updates using reducers.

### 3. Resilience & Human Interaction
*   **Persistence Layer**: Checkpointing conversation states across executions using `InMemorySaver` and `SqliteSaver` database persistence.
*   **Time Travel**: Using the persistence layer to rollback state history, fork runs, or replay conversations from earlier checkpoints.
*   **Human-in-the-Loop (HITL)**: Implementing breakpoints (`interrupt_before` / `interrupt_after`) to pause graph execution for human verification or feedback before continuing.

### 4. Model Context Protocol (MCP)
*   **Multi-Server Integration**: Using `MultiServerMCPClient` to connect an LLM to multiple specialized server contexts simultaneously (e.g. Math, Expense, Manim).
*   **Transport Modes**: Implemented stdio-based transport (local subprocess) and HTTP-based server endpoints.
*   **Resources**: Reading dynamic configurations like expense categories or server metadata through the URI resource model (`expense:///categories`, `server:///info`).

---

## ⚡ Getting Started & Setup

### ⚙️ Prerequisites
1. **Python 3.11+**
2. **uv** (Recommended package manager) or **pip**
3. **SQLite3**

### 🔑 Environment Variables
Create a `.env` file in the root directory (and in `yt-mcp-client/`) with the following variables:
```env
OPENAI_API_KEY=your-openai-api-key-here
GROQ_API_KEY=your-groq-api-key-here
```

---

## 🚀 Running the Sub-Projects

### 1. The RAG Chatbot Backend (`chat_bot/`)
The RAG chatbot supports uploading a PDF, vectorizing it in memory using **FAISS**, and asking context-driven questions using tool-calling.

To use the backend classes:
```python
from chat_bot.langraph_rag_backend import chatbot, ingest_pdf

# 1. Ingest a PDF file for a thread
with open("document.pdf", "rb") as f:
    ingest_pdf(f.read(), thread_id="thread_123", filename="document.pdf")

# 2. Query the chatbot with persistence config
config = {"configurable": {"thread_id": "thread_123"}}
response = chatbot.invoke({"messages": [("user", "Summarize this PDF")]}, config=config)
```

---

### 2. Running the MCP Servers

#### Expense Tracker MCP Server (`mcp/`)
This server manages expenses in an async SQLite database. It runs on the **stdio** transport.
```bash
cd mcp
# Initialize and run using fastmcp CLI
uv run fastmcp run main.py
```
*   **Tools**: `add_expense`, `list_expenses`, `summarize`
*   **Resources**: `expense:///categories`

#### Simple Calculator MCP Server (`remote-add-server/`)
This server runs as a remote HTTP server.
```bash
cd remote-add-server
uv run fastmcp run main.py --transport http
```
*   **Tools**: `add`, `random_number`, `server_info`
*   **Resources**: `server:///info`

---

### 3. Running the MCP Client UI (`yt-mcp-client/`)

A Streamlit application ([client2.py](file:///C:/LangGraph/yt-mcp-client/client2.py)) provides an interactive interface for chatting with an LLM that has access to all configured MCP servers.

1.  Open a terminal and navigate to the client folder:
    ```bash
    cd yt-mcp-client
    ```
2.  Install dependencies:
    ```bash
    uv pip install -r pyproject.toml
    ```
3.  Configure server endpoints in `client2.py` under the `SERVERS` dictionary (pointing to your running MCP servers).
4.  Run the Streamlit application:
    ```bash
    streamlit run client2.py
    ```

---

## 📓 Learning Notebooks Index

| Notebook | Description | Key Concept |
| :--- | :--- | :--- |
| **[9_basic_chatbot.ipynb](file:///C:/LangGraph/9_basic_chatbot.ipynb)** | Building a basic stateful conversational assistant. | `StateGraph`, `add_messages` |
| **[simple_llm_workflow.ipynb](file:///C:/LangGraph/simple_llm_workflow.ipynb)** | The simplest form of calling LLMs inside nodes. | Node invocation |
| **[non_llm_workflow.ipynb](file:///C:/LangGraph/non_llm_workflow.ipynb)** | Building deterministic workflows without LLMs. | Stateful code execution |
| **[prompt_chaining.ipynb](file:///C:/LangGraph/prompt_chaining.ipynb)** | Executing sequential LLM prompts where input depends on prior output. | Pipeline chaining |
| **[essay_workflow.ipynb](file:///C:/LangGraph/essay_workflow.ipynb)** | Iterative agent that drafts, critiques, and revises essays. | Evaluator-Optimizer loop |
| **[X_post_gen.ipynb](file:///C:/LangGraph/X_post_gen.ipynb)** | Social media post planning, drafting, and optimization flow. | Feedback loops |
| **[_hitl.ipynb](file:///C:/LangGraph/_hitl.ipynb)** | Implementing approval breakpoints where execution waits for human input. | Human-in-the-Loop |
| **[_rag.ipynb](file:///C:/LangGraph/_rag.ipynb)** | Integrating retrieval stores (vector DBs) inside LangGraph nodes. | Retrieval-Augmented Gen |
| **[Fault_Tolerance.ipynb](file:///C:/LangGraph/Fault_Tolerance.ipynb)** | Implementing retries, checkpointers, and fallback nodes. | SqliteSaver, Error Handling |
| **[base_review_reply.ipynb](file:///C:/LangGraph/base_review_reply.ipynb)** | Sentiment analysis and automated response formulation workflow. | Routing & classification |
| **[quadratic.ipynb](file:///C:/LangGraph/quadratic.ipynb)** | Mathematical evaluation nodes that resolve math equations. | Algorithmic state updates |
| **[cric_project_workflow.ipynb](file:///C:/LangGraph/cric_project_workflow.ipynb)** | Live scores/stats retrieval or routing logic workflow. | Tool use orchestration |

---

## 🛠️ Built With

*   [LangGraph](https://github.com/langchain-ai/langgraph) — Stateful agent coordination framework.
*   [LangChain](https://github.com/langchain-ai/langchain) — LLM chain toolkit.
*   [FastMCP](https://github.com/jlonge/fastmcp) — Fast Python SDK for Model Context Protocol.
*   [Streamlit](https://streamlit.io/) — Fast web application dashboard framework.
*   [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search library.
*   [aiosqlite](https://github.com/omnilib/aiosqlite) — Asynchronous SQLite driver for Python.
