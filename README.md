# 🤖 Agentic AI: Production-Grade Stateful Agentic Workflows & Multi-Agent Systems

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-Ecosystem-1C3C3C?style=for-the-badge)](https://github.com/langchain-ai/langchain)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Standard-8A2BE2?style=for-the-badge)](https://modelcontextprotocol.io/)
[![FastMCP](https://img.shields.io/badge/FastMCP-Async_Servers-009688?style=for-the-badge)](https://github.com/jlonge/fastmcp)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-blue?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A comprehensive, production-grade sandbox and master repository for **Autonomous AI Agents**, **Stateful Workflows with LangGraph**, **Model Context Protocol (MCP)** client/server ecosystems, **Self-Corrective RAG (Self-RAG)**, **Stateful Memory Systems (STM & LTM)**, and **Multi-Agent Orchestration (Map-Reduce & Subgraphs)**.

---

## 📑 Table of Contents

- [Overview & Capabilities](#-overview--capabilities)
- [System Architectures & Workflow Diagrams](#-system-architectures--workflow-diagrams)
  - [1. Self-Corrective RAG (Self-RAG)](#1-self-corrective-rag-self-rag-architecture)
  - [2. Multi-Agent Map-Reduce Blog Writer (BWA)](#2-autonomous-blog-writing-agent-bwa-map-reduce)
  - [3. Model Context Protocol (MCP) Ecosystem](#3-model-context-protocol-mcp-multi-server-orchestration)
- [Key Systems & Sub-Projects](#-key-systems--sub-projects)
  - [Self-RAG Engine](#-self-corrective-rag-engine)
  - [Autonomous Blog Writing Agent (BWA)](#-autonomous-blog-writing-agent-bwa)
  - [Memory Architecture (STM, Trimming, Summarization & LTM)](#-stateful-memory-architecture)
  - [Model Context Protocol (MCP) Servers & Streamlit UI](#-model-context-protocol-mcp-suite)
  - [Modular Chatbot Backends](#-modular-chatbot-backends)
- [Curated Learning & Implementation Roadmap](#-curated-learning--implementation-roadmap)
- [Repository Structure](#-repository-structure)
- [Technical Guides & Publications](#-technical-guides--publications)
- [Getting Started & Setup](#-getting-started--setup)
  - [Prerequisites](#-prerequisites)
  - [Environment Variables](#-environment-variables)
  - [Quickstart Commands](#-quickstart-commands)
- [Tech Stack](#-tech-stack)

---

## 🌟 Overview & Capabilities

This repository serves as an end-to-end laboratory demonstrating how to build autonomous, reliable, and production-ready agentic AI systems. Moving beyond basic linear chains, it explores non-linear, cyclic, and stateful multi-agent graphs:

* 🔄 **Self-Corrective RAG (Self-RAG)**: Self-grading document retrieval with dynamic routing, vector search, LLM-as-a-judge relevance checks, autonomous query rewriting, and Tavily/DuckDuckGo web search fallback.
* 👥 **Multi-Agent Orchestration**: Hierarchical Orchestrator-Worker-Reducer patterns leveraging LangGraph's dynamic `Send` API to parallelize tasks and reduce results via subgraphs.
* 🧠 **Dual-Tier Memory**:
  * **Short-Term Memory (STM)**: Thread-scoped message state with token-bounded trimming (`trim_messages`) and recursive conversation summarization (`ChatState`).
  * **Long-Term Memory (LTM)**: Persistent cross-session user memories and profiles backed by PostgreSQL (`langgraph-checkpoint-postgres`).
* 🔌 **Model Context Protocol (MCP)**: Custom FastMCP servers exposing tools and URI resources over `stdio` and `HTTP` transports, coordinated by LangChain `MultiServerMCPClient` and an interactive Streamlit UI.
* 🛡️ **Resilience & Human-in-the-Loop (HITL)**: State checkpointing (`SqliteSaver`, `InMemorySaver`), rollback time travel, breakpoint interrupts (`interrupt_before`/`interrupt_after`), and fault-tolerant retry nodes.
* 🎯 **Fine-Tuning Engineering**: Complete LoRA and QLoRA fine-tuning workflows and architectural guides for adapting open-source foundation models.

---

## 📐 System Architectures & Workflow Diagrams

### 1. Self-Corrective RAG (Self-RAG) Architecture

The Self-RAG pipeline (`yt-mcp-client/self_rag.py`) dynamically evaluates whether retrieval is necessary, scores document relevance, rewrites search queries upon poor retrieval, and queries the live web before generating grounded responses.

```mermaid
flowchart TD
    START([User Query]) --> Decide[decide_retrieval<br/>Structured LLM Decision]
    
    Decide -->|need_retrieval = False| Direct[generate_direct<br/>LLM General Knowledge]
    Decide -->|need_retrieval = True| Retrieve[retrieve<br/>FAISS Vector Store Search]
    
    Retrieve --> Grade[is_relevant<br/>LLM-as-a-Judge Relevance Grading]
    
    Grade -->|Relevant Docs Found| ContextGen[generate_from_context<br/>Grounded Business RAG Answer]
    Grade -->|No Relevant Docs & 1st Attempt| Rewrite[rewrite_query<br/>Query Keyword Transformation]
    Grade -->|No Relevant Docs & Search Done| ContextGen
    
    Rewrite --> WebSearch[web_search<br/>Tavily / DuckDuckGo Fallback]
    WebSearch --> Grade
    
    Direct --> END([Final Answer])
    ContextGen --> END
```

### 2. Autonomous Blog Writing Agent (BWA) Map-Reduce

The Blog Writing Agent (`bwa.ipynb`) implements an Orchestrator-Worker-Reducer pattern to plan, research, write, and enrich long-form technical publications.

```mermaid
flowchart TD
    START([Topic Input]) --> Router{router<br/>Closed vs Open Book?}
    
    Router -->|Needs Research| Research[research<br/>Tavily Web Evidence Collection]
    Router -->|Closed Book| Orch[orchestrator<br/>Pydantic Task Breakdown]
    Research --> Orch
    
    Orch -->|Send API Map Dispatch| W1[Worker 1: Intro Section]
    Orch -->|Send API Map Dispatch| W2[Worker 2: Architecture Section]
    Orch -->|Send API Map Dispatch| W3[Worker 3: Code Implementation]
    Orch -->|Send API Map Dispatch| WN[Worker N: Optimization & Summary]
    
    W1 & W2 & W3 & WN --> ReducerSubgraph[[Reducer Subgraph]]
    
    subgraph ReducerSubgraph [Reducer & Asset Assembly Subgraph]
        direction TB
        Merge[merge_content<br/>Sort & Assemble Sections] --> DecideImg[decide_images<br/>LLM Visual Asset Planner]
        DecideImg --> GenImg[generate_or_fetch_images<br/>Visual Generation / Placeholders]
        GenImg --> FinalMarkdown[assemble_final_markdown<br/>Export Production Article]
    end
    
    ReducerSubgraph --> END([Ready-to-Publish Markdown])
```

### 3. Model Context Protocol (MCP) Multi-Server Orchestration

Connects an LLM agent to heterogeneous tool servers through standard MCP transports.

```mermaid
flowchart LR
    subgraph Clients [Client Layer]
        StreamlitUI[Streamlit Chat Dashboard<br/>yt-mcp-client/client2.py]
        CLIClient[MultiServer MCP CLI<br/>yt-mcp-client/client1.py]
    end

    subgraph Orchestrator [LangChain MCP Layer]
        MultiClient[MultiServerMCPClient]
    end

    subgraph Servers [MCP Server Layer]
        ExpenseServer[Expense MCP Server<br/>FastMCP + Async SQLite<br/>stdio transport]
        MathServer[Math Calculator MCP Server<br/>FastMCP HTTP Transport<br/>port 8000]
    end

    StreamlitUI & CLIClient --> MultiClient
    MultiClient -->|stdio: python main.py| ExpenseServer
    MultiClient -->|SSE / HTTP: localhost:8000/sse| MathServer
```

---

## 🚀 Key Systems & Sub-Projects

### 🔍 Self-Corrective RAG Engine
* **Files**: [`yt-mcp-client/self_rag.py`](yt-mcp-client/self_rag.py), [`yt-mcp-client/self_rag.ipynb`](yt-mcp-client/self_rag.ipynb)
* **Highlights**:
  * Ingests local enterprise documents (`Company_Policies.pdf`, `Company_Profile.pdf`, `Product_and_Pricing.pdf`).
  * Structured decision node classifies whether query requires domain knowledge.
  * Grades each retrieved chunk independently using Pydantic structured outputs (`RelevanceDecision`).
  * Autonomous search query rewriting with live search fallback (Tavily Search API with DuckDuckGo backup).
  * Built-in loop cycle prevention flag (`web_search_attempted`) ensuring termination.

### ✍️ Autonomous Blog Writing Agent (BWA)
* **Files**: [`bwa.ipynb`](bwa.ipynb), [`Research_Planner/`](Research_Planner/)
* **Highlights**:
  * Pydantic schema validation for section task plans (`Task(id, title, goal)`).
  * Dynamic dispatch via LangGraph `Send(node, task)` to spin up isolated, parallel workers.
  * Subgraph Reducer architecture assembling markdown, determining image candidates, and compiling comprehensive technical guides.

### 🧠 Stateful Memory Architecture
* **Short-Term Memory (STM)**:
  * [`stm.ipynb`](stm.ipynb): Stateful dialogue using `MessagesState` and ChatGroq (`llama-3.1-8b-instant`).
  * [`3_stm_trimming.ipynb`](3_stm_trimming.ipynb): Context trimming using `trim_messages` bounded by max token budgets (`MAX_TOKENS = 150`).
  * [`5_stm_summarization.ipynb`](5_stm_summarization.ipynb): Memory compression using running summaries (`ChatState(summary=...)`) when message histories exceed threshold limits.
* **Long-Term Memory (LTM)**:
  * [`ltm_memory.ipynb`](ltm_memory.ipynb): Cross-session persistent user memories, personalized preference stores, and semantic memory recall using PostgreSQL checkpoints (`PostgresSaver`).

### 🔌 Model Context Protocol (MCP) Suite
* **Expense Tracker Server** ([`mcp/main.py`](mcp/main.py)):
  * Built on `FastMCP` with asynchronous SQLite database access (`aiosqlite`).
  * Tools: `add_expense`, `list_expenses`, `summarize`.
  * Resources: Dynamic resource provider (`expense:///categories`).
* **Remote Calculator Server** ([`remote-add-server/main.py`](remote-add-server/main.py)):
  * FastMCP server hosted via HTTP transport with SSE capabilities.
  * Tools: `add`, `random_number`, `server_info`.
* **Multi-Server Clients** ([`yt-mcp-client/`](yt-mcp-client/)):
  * Streamlit web application ([`client2.py`](yt-mcp-client/client2.py)) connecting to multiple servers simultaneously.
  * Interactive chat with tool-invocation tracing and streaming responses.

### 💬 Modular Chatbot Backends
* **Directory**: [`chat_bot/`](chat_bot/)
  * [`backend.py`](chat_bot/backend.py): In-memory stateful LangGraph conversational engine.
  * [`langgraph_backend.py`](chat_bot/langgraph_backend.py): OpenAI integrated graph agent.
  * [`langgraph_database_backend.py`](chat_bot/langgraph_database_backend.py): Production SQLite persistence with `SqliteSaver`.
  * [`langraph_rag_backend.py`](chat_bot/langraph_rag_backend.py): Document-grounded RAG chatbot with multi-tenant thread-isolated retrieval (`thread_id`) and FAISS.

---

## 🗺️ Curated Learning & Implementation Roadmap

| Stage | Focus Area | Notebook / File | Core LangGraph Primitives & Techniques |
| :---: | :--- | :--- | :--- |
| **01** | **Foundations & Graphs** | [`simple_llm_workflow.ipynb`](simple_llm_workflow.ipynb)<br>[`non_llm_workflow.ipynb`](non_llm_workflow.ipynb)<br>[`quadratic.ipynb`](quadratic.ipynb) | `StateGraph`, `START`, `END`, deterministic state updates, graph compilation |
| **02** | **Chaining & Workflows** | [`prompt_chaining.ipynb`](prompt_chaining.ipynb)<br>[`cric_project_workflow.ipynb`](cric_project_workflow.ipynb) | Sequential chaining, conditional edges, payload routing |
| **03** | **Evaluator-Optimizer Loops** | [`essay_workflow.ipynb`](essay_workflow.ipynb)<br>[`X_post_gen.ipynb`](X_post_gen.ipynb)<br>[`base_review_reply.ipynb`](base_review_reply.ipynb) | Cyclic graphs, feedback loops (Draft ➔ Critique ➔ Revise), sentiment routing |
| **04** | **Short-Term Memory (STM)** | [`stm.ipynb`](stm.ipynb)<br>[`3_stm_trimming.ipynb`](3_stm_trimming.ipynb)<br>[`5_stm_summarization.ipynb`](5_stm_summarization.ipynb) | `MessagesState`, `trim_messages`, token budgeting, running summaries (`ChatState`) |
| **05** | **Long-Term Memory (LTM)** | [`ltm_memory.ipynb`](ltm_memory.ipynb) | `PostgresSaver`, cross-thread memory stores, semantic profiles |
| **06** | **RAG & Self-Corrective RAG** | [`_rag.ipynb`](_rag.ipynb)<br>[`yt-mcp-client/self_rag.ipynb`](yt-mcp-client/self_rag.ipynb)<br>[`yt-mcp-client/self_rag.py`](yt-mcp-client/self_rag.py) | Vector stores, LLM relevance grading, query rewriting, web search fallback |
| **07** | **Human-in-the-Loop & Fault Tolerance** | [`_hitl.ipynb`](_hitl.ipynb)<br>[`Fault_Tolerance.ipynb`](Fault_Tolerance.ipynb) | `interrupt_before`, `interrupt_after`, `SqliteSaver`, state time-travel & rollback |
| **08** | **Multi-Agent Systems** | [`translator_shared.ipynb`](translator_shared.ipynb)<br>[`bwa.ipynb`](bwa.ipynb)<br>[`Research_Planner/`](Research_Planner/) | `Send` API, Map-Reduce, Parallel Workers, Subgraphs, Research Orchestrator |
| **09** | **Model Context Protocol (MCP)** | [`mcp/main.py`](mcp/main.py)<br>[`remote-add-server/main.py`](remote-add-server/main.py)<br>[`yt-mcp-client/client2.py`](yt-mcp-client/client2.py) | FastMCP, `stdio` & `HTTP` transports, `MultiServerMCPClient`, Streamlit UI |
| **10** | **Model Fine-Tuning** | [`fine_tune.ipynb`](fine_tune.ipynb)<br>[Fine-Tuning Guide](fine-tuning_llms_with_lora_and_qlora_a_practical_engineering_guide.md) | Parameter-efficient fine-tuning (PEFT), LoRA, QLoRA, 4-bit quantization |

---

## 📂 Repository Structure

```text
├── chat_bot/                               # Modular LangGraph Chatbot Backends
│   ├── backend.py                          # In-memory chatbot
│   ├── langgraph_backend.py                # OpenAI integrated chatbot
│   ├── langgraph_database_backend.py       # Chatbot backed by SqliteSaver persistence
│   └── langraph_rag_backend.py             # Full PDF RAG chatbot with thread-safe retriever
├── documents/                              # Knowledge base sample documents (PDFs)
│   ├── Company_Policies.pdf                # Company policy documentation
│   ├── Company_Profile.pdf                 # Company overview & leadership
│   └── Product_and_Pricing.pdf             # Enterprise software pricing tiers
├── mcp/                                    # FastMCP Expense Tracker Server
│   ├── main.py                             # FastMCP async server (aiosqlite)
│   ├── categories.json                     # Dynamic categories resource
│   ├── expenses.db                         # SQLite expense database
│   ├── proxy.py                            # Network proxy wrapper
│   ├── test_client.py                      # Server test suite
│   └── pyproject.toml                      # Server dependency specification
├── remote-add-server/                      # Remote Calculator MCP Server
│   ├── main.py                             # HTTP / SSE transport FastMCP server
│   └── pyproject.toml                      # Calculator dependency specification
├── Research_Planner/                       # Autonomous Research & Blog Planning
│   ├── bwa_basic.ipynb                     # Basic blog writing agent
│   ├── bwa-research.ipynb                  # Research-augmented blog writer
│   ├── improved_bwa.ipynb                  # Advanced multi-agent blog writer
│   ├── building_autonomous_ai_agents_...md # Autonomous Agents Architecture Guide
│   └── introduction_to_langgraph_...md     # LangGraph Beginner's Guide
├── yt-mcp-client/                          # MCP Multi-Server Client Suite & Self-RAG
│   ├── client1.py                          # MultiServerMCPClient CLI interface
│   ├── client2.py                          # Streamlit Multi-Server Chatbot UI
│   ├── self_rag.py                         # Standalone Self-Corrective RAG Pipeline
│   ├── self_rag.ipynb                      # Interactive Self-RAG Development Notebook
│   ├── documents/                          # Local copy of knowledge base PDFs
│   └── pyproject.toml                      # Client dependencies
├── 3_stm_trimming.ipynb                    # Short-term memory trimming with token limits
├── 5_stm_summarization.ipynb               # Short-term memory progressive summarization
├── 9_basic_chatbot.ipynb                   # Introductory LangGraph conversational state
├── base_review_reply.ipynb                 # Sentiment classification and response graph
├── building_autonomous_ai_agents_...md     # Comprehensive guide on state, memory & agents
├── bwa.ipynb                               # Production Blog Writing Agent (Orchestrator-Worker-Reducer)
├── cric_project_workflow.ipynb             # Sports query routing and analytics graph
├── essay_workflow.ipynb                    # Multi-stage essay writing & critique loop
├── Fault_Tolerance.ipynb                   # Persistence, checkpoint retries, and error recovery
├── fine-tuning_llms_with_lora_and_...md    # Engineering handbook on LoRA & QLoRA
├── fine_tune.ipynb                         # Practical fine-tuning pipeline notebook
├── ltm_memory.ipynb                        # Long-term memory with PostgresSaver
├── non_llm_workflow.ipynb                  # Deterministic code-only LangGraph pipelines
├── prompt_chaining.ipynb                   # Sequential LLM pipeline orchestration
├── quadratic.ipynb                         # Algorithmic state graph for quadratic equations
├── simple_llm_workflow.ipynb               # Minimal node state updates
├── stm.ipynb                               # Short-term memory with MessagesState
├── translator_shared.ipynb                 # Multi-agent shared state translation system
├── X_post_gen.ipynb                        # Social post generator with Evaluator-Optimizer
├── _hitl.ipynb                             # Human-in-the-Loop breakpoints and state edits
└── _rag.ipynb                              # Core RAG node integration in LangGraph
```

---

## 📚 Technical Guides & Publications

This repository includes detailed technical publications created during research and development:

1. **[Building Autonomous AI Agents with LangGraph: State, Memory, and Multi-Agent Workflows](building_autonomous_ai_agents_with_langgraph_state_memory_and_multi-agent_workflows.md)**  
   *Deep dive into state management, short-term vs. long-term memory architectures, subgraphs, and multi-agent coordination.*

2. **[Fine-Tuning LLMs with LoRA and QLoRA: A Practical Engineering Guide](fine-tuning_llms_with_lora_and_qlora_a_practical_engineering_guide.md)**  
   *Practical handbook covering low-rank adaptation, 4-bit NormalFloat (NF4) quantization, memory optimization, and Hugging Face PEFT integration.*

3. **[Introduction to LangGraph: A Beginner's Guide](Research_Planner/introduction_to_langgraph_a_beginner's_guide.md)**  
   *Clear foundational walkthrough of graphs, nodes, edges, reducers, and checkpointers for developers new to LangGraph.*

---

## ⚡ Getting Started & Setup

### ⚙️ Prerequisites

* **Python 3.11+** installed
* **uv** (recommended) or **pip**
* **SQLite3**
* Optional: **PostgreSQL** (for cross-session Long-Term Memory checkpointer)

### 🔑 Environment Variables

Create a `.env` file in the repository root (and within `yt-mcp-client/`):

```env
# Core LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Search API Key (for Self-RAG & BWA Web Research)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: PostgreSQL Connection (for ltm_memory.ipynb)
POSTGRES_DB_URI=postgresql://postgres:password@localhost:5432/agent_memory

# Optional: Tracing & Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

### 💻 Quickstart Commands

#### 1. Install Dependencies
Using `uv` (recommended):
```bash
uv venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

uv pip install langgraph langchain langchain-openai langchain-groq langchain-community faiss-cpu fastmcp pypdf duckduckgo-search tavily-python streamlit aiosqlite
```

#### 2. Run Self-Corrective RAG
Execute the end-to-end Self-RAG system with local vector indexing and web fallback:
```bash
python yt-mcp-client/self_rag.py
```

#### 3. Run MCP Servers & Streamlit UI
* **Terminal 1: Start Remote Math MCP Server**
  ```bash
  cd remote-add-server
  uv run fastmcp run main.py --transport http --port 8000
  ```
* **Terminal 2: Launch Streamlit Multi-Server Dashboard**
  ```bash
  cd yt-mcp-client
  streamlit run client2.py
  ```

#### 4. Run the PDF RAG Chatbot
```python
from chat_bot.langraph_rag_backend import chatbot, ingest_pdf

# Ingest document
with open("documents/Company_Policies.pdf", "rb") as f:
    ingest_pdf(f.read(), thread_id="session_01", filename="Company_Policies.pdf")

# Query with session persistence
config = {"configurable": {"thread_id": "session_01"}}
result = chatbot.invoke({"messages": [("user", "What is the policy on remote work?")]}, config=config)
print(result["messages"][-1].content)
```

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain Core](https://github.com/langchain-ai/langchain) |
| **LLMs & Inference** | [OpenAI](https://openai.com/) (`gpt-4o`, `gpt-4o-mini`), [Groq](https://groq.com/) (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) |
| **Tool Protocol** | [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), [FastMCP](https://github.com/jlonge/fastmcp) |
| **Vector Search** | [FAISS](https://github.com/facebookresearch/faiss), `OpenAIEmbeddings` (`text-embedding-3-large`) |
| **Search Engines** | [Tavily Search API](https://tavily.com/), [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) |
| **Persistence & DBs** | [SQLite3](https://sqlite.org/), [aiosqlite](https://github.com/omnilib/aiosqlite), [PostgreSQL](https://www.postgresql.org/) (`psycopg`, `langgraph-checkpoint-postgres`) |
| **UI & Dashboards** | [Streamlit](https://streamlit.io/) |
| **Package Management**| [uv](https://github.com/astral-sh/uv), [pip](https://pypi.org/project/pip/) |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
