import os
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables (OpenAI, Tavily, etc.)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END

# -------------------------------------------------------------
# 1. Document Loading & Indexing
# -------------------------------------------------------------
print("Loading PDF documents...")
docs = (
    PyPDFLoader("./documents/Company_Policies.pdf").load()
    + PyPDFLoader("./documents/Company_Profile.pdf").load()
    + PyPDFLoader("./documents/Product_and_Pricing.pdf").load()
)

chunks = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=150
).split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------------------------------------------
# 2. State Definition
# -------------------------------------------------------------
class State(TypedDict):
    question: str
    need_retrieval: bool

    docs: List[Document]
    relevant_docs: List[Document]

    context: str
    answer: str

    # Web query and loop safety
    web_query: str
    web_search_attempted: bool

# -------------------------------------------------------------
# 3. Decision Node: Whether Retrieval is Needed
# -------------------------------------------------------------
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer reliably, else False."
    )

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON that matches this schema:\n"
            "{{'should_retrieve': boolean}}\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts, citations, or info likely not in the model.\n"
            "- should_retrieve=False for general explanations, definitions, or reasoning that doesn't need sources.\n"
            "- If unsure, choose True."
        ),
        ("human", "Question: {question}"),
    ]
)

should_retrieve_llm = llm.with_structured_output(RetrieveDecision)

def decide_retrieval(state: State):
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    return {"need_retrieval": decision.should_retrieve}

# -------------------------------------------------------------
# 4. Direct Generation Node (No Retrieval Needed)
# -------------------------------------------------------------
direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only your general knowledge.\n"
            "Do NOT assume access to external documents.\n"
            "If you are unsure or the answer requires specific sources, say:\n"
            "'I don't know based on my general knowledge.'"
        ),
        ("human", "{question}"),
    ]
)

def generate_direct(state: State):
    out = llm.invoke(
        direct_generation_prompt.format_messages(
            question=state["question"]
        )
    )
    return {"answer": out.content}

# -------------------------------------------------------------
# 5. Local Document Retrieval Node
# -------------------------------------------------------------
def retrieve(state: State):
    return {"docs": retriever.invoke(state["question"])}

# -------------------------------------------------------------
# 6. Document Relevance Grading Node
# -------------------------------------------------------------
class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if the document helps answer the question, else False."
    )

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance.\n"
            "Return JSON that matches this schema:\n"
            "{{'is_relevant': boolean}}\n\n"
            "A document is relevant if it contains information useful for answering the question."
        ),
        (
            "human",
            "Question:\n{question}\n\nDocument:\n{document}"
        ),
    ]
)

relevance_llm = llm.with_structured_output(RelevanceDecision)

def is_relevant(state: State):
    relevant_docs: List[Document] = []

    for doc in state.get("docs", []):
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content
            )
        )

        if decision.is_relevant:
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}

# -------------------------------------------------------------
# 7. Context Generation Node (Business RAG)
# -------------------------------------------------------------
rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business RAG assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "If the context does not contain enough information, say:\n"
            "'No relevant document found.'\n"
            "Do not use outside knowledge.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)

def generate_from_context(state: State):
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()

    if not context:
        return {"answer": "No relevant document found.", "context": ""}

    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context
        )
    )
    return {"answer": out.content, "context": context}

# -------------------------------------------------------------
# 8. Query Rewriter & Web Search Fallback Node
# -------------------------------------------------------------
class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency, add (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

def rewrite_query_node(state: State):
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}

def execute_web_search(query: str, max_results: int = 5):
    # 1. Use Tavily if TAVILY_API_KEY is available
    if os.getenv("TAVILY_API_KEY"):
        try:
            tavily = TavilySearchResults(max_results=max_results)
            return tavily.invoke({"query": query})
        except Exception as e:
            print(f"Tavily search notice: {e}")

    # 2. DuckDuckGo fallback
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "content": item.get("body", "")
                }
                for item in raw
            ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def web_search_node(state: State):
    q = state.get("web_query") or state["question"]
    results = execute_web_search(q, max_results=5)

    docs = []
    for r in results or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        docs.append(
            Document(
                page_content=text,
                metadata={"source": "web", "url": url, "title": title},
            )
        )

    return {"docs": docs, "web_search_attempted": True}

# -------------------------------------------------------------
# 9. Conditional Edge Functions
# -------------------------------------------------------------
def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    return "generate_direct"

def route_after_relevance(state: State) -> Literal["generate_from_context", "rewrite_query"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    # Loop protection: if web search was already executed once, stop cycling and generate answer
    if state.get("web_search_attempted"):
        return "generate_from_context"
    return "rewrite_query"

# -------------------------------------------------------------
# 10. Graph Construction
# -------------------------------------------------------------
g = StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)

g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)

g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)

# Edges
g.add_edge(START, "decide_retrieval")

g.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {
        "generate_direct": "generate_direct",
        "retrieve": "retrieve",
    },
)

g.add_edge("generate_direct", END)

# vector retrieval -> relevance
g.add_edge("retrieve", "is_relevant")

# relevance router: if relevant -> generate, else -> rewrite_query
g.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "rewrite_query": "rewrite_query",
    },
)

# web fallback path
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "is_relevant")  # circle back to grade web search docs

# final
g.add_edge("generate_from_context", END)

app = g.compile()

if __name__ == "__main__":
    print("Self-RAG workflow ready!")
    
    # Test 1: Direct Generation
    print("\n" + "="*50)
    print("Test 1: Direct Generation (What is Machine Learning)")
    print("="*50)
    res1 = app.invoke({
        "question": "what is Machine Learning",
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "web_query": "",
        "web_search_attempted": False,
    })
    print(res1["answer"])

    # Test 2: Web Search Fallback
    print("\n" + "="*50)
    print("Test 2: Web Search Fallback (Aus vs Zim World T20 2026)")
    print("="*50)
    res2 = app.invoke({
        "question": "Who won the Aus vs Zim World T20 match 2026 and who was the top scorer",
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "web_query": "",
        "web_search_attempted": False,
    })
    print("Answer:\n", res2["answer"])
    print(f"\nFound {len(res2.get('relevant_docs', []))} relevant documents:")
    for doc in res2.get("relevant_docs", []):
        print("-" * 40)
        print(doc.page_content[:300] + "...")
