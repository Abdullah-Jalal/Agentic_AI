from langgraph.graph import StateGraph,END,START
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message  import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return{"messages":[response]}
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL","llama-3.1-8b-instant"),
    temperature=float(os.getenv("GROQ_TEMPERATURE","0.7")),
    api_key = os.getenv("GROQ_API_KEY")
)
conn = sqlite3.connect(database = "chatbot.db" , check_same_thread=False)
checkpointer =  SqliteSaver(conn=conn)
graph = StateGraph(ChatState)
graph.add_node("chat_node" , chat_node),
graph.add_edge(START, "chat_node"),
graph.add_edge("chat_node",END)
graph.compile()

png_data = graph.get_graph().draw_mermaid_png()
print(png_data)

# try:
#     print(llm.invoke("hello"))
# except Exception as e:
#     print(f"error occured {e}")
# response = llm.invoke([HumanMessage(content = "hello, how are you?")])
# print(response.content)
# user = llm.invoke([response])
# print(user.content)

