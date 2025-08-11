from dotenv import load_dotenv
import os
from document_engine_fixed import fetch_text_from_api
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage

load_dotenv()

# Try to load FAISS database, create empty one if doesn't exist
try:
    db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    print("✅ Loaded existing FAISS database")
except Exception as e:
    print(f"⚠️ Could not load FAISS database: {e}")
    print("Creating empty FAISS database...")
    # Create a dummy document to initialize FAISS
    from langchain.schema import Document
    dummy_doc = Document(page_content="Initial document for FAISS initialization")
    db = FAISS.from_documents([dummy_doc], OpenAIEmbeddings())
    db.save_local("faiss_index")
    print("✅ Created empty FAISS database")

url = "http://localhost:3000/data"  # express api endpoint
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Minimize hallucination

# Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Vector DB.
    """
    try:
        docs = retriever.invoke(query)
        
        if not docs:
            return "I found no relevant information in the database."
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}")
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching database: {str(e)}"

tools = [retriever_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> bool:
    """Check if the last message contains tool calls."""
    try:
        result = state['messages'][-1]
        return isinstance(result, AIMessage) and hasattr(result, 'tool_calls') and bool(result.tool_calls)
    except (IndexError, KeyError):
        return False

system_prompt = """
You are an intelligent AI assistant who answers questions related to the query and answers based on the information loaded into your knowledge base.
Use the retriever tool available to answer questions about the query. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {tool.name: tool for tool in tools}  # Fixed variable name

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    try:
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        message = llm.invoke(messages)
        return {'messages': [message]}
    except Exception as e:
        error_message = AIMessage(content=f"Error processing request: {str(e)}")
        return {'messages': [error_message]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    try:
        last_message = state['messages'][-1]

        if not isinstance(last_message, AIMessage):
            raise TypeError("Expected AIMessage as the last message.")

        tool_calls = getattr(last_message, "tool_calls", [])

        if not tool_calls:
            raise ValueError("No tool calls found in the last AIMessage.")
        
        results = []
        for t in tool_calls:
            print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
            
            if t['name'] not in tools_dict:
                print(f"\nTool: {t['name']} does not exist.")
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
                print(f"Result length: {len(str(result))}")

            # Appends the Tool Message
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

        print("Tools Execution Complete. Back to the model!")
        return {'messages': results}
    
    except Exception as e:
        error_message = ToolMessage(
            tool_call_id="error", 
            name="error", 
            content=f"Error executing tools: {str(e)}"
        )
        return {'messages': [error_message]}

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT ===")
    
    while True:
        try:
            user_input = input("\nWhat is your question: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            messages = [HumanMessage(content=user_input)]
            result = rag_agent.invoke({"messages": messages})
            
            print("\n=== ANSWER ===")
            print(result['messages'][-1].content)
            
        except Exception as e:
            print(f"Error processing query: {e}")
            continue

if __name__ == "__main__":
    running_agent()