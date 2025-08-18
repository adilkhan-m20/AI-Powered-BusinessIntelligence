
from dotenv import load_dotenv
import os
import logging
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_community.llms import HuggingFaceHub

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Try to load FAISS database, create empty one if doesn't exist
DB_PATH = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    logger.info("✅ Loaded existing FAISS database")
except Exception as e:
    logger.warning(f"⚠️ Could not load FAISS database: {e}")
    logger.info("Creating empty FAISS database...")
    # Create a dummy document to initialize FAISS
    from langchain.schema import Document
    dummy_doc = Document(page_content="Initial document for FAISS initialization")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents([dummy_doc], embeddings)
    db.save_local(DB_PATH)
    logger.info("✅ Created empty FAISS database")

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
tools_dict = {tool.name: tool for tool in tools}  # Initialize before using

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> bool:
    """Check if the last message contains tool calls."""
    try:
        result = state['messages'][-1]
        return isinstance(result, HumanMessage) and hasattr(result, 'tool_calls') and bool(result.tool_calls)
    except (IndexError, KeyError):
        return False

system_prompt = """
You are an intelligent AI assistant who answers questions related to the query and answers based on the information loaded into your knowledge base.
Use the retriever tool available to answer questions about the query. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

# LLM setup - using Hugging Face instead of OpenAI for local deployment
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    try:
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        message = llm.invoke(messages)
        return {'messages': [message]}
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        error_message = HumanMessage(content=f"Error processing request: {str(e)}")
        return {'messages': [error_message]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    try:
        last_message = state['messages'][-1]

        if not isinstance(last_message, HumanMessage):
            raise TypeError("Expected HumanMessage as the last message.")

        tool_calls = getattr(last_message, "tool_calls", [])

        if not tool_calls:
            raise ValueError("No tool calls found in the last HumanMessage.")
        
        results = []
        for t in tool_calls:
            logger.info(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
            
            if t['name'] not in tools_dict:
                logger.warning(f"\nTool: {t['name']} does not exist.")
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
                logger.info(f"Result length: {len(str(result))}")

            # Appends the Tool Message
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

        logger.info("Tools Execution Complete. Back to the model!")
        return {'messages': results}
    
    except Exception as e:
        logger.error(f"Error executing tools: {e}")
        error_message = ToolMessage(
            tool_call_id="error", 
            name="error", 
            content=f"Error executing tools: {str(e)}"
        )
        return {'messages': [error_message]}

def running_agent():
    logger.info("\n=== RAG AGENT ===")
    
    while True:
        try:
            user_input = input("\nWhat is your question: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            messages = [HumanMessage(content=user_input)]
            # For demonstration, we'll use a simple flow without state graph
            # In production, you'd use the StateGraph approach
            
            # First, get relevant documents
            retrieved = retriever_tool.invoke(user_input)
            
            # Then, generate response with context
            prompt = f"""
            Use the following context to answer the question:
            
            {retrieved}
            
            Question: {user_input}
            Answer:
            """
            
            response = llm.invoke(prompt)
            print("\n=== ANSWER ===")
            print(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            continue

if __name__ == "__main__":
    running_agent()