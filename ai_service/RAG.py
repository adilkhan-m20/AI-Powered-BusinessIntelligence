
# ai_service/RAG.py - Fixed RAG System for Windows
from dotenv import load_dotenv
import os
import logging
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.tools import tool
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix for Windows symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
        messages = state['messages']
        if not messages:
            return False
            
        # Check if the last message has tool calls
        last_message = messages[-1]
        
        # For HumanMessage, check if it has tool_calls attribute
        if hasattr(last_message, 'tool_calls'):
            return bool(getattr(last_message, 'tool_calls', None))
            
        # Alternative check for tool calls in the message content
        if hasattr(last_message, 'content'):
            content = getattr(last_message, 'content', '')
            return "tool_calls" in str(content).lower()
            
        return False
    except (IndexError, KeyError):
        return False

system_prompt = """
You are an intelligent AI assistant who answers questions related to the query and answers based on the information loaded into your knowledge base.
Use the retriever tool available to answer questions about the query. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

# Local LLM setup using a smaller, local model that doesn't require API tokens
try:
    # Use the correct model class for T5 (sequence-to-sequence)
    model_name = "google/flan-t5-small"
    
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    
    # Create tokenizer and model with the correct class
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Create pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.5,
        device=device
    )
    
    # Create LLM from pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    
    logger.info(f"✅ Successfully loaded local model: {model_name}")
    
except Exception as e:
    logger.warning(f"⚠️ Could not load local model: {e}")
    logger.info("Using a very simple fallback model...")
    
    # Create a very simple fallback that just returns the context
    def simple_llm(prompt):
        # Extract context and question from prompt
        if "Use the following context to answer the question:" in prompt:
            parts = prompt.split("Use the following context to answer the question:")
            context = parts[1].split("Question:")[0].strip()
            question = parts[1].split("Question:")[1].split("Answer:")[0].strip()
            
            # Just return a simple answer with context
            return f"Based on the information provided: {context[:200]}... I believe the answer to '{question}' is related to this context."
        return "I'm sorry, I couldn't process your request properly."
    
    # Create a simple wrapper to match the LLM interface
    class SimpleLLM:
        def invoke(self, prompt):
            return simple_llm(prompt)
    
    llm = SimpleLLM()
    logger.info("✅ Using simple fallback model")

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    try:
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        
        # Convert messages to a simple string prompt for the LLM
        prompt = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                prompt += f"Human: {msg.content}\n"
            elif isinstance(msg, SystemMessage):
                prompt += f"System: {msg.content}\n"
            elif isinstance(msg, ToolMessage):
                prompt += f"Tool Response: {msg.content}\n"
        
        # Add instruction for the LLM
        prompt += "Assistant:"
        
        response = llm.invoke(prompt)
        return {'messages': [HumanMessage(content=response)]}
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        error_message = HumanMessage(content=f"Error processing request: {str(e)}")
        return {'messages': [error_message]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    try:
        messages = state['messages']
        if not messages:
            return state
            
        last_message = messages[-1]
        
        # Since we can't reliably access tool_calls, let's use a simpler approach
        # We'll check if the message content indicates a tool should be called
        if hasattr(last_message, 'content'):
            content = getattr(last_message, 'content', '').lower()
            
            # Look for keywords that might indicate a search is needed
            search_keywords = ['search', 'find', 'look up', 'information about', 'what is', 'who is', 'where is']
            if any(keyword in content for keyword in search_keywords):
                # Extract query from the message
                query = content
                
                # Call the retriever tool
                result = retriever_tool.invoke(query)
                
                # Return tool message
                tool_msg = ToolMessage(
                    tool_call_id="search_1", 
                    name="retriever_tool", 
                    content=str(result)
                )
                return {'messages': [tool_msg]}
        
        # If no tool call is needed, return empty result
        return {'messages': []}
    
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
                
            # Simple RAG flow without complex state management
            # 1. Get relevant documents
            retrieved = retriever_tool.invoke(user_input)
            
            # 2. Generate response with context
            prompt = f"""
            Use the following context to answer the question:
            
            {retrieved}
            
            Question: {user_input}
            Answer:
            """
            
            response = llm.invoke(prompt)
            print("\n=== ANSWER ===")
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Sorry, I encountered an error: {e}")
            continue

if __name__ == "__main__":
    running_agent()