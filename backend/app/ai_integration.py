
# backend/app/ai_integration.py - Fixed Bridge to AI Service
import os
import sys
import asyncio
import logging
import importlib.util
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
AI_SERVICE_PATH = Path(__file__).parent.parent.parent / "ai_service"
AI_SERVICE_PATH = AI_SERVICE_PATH.resolve()

# Add the project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verify paths exist
if not AI_SERVICE_PATH.exists():
    logger.error(f"AI service path does not exist: {AI_SERVICE_PATH}")
    raise FileNotFoundError(f"AI service path not found: {AI_SERVICE_PATH}")

# Verify critical files exist
required_files = ["document_engine.py", "RAG.py"]
missing_files = [f for f in required_files if not (AI_SERVICE_PATH / f).exists()]
if missing_files:
    logger.error(f"Missing required files in AI service directory: {missing_files}")
    raise FileNotFoundError(f"Missing AI service files: {missing_files}")

class AIServiceIntegration:
    """Integration bridge to your existing AI service"""
    
    def __init__(self):
        self.ai_service_url = os.getenv("AI_SERVICE_URL", "http://localhost:3000")
        self.ai_service_path = AI_SERVICE_PATH
        logger.info(f"AI service path: {self.ai_service_path}")
        
    async def process_document_with_ai(self, document_path: str, document_id: int) -> Dict[str, Any]:
        """Process document using your existing AI service"""
        try:
            # Run your existing document processing logic
            result = await self._run_document_processing(document_path, document_id)
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_created": result.get("chunks", 0),
                "embeddings_created": result.get("embeddings", 0),
                "processing_stats": result.get("stats", {}),
                "faiss_updated": result.get("faiss_updated", True)
            }
            
        except Exception as e:
            logger.exception(f"AI processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }
    
    async def _run_document_processing(self, document_path: str, document_id: int) -> Dict[str, Any]:
        """Run your existing document processing in async context"""
        
        try:
            # Run your document processing logic in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._process_document_sync, document_path, document_id)
            
            return result
            
        except Exception as e:
            logger.exception(f"Document processing failed: {e}")
            raise Exception(f"AI processing failed: {str(e)}")
    
    def _process_document_sync(self, document_path: str, document_id: int) -> Dict[str, Any]:
        """Synchronous wrapper for your existing document processing"""
        
        try:
            # Add AI service path to sys.path temporarily
            sys.path.insert(0, str(self.ai_service_path))
            
            try:
                # First try direct import (works if ai_service is a proper package)
                try:
                    from ai_service.document_engine import process_document_pipeline
                    logger.info("✅ Successfully imported document_engine via package import")
                except ImportError:
                    # Try module import (works if ai_service is not a package)
                    from document_engine import process_document_pipeline  # type: ignore
                    logger.info("✅ Successfully imported document_engine via module import")
                
                # Process the document
                result = process_document_pipeline(document_id, document_path)
                
                return {
                    "chunks": result.get("chunks", 0),
                    "embeddings": result.get("embeddings", 0),
                    "stats": result.get("stats", {}),
                    "faiss_updated": True
                }
                
            finally:
                # Remove the path from sys.path
                if str(self.ai_service_path) in sys.path:
                    sys.path.remove(str(self.ai_service_path))
                
        except ImportError as e:
            logger.exception(f"Failed to import AI modules: {e}")
            raise Exception(f"Failed to import AI modules: {str(e)}")
        except Exception as e:
            logger.exception(f"Document processing failed: {e}")
            raise Exception(f"Document processing failed: {str(e)}")
    
    async def query_rag_system(self, query: str, user_id: int, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Query your existing RAG system"""
        try:
            # Run RAG query in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._run_rag_query, query, user_id, filters)
            
            return {
                "success": True,
                "response": result.get("response", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "tool_calls": result.get("tool_calls", 0)
            }
            
        except Exception as e:
            logger.exception(f"RAG query failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_rag_query(self, query: str, user_id: int, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous wrapper for RAG query using your existing code"""
        
        try:
            # Add AI service path to sys.path temporarily
            sys.path.insert(0, str(self.ai_service_path))
            
            try:
                # Import your existing RAG modules with proper paths
                from langchain_core.messages import HumanMessage
                
                # Import RAG with multiple fallback strategies
                try:
                    # First try: package-style import
                    from ai_service.RAG import retriever_tool, llm
                    logger.info("✅ Successfully imported RAG via package import")
                except ImportError:
                    try:
                        # Second try: direct module import
                        from RAG import retriever_tool, llm  # type: ignore
                        logger.info("✅ Successfully imported RAG via module import")
                    except ImportError:
                        # Third try: explicit path import
                        logger.warning("⚠️ Falling back to explicit path import for RAG")
                        rag_path = self.ai_service_path / "RAG.py"
                        
                        # Check if file exists
                        if not rag_path.exists():
                            raise FileNotFoundError(f"RAG.py not found at {rag_path}")
                        
                        # Load module safely
                        spec = importlib.util.spec_from_file_location("RAG", str(rag_path))
                        if spec is None or spec.loader is None:
                            raise ImportError(f"Could not create module spec for RAG.py at {rag_path}")
                        
                        rag_module = importlib.util.module_from_spec(spec)
                        sys.modules["RAG"] = rag_module
                        spec.loader.exec_module(rag_module)  # This is safe now with the None check
                        
                        retriever_tool = rag_module.retriever_tool
                        llm = rag_module.llm
                        logger.info("✅ Successfully imported RAG via explicit path import")
                
                # Create the message for your RAG agent
                messages = [HumanMessage(content=query)]
                
                # Process query with context
                # First, get relevant documents
                retrieved = retriever_tool.invoke(query)
                
                # Then, generate response with context
                prompt = f"""
                Use the following context to answer the question:
                
                {retrieved}
                
                Question: {query}
                Answer:
                """
                
                # Invoke the LLM
                response = llm.invoke(prompt)
                
                # Try to extract sources using your retriever tool directly
                sources_info = []
                try:
                    if "Document 1" in retrieved:
                        # Parse the retriever result to extract sources
                        source_docs = retrieved.split("Document ")
                        for i, doc in enumerate(source_docs[1:], 1):  # Skip first empty element
                            if doc.strip():
                                # Clean up the document text
                                clean_text = doc.strip()[:200]
                                sources_info.append({
                                    "document_id": f"doc_{i}",
                                    "document_name": f"Document {i}",
                                    "chunk_text": clean_text + ("..." if len(doc.strip()) > 200 else ""),
                                    "page_number": None,
                                    "relevance_score": max(0.1, 0.9 - (i * 0.1))  # Decreasing relevance
                                })
                except Exception as source_error:
                    logger.warning(f"Error extracting sources: {source_error}")
                
                return {
                    "response": response,
                    "sources": sources_info,
                    "confidence": min(0.95, 0.7 + (len(sources_info) * 0.05)),  # Dynamic confidence
                    "tool_calls": 1
                }
                
            finally:
                # Remove the path from sys.path
                if str(self.ai_service_path) in sys.path:
                    sys.path.remove(str(self.ai_service_path))
            
        except Exception as e:
            logger.exception(f"RAG query failed: {e}")
            raise Exception(f"RAG query failed: {str(e)}")
    
    async def validate_document_quality(self, document_path: str) -> Dict[str, Any]:
        """Validate document quality using AI"""
        
        try:
            # Basic file validation
            if not os.path.exists(document_path):
                return {
                    "is_valid": False,
                    "quality_score": 0.0,
                    "errors": ["File does not exist"],
                    "warnings": [],
                    "file_size": 0
                }
            
            file_size = os.path.getsize(document_path)
            
            # Check if file is empty
            if file_size == 0:
                return {
                    "is_valid": False,
                    "quality_score": 0.0,
                    "errors": ["File is empty"],
                    "warnings": [],
                    "file_size": 0
                }
            
            # Check file extension
            _, ext = os.path.splitext(document_path.lower())
            supported_extensions = ['.txt', '.pdf', '.xlsx', '.xls']
            
            warnings = []
            if ext not in supported_extensions:
                warnings.append(f"File extension {ext} may not be fully supported")
            
            # Try to load the document to check if it's readable
            quality_score = 0.5  # Base score
            
            try:
                loader = self._create_document_loader(document_path)
                if loader:
                    docs = loader.load()
                    if docs and len(docs) > 0:
                        quality_score = 0.8
                        # Check if there's actual content
                        total_content = sum(len(doc.page_content.strip()) for doc in docs)
                        if total_content > 100:  # Reasonable amount of content
                            quality_score = 0.9
                    else:
                        warnings.append("Document loaded but no content extracted")
                else:
                    warnings.append("Could not create document loader")
                    quality_score = 0.3
                    
            except Exception as load_error:
                warnings.append(f"Error loading document: {str(load_error)}")
                quality_score = 0.4
            
            return {
                "is_valid": quality_score > 0.5,
                "quality_score": quality_score,
                "errors": [],
                "warnings": warnings,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.exception(f"Document validation failed: {e}")
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "errors": [str(e)],
                "warnings": [],
                "file_size": 0
            }
    
    def _create_document_loader(self, document_path: str):
        """Create appropriate document loader based on file type"""
        try:
            # Import with correct paths
            from langchain_community.document_loaders.text import TextLoader
            from langchain_community.document_loaders.pdf import PyPDFLoader
            from langchain_community.document_loaders.excel import UnstructuredExcelLoader
            
            # Get file extension
            _, ext = os.path.splitext(document_path.lower())
            
            # Create appropriate loader
            if ext == ".txt":
                return TextLoader(document_path, encoding='utf-8')
            elif ext == ".pdf":
                return PyPDFLoader(document_path)
            elif ext in [".xlsx", ".xls"]:
                return UnstructuredExcelLoader(document_path)
            else:
                # Try TextLoader as fallback
                return TextLoader(document_path, encoding='utf-8')
                
        except Exception as e:
            logger.exception(f"Error creating document loader: {e}")
            return None
    
    async def get_rag_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        try:
            # Add AI service path to sys.path temporarily
            sys.path.insert(0, str(self.ai_service_path))
            
            try:
                # Import with correct paths
                from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
                from langchain_community.vectorstores.faiss import FAISS
                
                # Load FAISS database to get stats
                if os.path.exists("faiss_index"):
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    
                    return {
                        "faiss_index_exists": True,
                        "estimated_chunks": "Available",
                        "index_path": "faiss_index",
                        "embedding_model": "all-MiniLM-L6-v2"
                    }
                else:
                    return {
                        "faiss_index_exists": False,
                        "estimated_chunks": 0,
                        "index_path": "faiss_index",
                        "embedding_model": None
                    }
                    
            finally:
                # Remove the path from sys.path
                if str(self.ai_service_path) in sys.path:
                    sys.path.remove(str(self.ai_service_path))
                
        except Exception as e:
            logger.exception(f"RAG stats retrieval failed: {e}")
            return {
                "faiss_index_exists": False,
                "error": str(e),
                "estimated_chunks": 0
            }

# Global AI service integration instance
ai_service = AIServiceIntegration()

# Helper functions for easy integration
async def process_document_with_ai(document_path: str, document_id: int) -> Dict[str, Any]:
    """Helper function to process document"""
    return await ai_service.process_document_with_ai(document_path, document_id)

async def query_rag_with_ai(query: str, user_id: int, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """Helper function to query RAG"""
    return await ai_service.query_rag_system(query, user_id, filters)

async def validate_document_with_ai(document_path: str) -> Dict[str, Any]:
    """Helper function to validate document"""
    return await ai_service.validate_document_quality(document_path)

async def get_rag_stats() -> Dict[str, Any]:
    """Helper function to get RAG statistics"""
    return await ai_service.get_rag_agent_stats()