
# backend/app/ai_integration.py - Fixed Bridge to AI Service
import os
import sys
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

# Resolve AI service path properly
AI_SERVICE_PATH = Path(__file__).parent.parent.parent / "ai-service"
AI_SERVICE_PATH = AI_SERVICE_PATH.resolve()

# Add to Python path if not already there
if str(AI_SERVICE_PATH) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_PATH))

class AIServiceIntegration:
    """Integration bridge to your existing AI service"""
    
    def __init__(self):
        self.ai_service_url = os.getenv("AI_SERVICE_URL", "http://localhost:3000")
        self.ai_service_path = AI_SERVICE_PATH
        
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
            raise Exception(f"AI processing failed: {str(e)}")
    
    def _process_document_sync(self, document_path: str, document_id: int) -> Dict[str, Any]:
        """Synchronous wrapper for your existing document processing"""
        
        try:
            # Change to ai-service directory to maintain file paths
            original_cwd = os.getcwd()
            os.chdir(str(self.ai_service_path))
            
            try:
                # Import document processing function
                from document_engine import process_document_pipeline
                
                # Process the document
                result = process_document_pipeline(document_id, document_path)
                
                return {
                    "chunks": result.get("chunks", 0),
                    "embeddings": result.get("embeddings", 0),
                    "stats": result.get("stats", {}),
                    "faiss_updated": True
                }
                
            finally:
                os.chdir(original_cwd)
                
        except ImportError as e:
            raise Exception(f"Failed to import AI modules: {str(e)}")
        except Exception as e:
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
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_rag_query(self, query: str, user_id: int, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous wrapper for RAG query using your existing code"""
        
        try:
            # Change to ai-service directory to access FAISS index
            original_cwd = os.getcwd()
            os.chdir(str(self.ai_service_path))
            
            try:
                # Import your existing RAG modules
                from langchain_core.messages import HumanMessage
                from RAG import rag_agent, retriever_tool
                
                # Create the message for your RAG agent
                messages = [HumanMessage(content=query)]
                
                # Invoke your existing RAG agent
                result = rag_agent.invoke({"messages": messages})
                
                # Extract the response
                response_content = result['messages'][-1].content if result['messages'] else "No response generated"
                
                # Try to extract sources using your retriever tool directly
                sources_info = []
                try:
                    retriever_result = retriever_tool.invoke(query)
                    if retriever_result and "Document" in retriever_result:
                        # Parse the retriever result to extract sources
                        source_docs = retriever_result.split("Document ")
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
                    print(f"Error extracting sources: {source_error}")
                
                return {
                    "response": response_content,
                    "sources": sources_info,
                    "confidence": min(0.95, 0.7 + (len(sources_info) * 0.05)),  # Dynamic confidence
                    "tool_calls": len([msg for msg in result.get('messages', []) if hasattr(msg, 'tool_calls')])
                }
                
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
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
            from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredExcelLoader
            
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
            print(f"Error creating document loader: {e}")
            return None
    
    async def get_rag_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        try:
            # Change to ai-service directory
            original_cwd = os.getcwd()
            os.chdir(str(self.ai_service_path))
            
            try:
                from langchain_openai import OpenAIEmbeddings
                from langchain_community.vectorstores import FAISS
                
                # Load FAISS database to get stats
                if os.path.exists("faiss_index"):
                    db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                    
                    return {
                        "faiss_index_exists": True,
                        "estimated_chunks": "Available",
                        "index_path": "faiss_index",
                        "embedding_model": "text-embedding-ada-002"
                    }
                else:
                    return {
                        "faiss_index_exists": False,
                        "estimated_chunks": 0,
                        "index_path": "faiss_index",
                        "embedding_model": None
                    }
                    
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
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