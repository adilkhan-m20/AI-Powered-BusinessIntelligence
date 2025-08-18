
# app/task_queue.py - Fixed Advanced Task Queue with Real-time Updates
import asyncio
import json
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int = 0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processing_time: Optional[float] = None

class TaskQueue:
    """Advanced async task queue with real-time updates and monitoring"""
    
    def __init__(self):
        self.workers: Dict[str, asyncio.Task] = {}
        self.task_processors = {}
        self._shutdown = False
        self.task_storage: Dict[str, TaskResult] = {}
        self.task_queue_storage: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize task queue and start workers"""
        # Register task processors
        self._register_processors()
        
        # Start worker processes
        await self._start_workers()
        
        logger.info("✅ Task Queue initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources"""
        self._shutdown = True
        
        # Cancel all workers
        for worker_task in self.workers.values():
            worker_task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers.values(), return_exceptions=True)
        
        logger.info("🧹 Task Queue cleanup completed")
    
    def _register_processors(self):
        """Register all task processors"""
        self.task_processors = {
            "document_processing": self._process_document,
            "rag_indexing": self._process_rag_indexing,
            "quality_validation": self._process_quality_validation,
            "batch_processing": self._process_batch_documents,
            "model_training": self._process_model_training
        }
    
    async def _start_workers(self, num_workers: int = 3):
        """Start worker processes"""
        for i in range(num_workers):
            worker_name = f"worker_{i}"
            self.workers[worker_name] = asyncio.create_task(
                self._worker_loop(worker_name)
            )
        
        logger.info(f"👷 Started {num_workers} worker processes")
    
    async def _worker_loop(self, worker_name: str):
        """Main worker loop"""
        logger.info(f"🧵 Worker {worker_name} started")
        
        while not self._shutdown:
            try:
                task_info = None
                
                # In-memory queue
                if self.task_queue_storage:
                    task_info = self.task_queue_storage.pop(0)
                else:
                    await asyncio.sleep(1)  # Wait for tasks
                
                if task_info:
                    # Process the task
                    await self._process_task(task_info, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"🛑 Worker {worker_name} stopped")
    
    async def _process_task(self, task_info: Dict[str, Any], worker_name: str):
        """Process a single task"""
        task_id = task_info["task_id"]
        task_type = task_info["task_type"]
        
        try:
            # Update task status to processing
            await self._update_task_status(task_id, TaskStatus.PROCESSING, 0, f"Processing with {worker_name}")
            
            # Get the processor
            processor = self.task_processors.get(task_type)
            if not processor:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # Process the task
            start_time = datetime.utcnow()
            result = await processor(task_info)
            end_time = datetime.utcnow()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Update task status to completed
            await self._update_task_status(
                task_id, 
                TaskStatus.COMPLETED, 
                100, 
                "Task completed successfully",
                result,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Update task status to failed
            await self._update_task_status(
                task_id, 
                TaskStatus.FAILED, 
                0, 
                f"Task failed: {str(e)}"
            )
    
    async def _update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus, 
        progress: int, 
        message: str,
        result: Optional[Dict[str, Any]] = None,
        processing_time: Optional[float] = None
    ):
        """Update task status and notify clients"""
        
        # Get or create task result
        task_result = self.task_storage.get(task_id)
        if not task_result:
            task_result = TaskResult(
                task_id=task_id,
                status=status,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        
        # Update fields
        task_result.status = status
        task_result.progress = progress
        task_result.message = message
        task_result.updated_at = datetime.utcnow()
        
        if result:
            task_result.result = result
        if processing_time:
            task_result.processing_time = processing_time
        
        # Store updated task
        self.task_storage[task_id] = task_result
        
        # Send real-time update
        user_id = task_info.get("user_id") if 'task_info' in locals() else None
        if user_id:
            from .websocket_manager import websocket_manager
            await websocket_manager.send_to_user(
                int(user_id),
                {
                    "type": "task_update",
                    "task_id": task_id,
                    "status": status.value,
                    "progress": progress,
                    "message": message
                }
            )
    
    # Task Processors
    async def _process_document(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process document - OCR, chunking, embedding"""
        document_id = task_info["document_id"]
        file_path = task_info["file_path"]
        
        # Update progress
        await self._update_task_status(task_info["task_id"], TaskStatus.PROCESSING, 20, "Loading document...")
        
        try:
            # Import AI integration
            from .ai_integration import process_document_with_ai
            
            # Process document with progress updates
            await self._update_task_status(task_info["task_id"], TaskStatus.PROCESSING, 40, "Extracting text...")
            
            # Use AI integration to process document
            result = await process_document_with_ai(file_path, document_id)
            
            await self._update_task_status(task_info["task_id"], TaskStatus.PROCESSING, 80, "Creating embeddings...")
            
            if result["success"]:
                return {
                    "document_id": document_id,
                    "chunks_created": result["chunks_created"],
                    "embeddings_created": result["embeddings_created"],
                    "processing_stats": result["processing_stats"]
                }
            else:
                raise Exception(result.get("error", "Unknown processing error"))
                
        except Exception as e:
            raise Exception(f"Document processing failed: {str(e)}")
    
    async def _process_rag_indexing(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process RAG indexing for user's documents"""
        user_id = task_info["user_id"]
        document_ids = task_info["document_ids"]
        
        await self._update_task_status(task_info["task_id"], TaskStatus.PROCESSING, 30, "Building vector index...")
        
        # Simulate indexing process
        index_stats = await self._build_user_index(user_id, document_ids)
        
        await self._update_task_status(task_info["task_id"], TaskStatus.PROCESSING, 90, "Finalizing index...")
        
        return {
            "user_id": user_id,
            "documents_indexed": len(document_ids),
            "total_chunks": index_stats.get("chunks", 0),
            "index_size": index_stats.get("size", 0)
        }
    
    async def _process_quality_validation(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process document quality validation"""
        document_id = task_info["document_id"]
        file_path = task_info.get("file_path", "")
        
        try:
            from .ai_integration import validate_document_with_ai
            
            validation_result = await validate_document_with_ai(file_path)
            
            return {
                "document_id": document_id,
                "quality_score": validation_result.get("quality_score", 0.0),
                "validation_passed": validation_result.get("is_valid", False),
                "warnings": validation_result.get("warnings", []),
                "errors": validation_result.get("errors", [])
            }
        except Exception as e:
            return {
                "document_id": document_id,
                "quality_score": 0.0,
                "validation_passed": False,
                "error": str(e)
            }
    
    async def _process_batch_documents(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process multiple documents in batch"""
        document_ids = task_info["document_ids"]
        
        results = []
        total_docs = len(document_ids)
        
        for i, doc_id in enumerate(document_ids):
            progress = int((i / total_docs) * 100)
            await self._update_task_status(
                task_info["task_id"], 
                TaskStatus.PROCESSING, 
                progress, 
                f"Processing document {i+1}/{total_docs}"
            )
            
            # Process individual document
            doc_result = await self._process_single_document(doc_id)
            results.append(doc_result)
        
        return {
            "total_processed": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "results": results
        }
    
    async def _process_model_training(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process model training task"""
        # Placeholder for model training
        await asyncio.sleep(5)  # Simulate training time
        
        return {
            "model_id": task_info.get("model_id", "default"),
            "training_completed": True,
            "accuracy": 0.92
        }
    
    # Public API methods
    async def enqueue_document_processing(self, document_id: int, file_path: str, user_id: int) -> str:
        """Enqueue document processing task"""
        task_id = str(uuid.uuid4())
        
        task_info = {
            "task_id": task_id,
            "task_type": "document_processing",
            "document_id": document_id,
            "file_path": file_path,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Add to queue
        self.task_queue_storage.append(task_info)
        
        # Initialize task status
        await self._update_task_status(task_id, TaskStatus.PENDING, 0, "Task queued")
        
        return task_id
    
    async def enqueue_batch_processing(self, document_ids: List[int], user_id: int) -> str:
        """Enqueue batch processing task"""
        task_id = str(uuid.uuid4())
        
        task_info = {
            "task_id": task_id,
            "task_type": "batch_processing",
            "document_ids": document_ids,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.task_queue_storage.append(task_info)
            
        await self._update_task_status(task_id, TaskStatus.PENDING, 0, "Batch processing queued")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task status"""
        return self.task_storage.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        # Mark task as cancelled
        await self._update_task_status(task_id, TaskStatus.CANCELLED, 0, "Task cancelled by user")
        return True
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "pending_tasks": len(self.task_queue_storage),
            "active_workers": len(self.workers),
            "task_types": list(self.task_processors.keys()),
            "storage_backend": "memory"
        }
    
    async def health_check(self) -> bool:
        """Check if task queue is healthy"""
        # Memory backend is always healthy if workers are running
        return len(self.workers) > 0
    
    # Helper methods
    async def _build_user_index(self, user_id: int, document_ids: List[int]) -> Dict[str, Any]:
        """Build vector index for user's documents"""
        # Simulate index building
        await asyncio.sleep(2)
        
        return {
            "chunks": len(document_ids) * 10,  # Assume 10 chunks per document
            "size": len(document_ids) * 1024,  # Assume 1KB per document
            "user_id": user_id
        }
    
    async def _process_single_document(self, document_id: int) -> Dict[str, Any]:
        """Process a single document"""
        try:
            # Simulate document processing
            await asyncio.sleep(1)
            
            return {
                "success": True, 
                "document_id": document_id,
                "chunks_created": 5,
                "processing_time": 1.0
            }
        except Exception as e:
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e)
            }

# Global task queue instance
task_queue = TaskQueue()