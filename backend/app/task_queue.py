
# app/task_queue.py - Advanced Task Queue with Real-time Updates
import asyncio
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as aioredis
from pydantic import BaseModel
import logging

from .websocket_manager import websocket_manager
from .monitoring import metrics_collector

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
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.workers: Dict[str, asyncio.Task] = {}
        self.task_processors = {}
        self._shutdown = False
        
    async def initialize(self):
        """Initialize Redis connection and start workers"""
        self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
        
        # Register task processors
        self._register_processors()
        
        # Start worker processes
        await self._start_workers()
        
        logger.info("Task Queue initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources"""
        self._shutdown = True
        
        # Cancel all workers
        for worker_task in self.workers.values():
            worker_task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers.values(), return_exceptions=True)
        
        if self.redis:
            await self.redis.close()
        
        logger.info("Task Queue cleanup completed")
    
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
        
        logger.info(f"Started {num_workers} worker processes")
    
    async def _worker_loop(self, worker_name: str):
        """Main worker loop"""
        logger.info(f"Worker {worker_name} started")
        
        while not self._shutdown:
            try:
                # Block for up to 5 seconds waiting for a task
                task_data = await self.redis.brpop("task_queue", timeout=5)
                
                if task_data:
                    queue_name, task_json = task_data
                    task_info = json.loads(task_json)
                    
                    # Process the task
                    await self._process_task(task_info, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_name} stopped")
    
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
            
            # Log metrics
            await metrics_collector.log_task_completion(task_type, processing_time, success=True)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Update task status to failed
            await self._update_task_status(
                task_id, 
                TaskStatus.FAILED, 
                0, 
                f"Task failed: {str(e)}"
            )
            
            # Log metrics
            await metrics_collector.log_task_completion(task_type, 0, success=False)
    
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
        
        # Get current task data
        current_data = await self.redis.get(f"task:{task_id}")
        if current_data:
            task_result = TaskResult.parse_raw(current_data)
        else:
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
        await self.redis.set(
            f"task:{task_id}", 
            task_result.json(),
            ex=86400  # 24 hours expiry
        )
        
        # Send real-time update
        user_id = await self.redis.get(f"task_user:{task_id}")
        if user_id:
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
        
        # Import your existing document processing logic
        from document_engine import process_document_pipeline
        
        # Process document with progress updates
        await self._update_task_status(task_info["task_id"], TaskStatus.PROCESSING, 40, "Extracting text...")
        
        # Your document processing logic here
        result = await process_document_pipeline(document_id, file_path)
        
        await self._update_task_status(task_info["task_id"], TaskStatus.PROCESSING, 80, "Creating embeddings...")
        
        return {
            "document_id": document_id,
            "chunks_created": result["chunks"],
            "embeddings_created": result["embeddings"],
            "processing_stats": result["stats"]
        }
    
    async def _process_rag_indexing(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process RAG indexing for user's documents"""
        user_id = task_info["user_id"]
        document_ids = task_info["document_ids"]
        
        # Your RAG indexing logic here
        await self._update_task_status(task_info["task_id"], TaskStatus.PROCESSING, 30, "Building vector index...")
        
        # Process indexing
        index_stats = await self._build_user_index(user_id, document_ids)
        
        await self._update_task_status(task_info["task_id"], TaskStatus.PROCESSING, 90, "Finalizing index...")
        
        return {
            "user_id": user_id,
            "documents_indexed": len(document_ids),
            "total_chunks": index_stats["chunks"],
            "index_size": index_stats["size"]
        }
    
    async def _process_quality_validation(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process document quality validation"""
        document_id = task_info["document_id"]
        
        # Your quality validation logic
        quality_score = await self._validate_document_quality(document_id)
        
        return {
            "document_id": document_id,
            "quality_score": quality_score,
            "validation_passed": quality_score > 0.7
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
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results
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
        
        # Store task info
        await self.redis.set(f"task_user:{task_id}", user_id, ex=86400)
        
        # Add to queue
        await self.redis.lpush("task_queue", json.dumps(task_info))
        
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
        
        await self.redis.set(f"task_user:{task_id}", user_id, ex=86400)
        await self.redis.lpush("task_queue", json.dumps(task_info))
        await self._update_task_status(task_id, TaskStatus.PENDING, 0, "Batch processing queued")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task status"""
        task_data = await self.redis.get(f"task:{task_id}")
        if task_data:
            return TaskResult.parse_raw(task_data)
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        # Mark task as cancelled
        await self._update_task_status(task_id, TaskStatus.CANCELLED, 0, "Task cancelled by user")
        return True
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        pending_tasks = await self.redis.llen("task_queue")
        
        return {
            "pending_tasks": pending_tasks,
            "active_workers": len(self.workers),
            "task_types": list(self.task_processors.keys())
        }
    
    async def health_check(self) -> bool:
        """Check if task queue is healthy"""
        try:
            await self.redis.ping()
            return True
        except:
            return False
    
    # Helper methods (implement based on your existing logic)
    async def _build_user_index(self, user_id: int, document_ids: List[int]) -> Dict[str, Any]:
        """Build vector index for user's documents"""
        # Integrate with your existing FAISS logic
        pass
    
    async def _validate_document_quality(self, document_id: int) -> float:
        """Validate document quality"""
        # Implement quality validation logic
        return 0.85  # Placeholder
    
    async def _process_single_document(self, document_id: int) -> Dict[str, Any]:
        """Process a single document"""
        # Implement single document processing
        return {"success": True, "document_id": document_id}

# Global task queue instance
task_queue = TaskQueue()