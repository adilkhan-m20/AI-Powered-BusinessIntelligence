
# app/main.py - Fixed Main FastAPI Application
import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional
import json
import logging
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .database import get_db, create_tables, check_database_health
from .auth import get_current_user, create_access_token, authenticate_user, get_current_admin_user
from .models import *
from .services import *
from .websocket_manager import websocket_manager, WebSocketManager
from .task_queue import task_queue
from .monitoring import metrics_collector

# Global managers
websocket_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    try:
        logger.info("🚀 Starting MultiModal AI Backend...")
        await create_tables()
        await task_queue.initialize()
        await metrics_collector.initialize()
        logger.info("✅ MultiModal AI Backend Started Successfully!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown  
    try:
        await task_queue.cleanup()
        await metrics_collector.cleanup()
        logger.info("👋 Backend Shutdown Complete")
    except Exception as e:
        logger.error(f"⚠️ Shutdown error: {e}")

app = FastAPI(
    title="EchoWeave - MultiModal AI",
    description="Enterprise-grade backend for intelligent document processing with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
try:
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
        logger.info("✅ Static files mounted")
    else:
        logger.warning("⚠️ Static directory not found")
except Exception as e:
    logger.error(f"⚠️ Could not mount static files: {e}")

# Root route
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Root endpoint with basic info"""
    return """
    <html>
        <head>
            <title>EchoWeave - MultiModal AI</title>
        </head>
        <body>
            <h1>🧠 EchoWeave - MultiModal AI</h1>
            <p>Enterprise-grade document processing with RAG</p>
            <ul>
                <li><a href="/docs">📖 API Documentation</a></li>
                <li><a href="/redoc">🔄 Interactive Docs</a></li>
                <li><a href="/health">❤️ Health Check</a></li>
                <li><a href="/static/index.html">🌐 Dashboard</a></li>
            </ul>
        </body>
    </html>
    """

# Health Check
@app.get("/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": metrics_collector.get_timestamp(),
        "services": {
            "database": await check_database_health(),
            "redis": await task_queue.health_check(),
            "ai_models": await check_ai_models_health()
        }
    }

# Authentication Routes
@app.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserCredentials, db: AsyncSession = Depends(get_db)):
    """User authentication endpoint"""
    user = await authenticate_user(db, credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token({"sub": user.username, "user_id": user.id})
    return TokenResponse(access_token=access_token, token_type="bearer")

@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """User registration endpoint"""
    try:
        # Check if user already exists
        existing_user = await UserService.get_user_by_username(db, user_data.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        
        # Create new user
        from .auth import AuthService
        hashed_password = AuthService.get_password_hash(user_data.password)
        user = await UserService.create_user(
            db, user_data.username, user_data.email, hashed_password
        )
        
        return UserResponse.from_orm(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document Management Routes
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process document asynchronously"""
    
    # Validate file
    validation_result = await validate_document(file)
    if not validation_result["is_valid"]:
        raise HTTPException(status_code=400, detail=validation_result["errors"])
    
    # Create document record
    document = await create_document_record(db, file, user.id)
    
    # Queue processing task
    task_id = await task_queue.enqueue_document_processing(
        document_id=document.id,
        file_path=document.file_path,
        user_id=user.id
    )
    
    # Update document with task ID
    document.task_id = task_id
    await db.commit()
    
    # Send real-time update
    await websocket_manager.send_to_user(
        user.id, 
        {"type": "document_uploaded", "document_id": document.id, "task_id": task_id}
    )
    
    return DocumentResponse.from_orm(document)

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List user's documents with filtering and pagination"""
    documents = await get_user_documents(db, user.id, skip, limit)
    return [DocumentResponse.from_orm(doc) for doc in documents]

@app.get("/documents/{document_id}/status", response_model=ProcessingStatus)
async def get_document_status(
    document_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get real-time document processing status"""
    document = await get_document_by_id(db, document_id, user.id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not document.task_id:
        # Document was processed before task queue system
        return ProcessingStatus(
            document_id=document_id,
            status=DocumentStatus(document.status),
            progress=100 if document.status == "completed" else 0,
            message=f"Document {document.status}",
            updated_at=document.updated_at
        )
    
    task_result = await task_queue.get_task_status(document.task_id)
    if not task_result:
        return ProcessingStatus(
            document_id=document_id,
            status=DocumentStatus.PENDING,
            progress=0,
            message="Task not found",
            updated_at=datetime.utcnow()
        )
    
    try:
        doc_status = DocumentStatus(task_result.status.value)
    except (ValueError, AttributeError):
        doc_status = DocumentStatus.PENDING
    
    return ProcessingStatus(
        document_id=document_id,
        status=doc_status,
        progress=getattr(task_result, "progress", 0) or 0,
        message=getattr(task_result, "message", "") or "",
        updated_at=getattr(task_result, "updated_at", datetime.utcnow())
    )

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete user's document"""
    document = await get_document_by_id(db, document_id, user.id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Cancel processing task if running
    if document.task_id:
        await task_queue.cancel_task(document.task_id)
    
    # Delete file from filesystem
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception as e:
        logger.warning(f"Warning: Could not delete file {document.file_path}: {e}")
    
    # Delete from database
    await db.delete(document)
    await db.commit()
    
    return {"message": "Document deleted successfully"}

# RAG Query Routes
@app.post("/rag/query", response_model=RAGResponse)
async def query_documents(
    query_request: RAGQueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Query documents using RAG with multi-agent system"""
    
    # Get user's processed documents
    user_documents = await get_processed_documents(db, user.id)
    if not user_documents:
        raise HTTPException(status_code=400, detail="No processed documents found")
    
    # Initialize multi-agent RAG system
    rag_system = await get_user_rag_system(user.id)
    
    # Process query with timing
    start_time = metrics_collector.get_timestamp()
    
    try:
        result = await rag_system.process_query(query_request.query, query_request.filters)
        processing_time = metrics_collector.get_timestamp() - start_time
        
        # Log metrics
        await metrics_collector.log_query(user.id, query_request.query, processing_time)
        
        return RAGResponse(
            query=query_request.query,
            response=result.response,
            sources=result.sources,
            confidence=result.confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = metrics_collector.get_timestamp() - start_time
        await metrics_collector.log_query(user.id, query_request.query, processing_time)
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# WebSocket for Real-time Updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """WebSocket connection for real-time updates"""
    connection_success = await websocket_manager.connect(user_id, websocket)
    
    if not connection_success:
        await websocket.close(code=1000, reason="Connection failed")
        return
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection health
            if data == "ping":
                await websocket.send_text("pong")
            elif data.startswith("{"):
                # Handle JSON messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "heartbeat":
                        await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))
                except json.JSONDecodeError:
                    pass
                
    except WebSocketDisconnect:
        await websocket_manager.disconnect(user_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        await websocket_manager.disconnect(user_id, websocket)

# Analytics and Monitoring Routes
@app.get("/analytics/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user dashboard metrics"""
    try:
        return await metrics_collector.get_user_dashboard(user.id)
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        # Fallback to basic stats
        stats = await AnalyticsService.get_user_stats(db, user.id)
        return DashboardMetrics(
            user_id=user.id,
            total_documents=stats["total_documents"],
            total_queries=stats["total_queries"],
            recent_activity_count=0,
            last_activity=None,
            avg_query_time=0.0,
            storage_used_mb=stats["storage_used_mb"]
        )

@app.get("/analytics/usage", response_model=UsageMetrics)
async def get_usage_metrics(
    days: int = 30,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user usage analytics"""
    try:
        return await metrics_collector.get_usage_metrics(user.id, days)
    except Exception as e:
        logger.error(f"Failed to get usage metrics: {e}")
        return UsageMetrics(
            period_days=days,
            total_activity=0,
            document_uploads=0,
            queries=0,
            daily_activity={},
            most_active_day=None
        )

# Admin Routes
@app.get("/admin/system-metrics", response_model=SystemMetrics)
async def get_system_metrics(
    user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get system-wide metrics (admin only)"""
    try:
        return await metrics_collector.get_system_metrics()
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve system metrics: {str(e)}")

@app.get("/admin/queue-stats")
async def get_queue_stats(
    user: User = Depends(get_current_admin_user)
):
    """Get task queue statistics (admin only)"""
    return await task_queue.get_queue_stats()

# Additional utility endpoints
@app.get("/version")
async def get_version():
    """Get API version information"""
    return {
        "name": "EchoWeave - MultiModal AI",
        "version": "1.0.0",
        "python_version": "3.8+",
        "fastapi_version": "0.104.1"
    }

async def check_ai_models_health() -> bool:
    """Check AI models health"""
    try:
        # Test basic AI service connectivity
        from .ai_integration import validate_document_with_ai
        test_result = await validate_document_with_ai("dummy_path")
        return test_result.get("is_valid") is not None
    except Exception as e:
        logger.error(f"AI health check failed: {e}")
        return False