
# app/main.py - Main FastAPI Application
import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db, engine, create_tables
from .auth import get_current_user, create_access_token
from .models import *
from .services import *
from .websocket_manager import WebSocketManager
from .task_queue import task_queue
from .monitoring import metrics_collector

# Global managers
websocket_manager = WebSocketManager()
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    await create_tables()
    await task_queue.initialize()
    await metrics_collector.initialize()
    print("🚀 MultiModal AI Backend Started Successfully!")
    
    yield
    
    # Shutdown  
    await task_queue.cleanup()
    await metrics_collector.cleanup()
    print("👋 Backend Shutdown Complete")

app = FastAPI(
    title="MultiModal AI Document Processing API",
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
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    if not validation_result.is_valid:
        raise HTTPException(status_code=400, detail=validation_result.errors)
    
    # Create document record
    document = await create_document_record(db, file, user.id)
    
    # Queue processing task
    task_id = await task_queue.enqueue_document_processing(
        document_id=document.id,
        file_path=document.file_path,
        user_id=user.id
    )
    
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
    
    # Get processing status from task queue
    status = await task_queue.get_task_status(document.task_id)
    return ProcessingStatus(
        document_id=document_id,
        status=status.status,
        progress=status.progress,
        message=status.message,
        updated_at=status.updated_at
    )

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

# WebSocket for Real-time Updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """WebSocket connection for real-time updates"""
    await websocket_manager.connect(user_id, websocket)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection health
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        await websocket_manager.disconnect(user_id)

# Analytics and Monitoring Routes
@app.get("/analytics/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user dashboard metrics"""
    return await metrics_collector.get_user_dashboard(user.id)

@app.get("/analytics/usage", response_model=UsageMetrics)
async def get_usage_metrics(
    days: int = 30,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user usage analytics"""
    return await metrics_collector.get_usage_metrics(user.id, days)

# Admin Routes (if user is admin)
@app.get("/admin/system-metrics", response_model=SystemMetrics)
async def get_system_metrics(
    user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Get system-wide metrics (admin only)"""
    return await metrics_collector.get_system_metrics()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )