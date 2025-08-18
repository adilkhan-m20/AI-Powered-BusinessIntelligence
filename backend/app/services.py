
# backend/app/services.py - Fixed Business Logic Services
import os
import uuid
import aiofiles
from typing import List, Dict, Any, Optional, Sequence
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import UploadFile, HTTPException

from .models import Document, User, DocumentChunk, Query
from .ai_integration import ai_service
from .database import engine

class DocumentService:
    """Document management service"""
    
    @staticmethod
    async def create_document_record(
        db: AsyncSession, 
        file: UploadFile, 
        user_id: int
    ) -> Document:
        """Create document record in database"""
        
        # Generate unique filename
        filename = file.filename or "unknown"
        file_extension = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = f"uploads/{unique_filename}"
        
        # Save file
        os.makedirs("uploads", exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Create database record
        document = Document(
            filename=unique_filename,
            original_filename=file.filename or "unknown",
            file_path=file_path,
            file_size=len(content),
            mime_type=file.content_type or "application/octet-stream",
            owner_id=user_id,
            status="pending"
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        return document
    
    @staticmethod
    async def get_user_documents(
        db: AsyncSession, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 100
    ) -> Sequence[Document]:
        """Get user's documents with pagination"""
        
        query = (
            select(Document)
            .where(Document.owner_id == user_id)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        
        result = await db.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def get_document_by_id(
        db: AsyncSession, 
        document_id: int, 
        user_id: int
    ) -> Optional[Document]:
        """Get document by ID (user must own it)"""
        
        query = (
            select(Document)
            .where(Document.id == document_id)
            .where(Document.owner_id == user_id)
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_processed_documents(
        db: AsyncSession, 
        user_id: int
    ) -> Sequence[Document]:
        """Get user's processed documents"""
        
        query = (
            select(Document)
            .where(Document.owner_id == user_id)
            .where(Document.status == "completed")
        )
        
        result = await db.execute(query)
        return result.scalars().all()

class ValidationService:
    """Document validation service"""
    
    @staticmethod
    async def validate_document(file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded document"""
        
        errors = []
        warnings = []
        
        # File size validation (100MB limit)
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE:
            errors.append(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE} bytes)")
        
        if file_size == 0:
            errors.append("File is empty")
        
        # File type validation
        ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.xlsx', '.csv']
        filename = file.filename or ""
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension not in ALLOWED_EXTENSIONS:
            errors.append(f"Unsupported file type: {file_extension}")
        
        # MIME type validation
        expected_mime_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.csv': 'text/csv'
        }
        
        expected_mime = expected_mime_types.get(file_extension)
        if expected_mime and file.content_type != expected_mime:
            warnings.append(f"MIME type mismatch: expected {expected_mime}, got {file.content_type}")
        
        # Reset file position
        await file.seek(0)
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "file_size": file_size
        }

class RAGService:
    """RAG query service"""
    
    @staticmethod
    async def get_user_rag_system(user_id: int):
        """Get or create RAG system for user"""
        return UserRAGSystem(user_id)

class UserRAGSystem:
    """User-specific RAG system"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
    
    async def process_query(self, query: str, filters: Optional[Dict] = None):
        """Process RAG query"""
        
        # Use the AI integration bridge
        from .ai_integration import query_rag_with_ai
        result = await query_rag_with_ai(query, self.user_id, filters)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return QueryResult(
            response=result["response"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0)
        )

class QueryResult:
    """RAG query result"""
    
    def __init__(self, response: str, sources: List, confidence: float):
        self.response = response
        self.sources = sources
        self.confidence = confidence

class UserService:
    """User management service"""
    
    @staticmethod
    async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
        """Get user by username"""
        query = select(User).where(User.username == username)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def create_user(
        db: AsyncSession, 
        username: str, 
        email: str, 
        hashed_password: str
    ) -> User:
        """Create new user"""
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_active=True
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return user

class AnalyticsService:
    """Analytics and metrics service"""
    
    @staticmethod
    async def log_user_activity(user_id: int, activity_type: str, metadata: Optional[Dict] = None):
        """Log user activity"""
        from .monitoring import metrics_collector
        await metrics_collector.log_user_activity(user_id, activity_type, metadata)
    
    @staticmethod
    async def get_user_stats(db: AsyncSession, user_id: int) -> Dict[str, Any]:
        """Get user statistics"""
        
        # Document count
        doc_query = select(Document).where(Document.owner_id == user_id)
        doc_result = await db.execute(doc_query)
        documents = doc_result.scalars().all()
        
        # Query count  
        query_query = select(Query).where(Query.user_id == user_id)
        query_result = await db.execute(query_query)
        queries = query_result.scalars().all()
        
        return {
            "total_documents": len(documents),
            "processed_documents": len([d for d in documents if d.status == "completed"]),
            "total_queries": len(queries),
            "total_chunks": sum(d.chunk_count for d in documents),
            "storage_used_mb": sum(d.file_size for d in documents) / (1024 * 1024)
        }

# Convenience functions for main.py
async def create_document_record(db, file, user_id):
    return await DocumentService.create_document_record(db, file, user_id)

async def validate_document(file):
    return await ValidationService.validate_document(file)

async def get_user_documents(db, user_id, skip: int = 0, limit: int = 100):
    return await DocumentService.get_user_documents(db, user_id, skip, limit)

async def get_document_by_id(db, document_id, user_id):
    return await DocumentService.get_document_by_id(db, document_id, user_id)

async def get_processed_documents(db, user_id):
    return await DocumentService.get_processed_documents(db, user_id)

async def get_user_rag_system(user_id: int):
    return await RAGService.get_user_rag_system(user_id)