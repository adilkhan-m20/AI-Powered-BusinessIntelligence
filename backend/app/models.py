
# app/models.py - Database Models and Pydantic Schemas
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="owner")
    queries = relationship("Query", back_populates="user")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    
    # Processing status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    task_id = Column(String(100), nullable=True)
    
    # Content analysis
    text_content = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    embedding_count = Column(Integer, default=0)
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)
    validation_errors = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    owner = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    
    # Vector embedding info
    embedding_model = Column(String(100), nullable=True)
    embedding_dimensions = Column(Integer, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign keys
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class Query(Base):
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=True)
    
    # Performance metrics
    processing_time = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    source_count = Column(Integer, nullable=True)
    
    # Context
    filters = Column(JSON, nullable=True)
    context_documents = Column(JSON, nullable=True)  # Document IDs used
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="queries")

class SystemMetric(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_type = Column(String(50), nullable=False)  # cpu, memory, requests, etc.
    metric_value = Column(Float, nullable=False)
    metadata = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)

# Pydantic Models for API
class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserResponse(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True

class UserCredentials(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes
    refresh_token: Optional[str] = None

class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_size: int
    mime_type: str

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: int
    status: DocumentStatus
    task_id: Optional[str]
    chunk_count: int
    embedding_count: int
    quality_score: Optional[float]
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]
    owner_id: int
    
    class Config:
        from_attributes = True

class DocumentUploadResponse(BaseModel):
    document_id: int
    task_id: str
    status: str
    message: str
    estimated_processing_time: Optional[int] = None

class ProcessingStatus(BaseModel):
    document_id: int
    status: DocumentStatus
    progress: int = Field(..., ge=0, le=100)
    message: str
    updated_at: datetime
    estimated_completion: Optional[datetime] = None

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    filters: Optional[Dict[str, Any]] = None
    max_sources: int = Field(default=5, ge=1, le=20)
    include_confidence: bool = True

class RAGSource(BaseModel):
    document_id: int
    document_name: str
    chunk_text: str
    page_number: Optional[int]
    relevance_score: float

class RAGResponse(BaseModel):
    query: str
    response: str
    sources: List[RAGSource]
    confidence: float
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchProcessingRequest(BaseModel):
    document_ids: List[int] = Field(..., min_items=1, max_items=50)
    processing_options: Optional[Dict[str, Any]] = None

class BatchProcessingResponse(BaseModel):
    batch_id: str
    task_id: str
    document_count: int
    estimated_completion: datetime
    status: str

# Analytics Models
class DashboardMetrics(BaseModel):
    user_id: int
    total_documents: int
    total_queries: int
    recent_activity_count: int
    last_activity: Optional[datetime]
    avg_query_time: float
    storage_used_mb: float

class UsageMetrics(BaseModel):
    period_days: int
    total_activity: int
    document_uploads: int
    queries: int
    daily_activity: Dict[str, int]
    most_active_day: Optional[str]

class SystemMetrics(BaseModel):
    system: Dict[str, float]
    requests: Dict[str, Any]
    endpoints: Dict[str, Dict[str, Any]]
    errors: Dict[str, int]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# WebSocket Models
class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class NotificationMessage(BaseModel):
    type: str = "notification"
    title: str
    message: str
    level: str = "info"  # info, warning, error, success
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TaskUpdateMessage(BaseModel):
    type: str = "task_update"
    task_id: str
    status: str
    progress: int = Field(..., ge=0, le=100)
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Validation Models
class DocumentValidation(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    quality_score: float = 0.0

class ValidationRule(BaseModel):
    rule_name: str
    description: str
    is_required: bool
    severity: str  # error, warning

# Settings Models
class AppSettings(BaseModel):
    app_name: str = "MultiModal AI Document Processing"
    version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # AI Models
    openai_api_key: str
    model_name: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # File Upload
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = [".pdf", ".txt", ".docx", ".xlsx"]
    
    # Processing
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    max_concurrent_tasks: int = 5

# Error Models
class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

class ValidationError(BaseModel):
    field: str
    message: str
    value: Any