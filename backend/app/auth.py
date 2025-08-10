
# app/auth.py - JWT Authentication & Security System
import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from .database import get_db
from .models import User

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class AuthService:
    """Comprehensive authentication and authorization service"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a plaintext password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)  # 7 days for refresh
        
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials"""
        # Get user from database
        query = select(User).where(User.username == username)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        if not AuthService.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        await db.commit()
        
        return user
    
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
        """Get user by ID"""
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def create_user(
        db: AsyncSession, 
        username: str, 
        email: str, 
        password: str,
        is_admin: bool = False
    ) -> User:
        """Create new user"""
        hashed_password = AuthService.get_password_hash(password)
        
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_admin=is_admin,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return user

# Authentication dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    
    # Decode token
    payload = AuthService.decode_token(credentials.credentials)
    
    # Validate token type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user ID from payload
    user_id: Optional[int] = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = await AuthService.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Admin user dependency
async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current authenticated admin user"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

# Optional authentication (for public endpoints that can benefit from user context)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, otherwise None"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None

# Rate limiting decorator
from functools import wraps
import time
from collections import defaultdict

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(list)
    
    def is_allowed(self, identifier: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is allowed based on rate limit"""
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] 
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        if len(self.requests[identifier]) >= max_requests:
            return False
        
        # Record this request
        self.requests[identifier].append(now)
        return True

rate_limiter = RateLimiter()

def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user from dependencies
            user = None
            for arg in args:
                if isinstance(arg, User):
                    user = arg
                    break
            
            if not user:
                # Extract from kwargs
                user = kwargs.get('current_user') or kwargs.get('user')
            
            if user:
                identifier = f"user_{user.id}"
            else:
                # Use IP address (you'd need to extract this from request)
                identifier = "anonymous"
            
            if not rate_limiter.is_allowed(identifier, max_requests, window_seconds):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Security middleware
class SecurityMiddleware:
    """Additional security features"""
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
        
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is strong"
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Basic input sanitization"""
        import html
        return html.escape(text.strip())
    
    @staticmethod
    def is_safe_filename(filename: str) -> bool:
        """Check if filename is safe"""
        import os
        import re
        
        # Check for directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return False
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            return False
        
        return True

# Utility functions for shortcuts
def create_access_token(data: Dict[str, Any]) -> str:
    """Shortcut for creating access token"""
    return AuthService.create_access_token(data)

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Shortcut for creating refresh token"""
    return AuthService.create_refresh_token(data)

async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """Shortcut for user authentication"""
    return await AuthService.authenticate_user(db, username, password)