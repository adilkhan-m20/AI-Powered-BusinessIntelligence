
# app/websocket_manager.py - Real-time WebSocket Connection Manager
import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketManager:
    """Advanced WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        # User ID -> List of WebSocket connections
        self.active_connections: Dict[int, List[WebSocket]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Message queue for offline users
        self.offline_messages: Dict[int, List[Dict[str, Any]]] = {}
    
    async def connect(self, user_id: int, websocket: WebSocket) -> bool:
        """Accept WebSocket connection and register user"""
        try:
            await websocket.accept()
            
            # Add connection
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
            self.active_connections[user_id].append(websocket)
            
            # Store metadata
            self.connection_metadata[websocket] = {
                "user_id": user_id,
                "connected_at": datetime.utcnow(),
                "last_ping": datetime.utcnow()
            }
            
            logger.info(f"👤 User {user_id} connected via WebSocket")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect user {user_id}: {e}")
            return False
    
    async def disconnect(self, user_id: int, websocket: Optional[WebSocket] = None):
        """Disconnect user's WebSocket connections"""
        if websocket:
            # Remove specific connection
            if user_id in self.active_connections:
                self.active_connections[user_id] = [
                    conn for conn in self.active_connections[user_id] 
                    if conn != websocket
                ]
                
                # Remove user if no connections left
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            
            # Clean metadata
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
        
        else:
            # Remove all connections for user
            if user_id in self.active_connections:
                # Clean metadata for all user connections
                for conn in self.active_connections[user_id]:
                    if conn in self.connection_metadata:
                        del self.connection_metadata[conn]
                
                del self.active_connections[user_id]
        
        logger.info(f"🚪 User {user_id} disconnected from WebSocket")
    
    async def send_to_user(self, user_id: int, message: Dict[str, Any]) -> int:
        """Send message to specific user"""
        if user_id not in self.active_connections:
            # User offline - queue message
            if user_id not in self.offline_messages:
                self.offline_messages[user_id] = []
            self.offline_messages[user_id].append(message)
            return False
        
        success_count = 0
        failed_connections = []
        
        # Send to all user connections
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_text(json.dumps(message))
                success_count += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to send message to user {user_id}: {e}")
                failed_connections.append(websocket)
        
        # Clean up failed connections
        for failed_conn in failed_connections:
            await self.disconnect(user_id, failed_conn)
        
        return success_count
    
    async def send_to_all(self, message: Dict[str, Any], exclude_users: Optional[List[int]] = None):
        """Send message to all connected users"""
        exclude_users = exclude_users or []
        sent_count = 0
        
        for user_id in list(self.active_connections.keys()):
            if user_id not in exclude_users:
                success = await self.send_to_user(user_id, message)
                if success:
                    sent_count += 1
        
        return sent_count
    
    async def _deliver_offline_messages(self, user_id: int):
        """Deliver queued messages to newly connected user"""
        if user_id in self.offline_messages:
            messages = self.offline_messages[user_id]
            
            for message in messages:
                await self.send_to_user(user_id, message)
            
            # Clear delivered messages
            del self.offline_messages[user_id]

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

# Start background maintenance task
async def start_websocket_maintenance():
    """Start background task for WebSocket maintenance"""
    while True:
        try:
            # Clean up stale connections every 5 minutes
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"WebSocket maintenance error: {e}")
            await asyncio.sleep(30)