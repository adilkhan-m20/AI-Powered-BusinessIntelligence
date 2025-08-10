
# app/websocket_manager.py - Real-time WebSocket Connection Manager
import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Advanced WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        # User ID -> List of WebSocket connections
        self.active_connections: Dict[int, List[WebSocket]] = defaultdict(list)
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Room-based connections (for future group features)
        self.rooms: Dict[str, Set[int]] = defaultdict(set)
        
        # Message queue for offline users
        self.offline_messages: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        
        # Connection statistics
        self.stats = {
            "total_connections": 0,
            "active_users": 0,
            "messages_sent": 0,
            "messages_failed": 0
        }
    
    async def connect(self, user_id: int, websocket: WebSocket) -> bool:
        """Accept WebSocket connection and register user"""
        try:
            await websocket.accept()
            
            # Add connection
            self.active_connections[user_id].append(websocket)
            
            # Store metadata
            self.connection_metadata[websocket] = {
                "user_id": user_id,
                "connected_at": datetime.utcnow(),
                "last_ping": datetime.utcnow()
            }
            
            # Update stats
            self.stats["total_connections"] += 1
            self.stats["active_users"] = len(self.active_connections)
            
            # Send any queued offline messages
            await self._deliver_offline_messages(user_id)
            
            # Notify user of successful connection
            await self.send_to_user(user_id, {
                "type": "connection_established",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Real-time updates enabled"
            })
            
            logger.info(f"User {user_id} connected via WebSocket")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect user {user_id}: {e}")
            return False
    
    async def disconnect(self, user_id: int, websocket: WebSocket = None):
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
        
        # Update stats
        self.stats["active_users"] = len(self.active_connections)
        
        logger.info(f"User {user_id} disconnected from WebSocket")
    
    async def send_to_user(self, user_id: int, message: Dict[str, Any]) -> bool:
        """Send message to specific user"""
        if user_id not in self.active_connections:
            # User offline - queue message
            await self._queue_offline_message(user_id, message)
            return False
        
        success_count = 0
        failed_connections = []
        
        # Send to all user connections
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_text(json.dumps(message))
                success_count += 1
                self.stats["messages_sent"] += 1
                
            except Exception as e:
                logger.error(f"Failed to send message to user {user_id}: {e}")
                failed_connections.append(websocket)
                self.stats["messages_failed"] += 1
        
        # Clean up failed connections
        for failed_conn in failed_connections:
            await self.disconnect(user_id, failed_conn)
        
        return success_count > 0
    
    async def send_to_all(self, message: Dict[str, Any], exclude_users: List[int] = None) -> int:
        """Send message to all connected users"""
        exclude_users = exclude_users or []
        sent_count = 0
        
        for user_id in list(self.active_connections.keys()):
            if user_id not in exclude_users:
                success = await self.send_to_user(user_id, message)
                if success:
                    sent_count += 1
        
        return sent_count
    
    async def send_to_room(self, room_name: str, message: Dict[str, Any]) -> int:
        """Send message to all users in a room"""
        if room_name not in self.rooms:
            return 0
        
        sent_count = 0
        for user_id in self.rooms[room_name]:
            success = await self.send_to_user(user_id, message)
            if success:
                sent_count += 1
        
        return sent_count
    
    async def join_room(self, user_id: int, room_name: str):
        """Add user to a room"""
        self.rooms[room_name].add(user_id)
        
        await self.send_to_user(user_id, {
            "type": "room_joined",
            "room": room_name,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def leave_room(self, user_id: int, room_name: str):
        """Remove user from a room"""
        if room_name in self.rooms:
            self.rooms[room_name].discard(user_id)
            
            # Clean empty rooms
            if not self.rooms[room_name]:
                del self.rooms[room_name]
        
        await self.send_to_user(user_id, {
            "type": "room_left", 
            "room": room_name,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _deliver_offline_messages(self, user_id: int):
        """Deliver queued messages to newly connected user"""
        if user_id in self.offline_messages:
            messages = self.offline_messages[user_id]
            
            for message in messages:
                await self.send_to_user(user_id, message)
            
            # Clear delivered messages
            del self.offline_messages[user_id]
    
    async def _queue_offline_message(self, user_id: int, message: Dict[str, Any]):
        """Queue message for offline user"""
        # Add timestamp and mark as queued
        message.update({
            "queued_at": datetime.utcnow().isoformat(),
            "is_queued_message": True
        })
        
        self.offline_messages[user_id].append(message)
        
        # Limit queue size to prevent memory issues
        if len(self.offline_messages[user_id]) > 100:
            self.offline_messages[user_id] = self.offline_messages[user_id][-100:]
    
    async def ping_all_connections(self):
        """Send ping to all connections to check health"""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        dead_connections = []
        
        for user_id, connections in self.active_connections.items():
            for websocket in connections:
                try:
                    await websocket.send_text(json.dumps(ping_message))
                    
                    # Update last ping time
                    if websocket in self.connection_metadata:
                        self.connection_metadata[websocket]["last_ping"] = datetime.utcnow()
                
                except Exception:
                    dead_connections.append((user_id, websocket))
        
        # Clean up dead connections
        for user_id, websocket in dead_connections:
            await self.disconnect(user_id, websocket)
    
    async def cleanup_stale_connections(self, max_age_minutes: int = 60):
        """Remove connections that haven't pinged recently"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        stale_connections = []
        
        for websocket, metadata in self.connection_metadata.items():
            if metadata["last_ping"] < cutoff_time:
                stale_connections.append((metadata["user_id"], websocket))
        
        for user_id, websocket in stale_connections:
            await self.disconnect(user_id, websocket)
            logger.info(f"Cleaned up stale connection for user {user_id}")
    
    def get_user_connection_count(self, user_id: int) -> int:
        """Get number of active connections for user"""
        return len(self.active_connections.get(user_id, []))
    
    def get_online_users(self) -> List[int]:
        """Get list of online user IDs"""
        return list(self.active_connections.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            **self.stats,
            "online_users": len(self.active_connections),
            "total_active_connections": sum(len(conns) for conns in self.active_connections.values()),
            "rooms_count": len(self.rooms),
            "queued_messages": sum(len(msgs) for msgs in self.offline_messages.values())
        }
    
    async def broadcast_system_message(self, message: str, level: str = "info"):
        """Send system-wide message to all users"""
        system_message = {
            "type": "system_message",
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_to_all(system_message)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

# Background task to maintain connections
async def websocket_maintenance_task():
    """Background task for WebSocket maintenance"""
    while True:
        try:
            # Ping all connections every 30 seconds
            await websocket_manager.ping_all_connections()
            
            # Clean up stale connections every 5 minutes
            await websocket_manager.cleanup_stale_connections()
            
            # Wait 30 seconds before next maintenance
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"WebSocket maintenance error: {e}")
            await asyncio.sleep(30)