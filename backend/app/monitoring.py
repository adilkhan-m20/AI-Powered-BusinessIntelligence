
# app/monitoring.py - Advanced Monitoring & Analytics System
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import psutil
import redis.asyncio as aioredis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db

class MetricsCollector:
    """Comprehensive system monitoring and analytics"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        
        # In-memory metrics (last 24 hours)
        self.request_metrics = deque(maxlen=86400)  # 1 per second
        self.user_activity = defaultdict(list)
        self.system_metrics = deque(maxlen=1440)  # 1 per minute
        self.error_metrics = defaultdict(int)
        
        # Performance tracking
        self.response_times = defaultdict(list)
        self.endpoint_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        
    async def initialize(self):
        """Initialize monitoring system"""
        self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
        
        # Start background collection tasks
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._cleanup_old_metrics())
        
    async def cleanup(self):
        """Cleanup monitoring resources"""
        if self.redis:
            await self.redis.close()
    
    def get_timestamp(self) -> float:
        """Get current timestamp"""
        return time.time()
    
    # Request Metrics
    async def log_request(self, method: str, endpoint: str, user_id: Optional[int] = None):
        """Log API request"""
        timestamp = self.get_timestamp()
        
        metric = {
            "timestamp": timestamp,
            "method": method,
            "endpoint": endpoint,
            "user_id": user_id,
            "date": datetime.utcnow().isoformat()
        }
        
        # Store in memory
        self.request_metrics.append(metric)
        
        # Store in Redis with expiration
        await self.redis.lpush("request_metrics", json.dumps(metric))
        await self.redis.expire("request_metrics", 86400)  # 24 hours
        
        # Track user activity
        if user_id:
            self.user_activity[user_id].append(timestamp)
            
            # Keep only last 100 activities per user
            if len(self.user_activity[user_id]) > 100:
                self.user_activity[user_id] = self.user_activity[user_id][-100:]
    
    async def log_response(self, endpoint: str, status_code: int, response_time: float, user_id: Optional[int] = None):
        """Log API response"""
        # Update endpoint statistics
        self.endpoint_stats[endpoint]["count"] += 1
        self.endpoint_stats[endpoint]["total_time"] += response_time
        
        if status_code >= 400:
            self.endpoint_stats[endpoint]["errors"] += 1
            self.error_metrics[f"{status_code}_{endpoint}"] += 1
        
        # Store response time
        self.response_times[endpoint].append(response_time)
        
        # Keep only last 1000 response times per endpoint
        if len(self.response_times[endpoint]) > 1000:
            self.response_times[endpoint] = self.response_times[endpoint][-1000:]
        
        # Store detailed metric in Redis
        response_metric = {
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time": response_time,
            "user_id": user_id,
            "timestamp": self.get_timestamp(),
            "date": datetime.utcnow().isoformat()
        }
        
        await self.redis.lpush("response_metrics", json.dumps(response_metric))
        await self.redis.expire("response_metrics", 86400)
    
    # User Activity Tracking
    async def log_user_login(self, user_id: int, ip_address: str = None):
        """Log user login event"""
        login_event = {
            "user_id": user_id,
            "event": "login",
            "ip_address": ip_address,
            "timestamp": self.get_timestamp(),
            "date": datetime.utcnow().isoformat()
        }
        
        await self.redis.lpush(f"user_events:{user_id}", json.dumps(login_event))
        await self.redis.expire(f"user_events:{user_id}", 2592000)  # 30 days
    
    async def log_document_upload(self, user_id: int, document_id: int, file_size: int):
        """Log document upload event"""
        upload_event = {
            "user_id": user_id,
            "event": "document_upload",
            "document_id": document_id,
            "file_size": file_size,
            "timestamp": self.get_timestamp(),
            "date": datetime.utcnow().isoformat()
        }
        
        await self.redis.lpush(f"user_events:{user_id}", json.dumps(upload_event))
        await self.redis.lpush("document_events", json.dumps(upload_event))
        await self.redis.expire("document_events", 86400)
    
    async def log_query(self, user_id: int, query: str, processing_time: float):
        """Log RAG query event"""
        query_event = {
            "user_id": user_id,
            "event": "rag_query",
            "query_length": len(query),
            "processing_time": processing_time,
            "timestamp": self.get_timestamp(),
            "date": datetime.utcnow().isoformat()
        }
        
        await self.redis.lpush(f"user_events:{user_id}", json.dumps(query_event))
        await self.redis.lpush("query_metrics", json.dumps(query_event))
        await self.redis.expire("query_metrics", 86400)
    
    async def log_task_completion(self, task_type: str, processing_time: float, success: bool):
        """Log task completion event"""
        task_event = {
            "task_type": task_type,
            "processing_time": processing_time,
            "success": success,
            "timestamp": self.get_timestamp(),
            "date": datetime.utcnow().isoformat()
        }
        
        await self.redis.lpush("task_metrics", json.dumps(task_event))
        await self.redis.expire("task_metrics", 86400)
    
    # System Metrics Collection
    async def _collect_system_metrics(self):
        """Background task to collect system metrics"""
        while True:
            try:
                # CPU and Memory usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network stats
                network = psutil.net_io_counters()
                
                system_metric = {
                    "timestamp": self.get_timestamp(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": (disk.used / disk.total) * 100,
                    "disk_used_gb": disk.used / (1024**3),
                    "disk_total_gb": disk.total / (1024**3),
                    "network_sent_gb": network.bytes_sent / (1024**3),
                    "network_recv_gb": network.bytes_recv / (1024**3),
                    "date": datetime.utcnow().isoformat()
                }
                
                # Store in memory
                self.system_metrics.append(system_metric)
                
                # Store in Redis
                await self.redis.lpush("system_metrics", json.dumps(system_metric))
                await self.redis.expire("system_metrics", 86400)
                
                # Check for alerts
                await self._check_system_alerts(system_metric)
                
                # Wait 60 seconds before next collection
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
    
    async def _check_system_alerts(self, metrics: Dict[str, Any]):
        """Check for system alert conditions"""
        alerts = []
        
        # High CPU usage
        if metrics["cpu_percent"] > 80:
            alerts.append({
                "type": "high_cpu",
                "value": metrics["cpu_percent"],
                "threshold": 80,
                "severity": "warning"
            })
        
        # High memory usage
        if metrics["memory_percent"] > 85:
            alerts.append({
                "type": "high_memory",
                "value": metrics["memory_percent"],
                "threshold": 85,
                "severity": "warning"
            })
        
        # High disk usage
        if metrics["disk_percent"] > 90:
            alerts.append({
                "type": "high_disk",
                "value": metrics["disk_percent"],
                "threshold": 90,
                "severity": "critical"
            })
        
        # Store alerts if any
        if alerts:
            alert_event = {
                "timestamp": self.get_timestamp(),
                "alerts": alerts,
                "date": datetime.utcnow().isoformat()
            }
            
            await self.redis.lpush("system_alerts", json.dumps(alert_event))
            await self.redis.expire("system_alerts", 86400)
    
    # Analytics and Reporting
    async def get_user_dashboard(self, user_id: int) -> Dict[str, Any]:
        """Get user dashboard metrics"""
        # Get user events from Redis
        user_events = await self.redis.lrange(f"user_events:{user_id}", 0, -1)
        
        # Parse events
        events = []
        for event_json in user_events:
            events.append(json.loads(event_json))
        
        # Calculate metrics
        total_documents = len([e for e in events if e["event"] == "document_upload"])
        total_queries = len([e for e in events if e["event"] == "rag_query"])
        
        # Recent activity (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_events = [
            e for e in events 
            if datetime.fromisoformat(e["date"]) > week_ago
        ]
        
        return {
            "user_id": user_id,
            "total_documents": total_documents,
            "total_queries": total_queries,
            "recent_activity_count": len(recent_events),
            "last_activity": events[0]["date"] if events else None,
            "avg_query_time": self._calculate_avg_query_time(events)
        }
    
    async def get_usage_metrics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get user usage metrics for specified period"""
        # Get user events
        user_events = await self.redis.lrange(f"user_events:{user_id}", 0, -1)
        
        # Filter by date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filtered_events = []
        
        for event_json in user_events:
            event = json.loads(event_json)
            if datetime.fromisoformat(event["date"]) > cutoff_date:
                filtered_events.append(event)
        
        # Group by day
        daily_activity = defaultdict(int)
        for event in filtered_events:
            day = datetime.fromisoformat(event["date"]).date().isoformat()
            daily_activity[day] += 1
        
        # Calculate trends
        document_uploads = len([e for e in filtered_events if e["event"] == "document_upload"])
        queries = len([e for e in filtered_events if e["event"] == "rag_query"])
        
        return {
            "period_days": days,
            "total_activity": len(filtered_events),
            "document_uploads": document_uploads,
            "queries": queries,
            "daily_activity": dict(daily_activity),
            "most_active_day": max(daily_activity.keys(), key=daily_activity.get) if daily_activity else None
        }
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        # Get recent system metrics
        system_metrics = await self.redis.lrange("system_metrics", 0, 59)  # Last hour
        
        # Get request metrics
        request_metrics = await self.redis.lrange("request_metrics", 0, 999)  # Last 1000 requests
        
        # Parse and analyze
        parsed_system = [json.loads(m) for m in system_metrics]
        parsed_requests = [json.loads(m) for m in request_metrics]
        
        # Calculate averages
        avg_cpu = sum(m["cpu_percent"] for m in parsed_system) / len(parsed_system) if parsed_system else 0
        avg_memory = sum(m["memory_percent"] for m in parsed_system) / len(parsed_system) if parsed_system else 0
        
        # Request stats
        total_requests = len(parsed_requests)
        unique_users = len(set(m["user_id"] for m in parsed_requests if m["user_id"]))
        
        return {
            "system": {
                "avg_cpu_percent": round(avg_cpu, 2),
                "avg_memory_percent": round(avg_memory, 2),
                "current_cpu": parsed_system[0]["cpu_percent"] if parsed_system else 0,
                "current_memory": parsed_system[0]["memory_percent"] if parsed_system else 0
            },
            "requests": {
                "total_requests": total_requests,
                "unique_users": unique_users,
                "requests_per_user": round(total_requests / unique_users, 2) if unique_users else 0
            },
            "endpoints": dict(self.endpoint_stats),
            "errors": dict(self.error_metrics)
        }
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        # Response time analysis
        performance_data = {}
        
        for endpoint, times in self.response_times.items():
            if times:
                performance_data[endpoint] = {
                    "avg_response_time": round(sum(times) / len(times), 3),
                    "min_response_time": round(min(times), 3),
                    "max_response_time": round(max(times), 3),
                    "total_requests": len(times)
                }
        
        # Get slow queries (>5 seconds)
        query_metrics = await self.redis.lrange("query_metrics", 0, -1)
        slow_queries = []
        
        for query_json in query_metrics:
            query = json.loads(query_json)
            if query["processing_time"] > 5.0:
                slow_queries.append({
                    "user_id": query["user_id"],
                    "processing_time": query["processing_time"],
                    "query_length": query["query_length"],
                    "date": query["date"]
                })
        
        return {
            "endpoint_performance": performance_data,
            "slow_queries": slow_queries[-50:],  # Last 50 slow queries
            "total_slow_queries": len(slow_queries)
        }
    
    # Utility methods
    def _calculate_avg_query_time(self, events: List[Dict[str, Any]]) -> float:
        """Calculate average query processing time"""
        query_events = [e for e in events if e["event"] == "rag_query"]
        
        if not query_events:
            return 0.0
        
        total_time = sum(e.get("processing_time", 0) for e in query_events)
        return round(total_time / len(query_events), 3)
    
    async def _cleanup_old_metrics(self):
        """Background task to cleanup old metrics"""
        while True:
            try:
                # Clean up metrics older than 24 hours
                cutoff = self.get_timestamp() - 86400
                
                # This is a simplified cleanup - in production, you might want
                # more sophisticated cleanup based on Redis sorted sets
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                print(f"Error in metrics cleanup: {e}")
                await asyncio.sleep(3600)

# Global metrics collector instance
metrics_collector = MetricsCollector()