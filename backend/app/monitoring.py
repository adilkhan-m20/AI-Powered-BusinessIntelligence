
# app/monitoring.py - Fixed Advanced Monitoring & Analytics System
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import psutil
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Comprehensive system monitoring and analytics"""

    def __init__(self):
        # In-memory metrics (last 24 hours)
        self.request_metrics = deque(maxlen=86400)  # 1 per second
        self.user_activity = defaultdict(list)
        self.system_metrics = deque(maxlen=1440)  # 1 per minute
        self.error_metrics = defaultdict(int)

        # Performance tracking
        self.response_times = defaultdict(list)
        self.endpoint_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "errors": 0})

    async def initialize(self):
        """Initialize monitoring system"""
        # Start background collection tasks
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._cleanup_old_metrics())
        logger.info("✅ Monitoring system initialized")

    async def cleanup(self):
        """Cleanup monitoring resources"""
        logger.info("🧹 Cleaning up monitoring resources")

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
            "date": datetime.utcnow().isoformat(),
        }

        # Store in memory
        self.request_metrics.append(metric)

    async def log_response(self, endpoint: str, status_code: int, response_time: float, user_id: Optional[int] = None):
        """Log API response"""
        # Update endpoint statistics
        self.endpoint_stats[endpoint]["count"] += 1
        self.endpoint_stats[endpoint]["total_time"] += float(response_time)

        if status_code >= 400:
            self.endpoint_stats[endpoint]["errors"] += 1

        # Store response time
        self.response_times[endpoint].append(response_time)

        # Keep only last 1000 response times per endpoint
        if len(self.response_times[endpoint]) > 1000:
            self.response_times[endpoint] = self.response_times[endpoint][-1000:]

    # User Activity Tracking
    async def log_user_login(self, user_id: int, ip_address: Optional[str] = None):
        """Log user login event"""
        login_event = {
            "user_id": user_id,
            "event": "login",
            "ip_address": ip_address,
            "timestamp": self.get_timestamp(),
            "date": datetime.utcnow().isoformat(),
        }

        # Track user activity
        self.user_activity[user_id].append(login_event["timestamp"])

        # Keep only last 100 activities per user
        if len(self.user_activity[user_id]) > 100:
            self.user_activity[user_id] = self.user_activity[user_id][-100:]

    async def log_document_upload(self, user_id: int, document_id: int, file_size: int):
        """Log document upload event"""
        upload_event = {
            "user_id": user_id,
            "event": "document_upload",
            "document_id": document_id,
            "file_size": file_size,
            "timestamp": self.get_timestamp(),
            "date": datetime.utcnow().isoformat(),
        }

    async def log_query(self, user_id: int, query: str, processing_time: float):
        """Log RAG query event"""
        query_event = {
            "user_id": user_id,
            "event": "rag_query",
            "query_length": len(query),
            "processing_time": processing_time,
            "timestamp": self.get_timestamp(),
            "date": datetime.utcnow().isoformat(),
        }

    async def log_task_completion(self, task_type: str, processing_time: float, success: bool):
        """Log task completion event"""
        task_event = {
            "task_type": task_type,
            "processing_time": processing_time,
            "success": success,
            "timestamp": self.get_timestamp(),
            "date": datetime.utcnow().isoformat(),
        }

    # System Metrics Collection
    async def _collect_system_metrics(self):
        """Background task to collect system metrics"""
        while True:
            try:
                # CPU and Memory usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                try:
                    disk = psutil.disk_usage("/")
                except:
                    # Fallback for Windows
                    disk = psutil.disk_usage("C:\\")

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
                    "date": datetime.utcnow().isoformat(),
                }

                # Store in memory
                self.system_metrics.append(system_metric)

                # Wait 60 seconds before next collection
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)

    # Analytics and Reporting
    async def get_user_dashboard(self, user_id: int) -> Dict[str, Any]:
        """Get user dashboard metrics"""
        # Calculate metrics from in-memory data
        total_documents = 0
        total_queries = 0
        recent_activity_count = 0
        last_activity = None
        avg_query_time = 0.0

        # Get user activity
        if user_id in self.user_activity:
            activity_times = self.user_activity[user_id]
            total_documents = sum(1 for t in activity_times if t > time.time() - 86400)
            total_queries = sum(1 for t in activity_times if t > time.time() - 86400)
            
            if activity_times:
                last_activity = datetime.fromtimestamp(activity_times[-1]).isoformat()
                recent_activity_count = len([t for t in activity_times if t > time.time() - 604800])  # Last 7 days

        return {
            "user_id": user_id,
            "total_documents": total_documents,
            "total_queries": total_queries,
            "recent_activity_count": recent_activity_count,
            "last_activity": last_activity,
            "avg_query_time": avg_query_time,
            "storage_used_mb": 0  # This would be calculated from database
        }

    async def get_usage_metrics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get user usage metrics for specified period"""
        # Group by day
        daily_activity = defaultdict(int)
        
        # Calculate trends
        document_uploads = 0
        queries = 0

        most_active_day = None
        if daily_activity:
            most_active_day = max(daily_activity.keys(), key=lambda k: daily_activity[k])

        return {
            "period_days": days,
            "total_activity": document_uploads + queries,
            "document_uploads": document_uploads,
            "queries": queries,
            "daily_activity": dict(daily_activity),
            "most_active_day": most_active_day,
        }

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        # Calculate averages
        avg_cpu = 0
        avg_memory = 0
        current_cpu = 0
        current_memory = 0
        
        if self.system_metrics:
            avg_cpu = sum(m["cpu_percent"] for m in self.system_metrics) / len(self.system_metrics)
            avg_memory = sum(m["memory_percent"] for m in self.system_metrics) / len(self.system_metrics)
            current_cpu = self.system_metrics[0]["cpu_percent"]
            current_memory = self.system_metrics[0]["memory_percent"]

        # Request stats
        total_requests = len(self.request_metrics)
        unique_users = len(set(m["user_id"] for m in self.request_metrics if m.get("user_id")))

        return {
            "system": {
                "avg_cpu_percent": round(avg_cpu, 2),
                "avg_memory_percent": round(avg_memory, 2),
                "current_cpu": current_cpu,
                "current_memory": current_memory,
            },
            "requests": {
                "total_requests": total_requests,
                "unique_users": unique_users,
                "requests_per_user": round(total_requests / unique_users, 2) if unique_users else 0,
            },
            "endpoints": dict(self.endpoint_stats),
            "errors": dict(self.error_metrics),
        }

# Global metrics collector instance
metrics_collector = MetricsCollector()