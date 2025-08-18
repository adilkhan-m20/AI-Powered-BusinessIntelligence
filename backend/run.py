
# backend/run.py - Application Launcher
import os
import asyncio
import uvicorn
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Main application launcher"""
    
    # Configuration from environment
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info("🚀 Starting MultiModal AI Backend...")
    logger.info(f"📡 Server: http://{host}:{port}")
    logger.info(f"📖 API Docs: http://{host}:{port}/docs")
    logger.info(f"🔄 Interactive Docs: http://{host}:{port}/redoc")
    logger.info(f"🔧 Debug Mode: {debug}")
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning",
        access_log=debug
    )

if __name__ == "__main__":
    main()