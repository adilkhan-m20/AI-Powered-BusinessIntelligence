
# backend/run.py - Application Launcher
import os
import asyncio
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main application launcher"""
    
    # Configuration from environment
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print("🚀 Starting MultiModal AI Backend...")
    print(f"📡 Server: http://{host}:{port}")
    print(f"📖 API Docs: http://{host}:{port}/docs")
    print(f"🔄 Interactive Docs: http://{host}:{port}/redoc")
    print(f"🔧 Debug Mode: {debug}")
    
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