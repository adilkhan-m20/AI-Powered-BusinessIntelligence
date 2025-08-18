
import requests
import mimetypes
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredExcelLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default database path
DB_PATH = "faiss_index"

def getFileName(response):
    """Extract filename from response headers"""
    content_disposition = response.headers.get("Content-Disposition", "")
    filename = None
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[-1].strip('"')
    return filename

def fetch_text_from_api(url):
    """Fetch text content from API endpoint"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "").lower()
            filename = getFileName(response)
            if content_type.startswith("text/") and filename and not filename.endswith(".txt"):
                return response.text
        return None
    except Exception as e:
        logger.error(f"Error fetching from API: {e}")
        return None

def upsert_documents(documents, db_path=None):
    """
    Creates a FAISS DB if it doesn't exist,
    else loads and updates the existing DB.
    """
    if db_path is None:
        db_path = DB_PATH
    
    # Use a local embedding model instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        if os.path.exists(db_path):
            # Load existing DB
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("✅ Loaded existing FAISS DB.")
            
            # Add new docs
            if documents:
                db.add_documents(documents)
                logger.info(f"✅ Added {len(documents)} new documents.")
        else:
            # Create new DB
            if documents:
                db = FAISS.from_documents(documents, embeddings)
                logger.info("✅ Created new FAISS DB.")
            else:
                logger.warning("⚠️ No documents to create FAISS DB with.")
                return

        # Save changes
        db.save_local(db_path)
        logger.info("✅ Saved FAISS DB.")
        
    except Exception as e:
        logger.error(f"❌ Error with FAISS operations: {e}")
        raise

def process_document_from_path(file_path, db_path=None):
    """Process a document from file path"""
    try:
        # Determine file type and create appropriate loader
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path)
        else:
            logger.warning(f"⚠️ Unsupported file type: {ext}")
            return None
        
        # Load and process document
        docs = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = text_splitter.split_documents(docs)
        
        # Update FAISS database
        upsert_documents(documents, db_path)
        
        return {
            "original_docs": len(docs),
            "chunks": len(documents),
            "embeddings": len(documents),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing document {file_path}: {e}")
        return {
            "original_docs": 0,
            "chunks": 0,
            "embeddings": 0,
            "success": False,
            "error": str(e)
        }

def process_document_pipeline(document_id, file_path, db_path=None):
    """Complete document processing pipeline"""
    logger.info(f"🔄 Processing document {document_id}: {file_path}")
    result = process_document_from_path(file_path, db_path)
    
    if result and result.get("success"):
        logger.info(f"✅ Successfully processed document {document_id}")
        return {
            "chunks": result["chunks"],
            "embeddings": result["embeddings"],
            "stats": {
                "original_docs": result["original_docs"],
                "total_chunks": result["chunks"],
                "processing_time": 0  # You can add timing if needed
            }
        }
    else:
        error = result.get("error", "Unknown error") if result else "Unknown error"
        logger.error(f"❌ Failed to process document {document_id}: {error}")
        raise Exception(f"Document processing failed: {error}")