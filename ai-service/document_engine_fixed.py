
import requests
import mimetypes
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredExcelLoader
import io
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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
        print(f"Error fetching from API: {e}")
        return None

def upsert_documents(documents, db_path=None):
    """
    Creates a FAISS DB if it doesn't exist,
    else loads and updates the existing DB.
    """
    if db_path is None:
        db_path = DB_PATH
    
    embeddings = OpenAIEmbeddings()
    
    try:
        if os.path.exists(db_path):
            # Load existing DB
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            print("✅ Loaded existing FAISS DB.")
            
            # Add new docs
            if documents:
                db.add_documents(documents)
                print(f"✅ Added {len(documents)} new documents.")
        else:
            # Create new DB
            if documents:
                db = FAISS.from_documents(documents, embeddings)
                print("✅ Created new FAISS DB.")
            else:
                print("⚠️ No documents to create FAISS DB with.")
                return

        # Save changes
        db.save_local(db_path)
        print("✅ Saved FAISS DB.")
        
    except Exception as e:
        print(f"❌ Error with FAISS operations: {e}")
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
            print(f"⚠️ Unsupported file type: {ext}")
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
        print(f"❌ Error processing document {file_path}: {e}")
        return {
            "original_docs": 0,
            "chunks": 0,
            "embeddings": 0,
            "success": False,
            "error": str(e)
        }

def process_document_pipeline(document_id, file_path, db_path=None):
    """Complete document processing pipeline"""
    print(f"🔄 Processing document {document_id}: {file_path}")
    result = process_document_from_path(file_path, db_path)
    
    if result and result.get("success"):
        print(f"✅ Successfully processed document {document_id}")
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
        print(f"❌ Failed to process document {document_id}: {error}")
        raise Exception(f"Document processing failed: {error}")

# Main execution (only runs if called directly)
if __name__ == "__main__":
    try:
        url = "http://localhost:3000/data"  # express api endpoint
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "").lower()
            filename = getFileName(response)
            
            # if no file name guess based on content type
            if not filename:
                extension = mimetypes.guess_extension(content_type.split(";")[0]) or ""
                filename = "downloaded_file" + (extension if extension else "")
            
            # Detect and handle content
            if "application/json" in content_type:
                json_obj = response.json()
                print("Json Data:", json_obj)
                
            elif content_type.startswith("text/") and filename and not filename.endswith(".txt"):
                # plain text from api
                print("Text Data:", response.text)
                
            elif filename.endswith(".txt") or "text/plain" in content_type:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response.text)
                loader = TextLoader(filename, encoding='utf-8')
                
            elif filename.endswith(".pdf") or "application/pdf" in content_type:
                with open(filename, "wb") as f:
                    f.write(response.content)
                loader = PyPDFLoader(filename)

            elif filename.endswith((".xlsx", ".xls")) or "application/vnd.ms-excel" in content_type or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type:
                with open(filename, "wb") as f:
                    f.write(response.content)
                loader = UnstructuredExcelLoader(filename)
                
            else:
                # Default case for other binary files
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"File saved as {filename} - cannot process this file type")
                loader = None
                
        else:
            print("Error:", response.status_code, response.text)
            loader = None

        if loader:
            try:
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                documents = text_splitter.split_documents(docs)
                
                embeddings = OpenAIEmbeddings()
                upsert_documents(documents)
                
            except Exception as e:
                print(f"Error processing documents: {e}")
                
    except Exception as e:
        print(f"Main execution error: {e}")