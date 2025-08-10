import requests
import mimetypes
from langchain_community.document_loaders import TextLoader,PyPDFLoader, UnstructuredExcelLoader
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

url = "http://localhost:3000/data" # express api endpoint
response = requests.get(url)

def getFileName(response):
    content_disposition = response.headers.get("Content-Disposition","")
    filename=None
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[-1].strip('"')
    return filename

def fetch_text_from_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "").lower()
        filename = getFileName(requests.get(url))
        if content_type.startswith("text/") and filename and not filename.endswith(".txt"):
            return response.text
    return None

def upsert_documents(documents):
    """
    Creates a FAISS DB if it doesn't exist,
    else loads and updates the existing DB.
    """
    if os.path.exists(DB_PATH):
        # Load existing DB
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing FAISS DB.")
        
        # Add new docs
        db.add_documents(documents)
        print(f"Added {len(documents)} new documents.")
    else:
        # Create new DB
        db = FAISS.from_documents(documents, embeddings)
        print("Created new FAISS DB.")

    # Save changes
    db.save_local(DB_PATH)
    print("Saved FAISS DB.")

if response.status_code == 200:
    content_type = response.headers.get("Content-Type","").lower()
    filename = getFileName(requests.get(url))
    
    # if no file name guess based on content type
    if not filename:
        extension = mimetypes.guess_extension(content_type.split(";")[0]) or ""
        filename = "downloaded_file"+(extension if extension else "")
    
    #Detect and handle content
    if "application/json" in content_type:
        json_obj = response.json()
        print("Json Data:",json_obj)
    elif content_type.startswith("text/") and not filename.endswith(".txt"):
        #plain text from api
        print("Text Data: ",response.text)
    elif filename.endswith(".txt") or "text/plain" in content_type:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)
        loader = TextLoader(filename)
        docs = loader.load()
        print("Loaded Text Documents:", docs)
        
    elif filename.endswith(".pdf") or "application/pdf" in content_type:
        with open(filename, "wb") as f:
            f.write(response.content)
        loader = PyPDFLoader(filename)

    elif filename.endswith(".xlsx") or filename.endswith(".xls") or "application/vnd.ms-excel" in content_type or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type:
        with open(filename, "wb") as f:
            f.write(response.content)
        loader = UnstructuredExcelLoader(filename)
    else:
        # Default case for other binary files
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"File saved as {filename} cannot process this file")
        loader = None
else:
    print("Error: ",response.status_code,response.text)
    loader = None

if loader:
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(docs)

    DB_PATH = "faiss_index"
    embeddings = OpenAIEmbeddings()
    upsert_documents(documents)