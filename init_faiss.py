
# init_faiss.py
import os
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document

# Create the necessary directories
os.makedirs("faiss_index", exist_ok=True)

# Initialize with a dummy document
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dummy_doc = Document(page_content="Initial document for FAISS initialization")
db = FAISS.from_documents([dummy_doc], embeddings)
db.save_local("faiss_index")

print("✅ Manually created FAISS database structure")