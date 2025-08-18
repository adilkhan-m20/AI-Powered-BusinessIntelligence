
# ai_service/initialize.py
import os
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Test critical imports first
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    import transformers
    print(f"✅ Transformers version: {transformers.__version__}")
    
    import sentence_transformers
    print(f"✅ Sentence Transformers version: {sentence_transformers.__version__}")
    
    # Now try to import our document engine
    from ai_service.document_engine import process_document_from_path
    print("✅ Successfully imported document_engine")
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Create a dummy document
    dummy_content = """This is a sample document to initialize the FAISS database.
    The system uses this to create the initial vector store structure.
    This document will be used to test the document processing pipeline."""
    
    dummy_file = uploads_dir / "dummy_document.txt"
    with open(dummy_file, 'w', encoding='utf-8') as f:
        f.write(dummy_content)
    
    print(f"✅ Created dummy document at {dummy_file}")
    
    # Process the document
    result = process_document_from_path(str(dummy_file))
    
    if result and result.get("success"):
        print("✅ Successfully initialized FAISS database!")
        print(f"Created {result['chunks']} chunks and {result['embeddings']} embeddings")
    else:
        print("❌ Failed to initialize FAISS database")
        if result:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Try reinstalling the packages with:")
    print("pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu")
    print("pip install transformers==4.37.2 sentence-transformers==2.2.2")
except Exception as e:
    print(f"❌ Error during initialization: {e}")