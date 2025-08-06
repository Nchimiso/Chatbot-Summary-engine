- Documentation set up
    1. Ensure **Ollama is installed** and running on your device with the following models downloaded:
    2. **Download Python 3.10+** from the official [Python website](https://www.python.org/downloads/)
        1. !!!Alternatively, use **VS Code** with the **Python extension** to make development easier.
    3. **Create a new Python script**, name it something like `roadmap_rag.py` in your working directory.
    4. **Install all required libraries** by running this in your terminal or command prompt:
        
        ```bash
        bash
        pip install llama-index gradio chromadb sentence-transformers
        
        ```
        
    5. **Your import section** should look like this:
        
        ```python
        python
        CopyEdit
        import os
        import shutil
        import chromadb
        import gradio as gr
        
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core import DocumentSummaryIndex
        from llama_index.core import VectorStoreIndex
        from llama_index.core import StorageContext
        from llama_index.core import get_response_synthesizer
        
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.postprocessor import SentenceTransformerRerank
        
        ```
        
    6. **Ensure you have a `roadmaps/` folder in the same directory** as your Python file.
        1. Add your text files (e.g., resumes, plans, etc.) inside this folder. Each will be parsed and embedded. 
            1. !!!For better performance using fewer document will yield much better results
    
    ---
    
    ### Script Execution Process
    
    1. Reset ChromaDB and summary index
    2. Load documents
    3. Setup models
    4. Generate document summary index
    5. Get a combined summary text
    6. Set up vector store
    7. Split into nodes and build a vector index
    8. Set up reranker
    9. Create query engines
    10. Gradio UI for Chatbot
- Usage
    1. In the command line, run the following command to run the Python file and ensure that the necessary folder is set up, including the desired documents. 
        
        ```bash
        bash
        python yourfilename.py (line to run)
        ```
        
    2. You will now see a link in the command line. Click this link and use the gradio UI to view the summary and query the document.
        1. !!! If you wish to use public ensure python script sets share to true for gradio.
