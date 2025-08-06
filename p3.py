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

# 0. Reset ChromaDB and summary index
for path in ["./chroma_db", "./summary_index"]:
    if os.path.exists(path):
        shutil.rmtree(path)

# 1. Load documents
documents = SimpleDirectoryReader(input_dir="./roadmaps").load_data()

# 2. Setup models
splitter = SentenceSplitter(chunk_size=1024)
llm = Ollama(model="tinyllama", temperature=0, request_timeout=300.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 3. Generate document summary index
synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    use_async=True,
    llm=llm
)
summary_index = DocumentSummaryIndex.from_documents(
    documents,
    llm=llm,
    transformations=[splitter],
    response_synthesizer=synthesizer,
    embed_model=embed_model,
    show_progress=True
)

summary_index.storage_context.persist("summary_index")

filename_to_id = {
    doc.metadata.get("file_name", "Unknown"): doc.doc_id
    for doc in documents
}

# Prepare filenames list for dropdown
filenames = list(filename_to_id.keys())

# 5. Set up vector store
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("roadmap_summaries")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 6. Split into nodes and build vector index
nodes = splitter.get_nodes_from_documents(documents)
vector_index = VectorStoreIndex.from_documents(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)

# 7. Set up reranker
rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=3
)

# 8. Create query engines
query_engine_docs = vector_index.as_query_engine(
    llm=llm,
    node_postprocessors=[rerank]
)
query_engine_summary = summary_index.as_query_engine(llm=llm)

# 9. Function to get the sumamry of a chosen file
def get_summary_for_file(selected_filename):
    doc_id = filename_to_id.get(selected_filename)
    if doc_id:
        summary_response = summary_index.get_document_summary(doc_id)
        return str(summary_response)
    else:
        return "File not found."

# 10. Gradio UI logic for chatbot
def user_query(message, history, source):
    if source == "Full Document":
        response = query_engine_docs.query(message)
    else:
        print("not possible")
    return str(response)


with gr.Blocks() as demo:
    gr.Markdown("## 📘 Roadmap Document Summaries")
    filename_dropdown = gr.Dropdown(filenames, label="Choose a document", value=filenames[0])
    summary_output = gr.Textbox(lines=15, label="Document Summary", interactive=False)
    filename_dropdown.change(fn=get_summary_for_file, inputs=filename_dropdown, outputs=summary_output)
    chatbot = gr.ChatInterface(
        fn=lambda message, history: user_query(message, history, source="Full Document"),
        title="Roadmap Chatbot",
        description="Ask questions about the roadmap files you've uploaded.",
    )
    summary_output.value = get_summary_for_file(filenames[0])

demo.launch(share=False)
