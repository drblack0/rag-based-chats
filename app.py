"""
🔍 RAG Chat Application
======================
A production-ready RAG (Retrieval-Augmented Generation) chat application
using MongoDB Atlas Vector Search and OpenAI GPT-4.

Features:
- Document upload and processing (PDF, TXT, JSON)
- Vector similarity search with MongoDB Atlas
- AI-powered question answering
- Clean, modern web interface
- Real-time document management
"""

import os
import logging
from typing import Tuple, Optional


def apply_openai_fix():
    """Apply compatibility fix for OpenAI/LangChain integration."""
    try:
        import openai._base_client as base_client

        # Patch HTTP client wrappers to remove incompatible 'proxies' parameter
        for wrapper_name in ["SyncHttpxClientWrapper", "AsyncHttpxClientWrapper"]:
            if hasattr(base_client, wrapper_name):
                wrapper_class = getattr(base_client, wrapper_name)
                original_init = wrapper_class.__init__

                def patched_init(self, **kwargs):
                    kwargs.pop("proxies", None)
                    return original_init(self, **kwargs)

                wrapper_class.__init__ = patched_init

        return True
    except Exception as e:
        logging.warning(f"Could not apply OpenAI compatibility fix: {e}")
        return False


# Apply compatibility fix before importing LangChain
apply_openai_fix()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import dependencies
import gradio as gr
import openai
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.documents import Document

# Import local modules
from config import Config
from document_processor import DocumentProcessor


class RAGChatSystem:
    """Main RAG Chat System class."""

    def __init__(self):
        """Initialize the RAG chat system."""
        logger.info("🚀 Initializing RAG Chat System...")

        try:
            # Validate configuration
            Config.validate()
            logger.info("✅ Configuration loaded")

            # Initialize MongoDB connection
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.DB_NAME]
            self.collection = self.db[Config.COLLECTION_NAME]
            logger.info("✅ MongoDB connected")

            # Initialize OpenAI components
            self.embeddings = OpenAIEmbeddings()
            self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info("✅ OpenAI initialized")

            # Initialize vector store
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name="vector_index",
            )
            logger.info("✅ Vector store initialized")

            # Initialize document processor
            self.document_processor = DocumentProcessor()
            logger.info("✅ Document processor ready")

            logger.info("🎉 RAG Chat System fully operational!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            raise

    def query_documents(self, question: str) -> Tuple[str, str]:
        """Query documents and generate AI response."""
        if not question.strip():
            return "Please enter a question.", ""

        try:
            # Perform vector similarity search
            docs = self.vector_store.similarity_search(question, k=3)

            if not docs:
                return "No relevant documents found for your question.", ""

            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content[:500] for doc in docs])

            # Generate AI response
            prompt = f"""Based on the following context, provide a comprehensive answer to the question:

Context:
{context}

Question: {question}

Please provide a detailed, accurate answer based only on the information provided in the context."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Always base your answers on the given information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )

            answer = response.choices[0].message.content

            # Format source documents
            sources = "**📚 Source Documents:**\n\n"
            for i, doc in enumerate(docs, 1):
                content_preview = (
                    doc.page_content[:150] + "..."
                    if len(doc.page_content) > 150
                    else doc.page_content
                )
                source = doc.metadata.get("source", "Unknown")
                sources += f"**Source {i}** ({source}):\n{content_preview}\n\n"

            return answer, sources

        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Error processing your question: {str(e)}", ""

    def upload_file(self, file) -> str:
        """Upload and process a new document."""
        if file is None:
            return "❌ No file uploaded."

        try:
            # Process the uploaded file
            documents = self.document_processor.process_file(file.name)

            if not documents:
                return f"❌ No content extracted from {file.name}"

            # Add documents to vector store
            self.vector_store.add_documents(documents)

            return f"✅ Successfully processed and indexed: **{file.name}** ({len(documents)} documents)"

        except Exception as e:
            logger.error(f"File upload error: {e}")
            return f"❌ Error processing file: {str(e)}"

    def get_system_stats(self) -> str:
        """Get system statistics."""
        try:
            total_docs = self.collection.count_documents({})
            return f"""📊 **System Statistics**
            
🗃️ **Database:** {Config.DB_NAME}
📁 **Collection:** {Config.COLLECTION_NAME}
📄 **Total Documents:** {total_docs}
🔍 **Vector Index:** vector_index
🤖 **AI Model:** GPT-4"""

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return f"❌ Unable to retrieve statistics: {str(e)}"


# Initialize the RAG system
try:
    rag_system = RAGChatSystem()
except Exception as e:
    logger.error(f"Failed to initialize system: {e}")
    rag_system = None


def create_interface():
    """Create the Gradio interface."""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        margin-bottom: 30px;
    }
    .tab-nav {
        margin-bottom: 20px;
    }
    """

    with gr.Blocks(
        css=custom_css, title="RAG Chat System", theme=gr.themes.Soft()
    ) as interface:
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🔍 RAG Chat System</h1>
            <p>Ask questions about your documents using AI-powered search and generation</p>
        </div>
        """)

        with gr.Tabs():
            # Main Chat Tab
            with gr.Tab("💬 Chat", elem_id="chat-tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="❓ Ask a Question",
                            placeholder="Enter your question about the documents...",
                            lines=2,
                        )
                        submit_btn = gr.Button(
                            "🔍 Search & Answer", variant="primary", size="lg"
                        )
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                    with gr.Column(scale=1):
                        stats_display = gr.Markdown(
                            value=rag_system.get_system_stats()
                            if rag_system
                            else "❌ System not initialized",
                            label="📊 System Info",
                        )

                with gr.Row():
                    with gr.Column():
                        answer_output = gr.Markdown(label="🤖 AI Answer", height=300)
                    with gr.Column():
                        sources_output = gr.Markdown(label="📚 Sources", height=300)

            # Document Management Tab
            with gr.Tab("📁 Documents", elem_id="docs-tab"):
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="📤 Upload Document",
                            file_types=[".pdf", ".txt", ".json"],
                            type="filepath",
                        )
                        upload_btn = gr.Button("📄 Process Document", variant="primary")
                        upload_status = gr.Markdown(label="📋 Upload Status")

                    with gr.Column():
                        gr.Markdown("""
                        ### 📋 Supported File Types
                        - **PDF**: Text extraction from PDF documents
                        - **TXT**: Plain text files
                        - **JSON**: Structured data files
                        
                        ### 💡 Tips
                        - Upload high-quality documents for best results
                        - Text-heavy documents work better than image-heavy ones
                        - JSON files should have text content in readable fields
                        """)

            # About Tab
            with gr.Tab("ℹ️ About", elem_id="about-tab"):
                gr.Markdown("""
                # 🔍 RAG Chat System
                
                ## What is this?
                This is a **Retrieval-Augmented Generation (RAG)** system that allows you to:
                - Upload your own documents
                - Ask questions about those documents
                - Get AI-powered answers based on the content
                
                ## How it works
                1. **Document Processing**: Your documents are processed and split into chunks
                2. **Vector Embedding**: Text chunks are converted to vector embeddings using OpenAI
                3. **Storage**: Embeddings are stored in MongoDB Atlas with vector search capabilities
                4. **Retrieval**: When you ask a question, similar document chunks are found
                5. **Generation**: GPT-4 generates an answer based on the retrieved context
                
                ## Technology Stack
                - **Frontend**: Gradio web interface
                - **Backend**: Python with LangChain
                - **Vector Database**: MongoDB Atlas Vector Search
                - **AI Model**: OpenAI GPT-4
                - **Embeddings**: OpenAI text-embedding-ada-002
                
                ## Features
                - 🔍 Semantic search across documents
                - 🤖 AI-powered question answering
                - 📄 Multi-format document support
                - 🔒 Secure cloud-based storage
                - 📊 Real-time system statistics
                """)

        # Event handlers
        if rag_system:
            submit_btn.click(
                fn=rag_system.query_documents,
                inputs=[question_input],
                outputs=[answer_output, sources_output],
            )

            upload_btn.click(
                fn=rag_system.upload_file, inputs=[file_upload], outputs=[upload_status]
            )

            # Auto-refresh stats when uploading
            upload_btn.click(fn=rag_system.get_system_stats, outputs=[stats_display])

        # Clear functionality
        clear_btn.click(
            fn=lambda: ("", "", ""),
            outputs=[question_input, answer_output, sources_output],
        )

        # Enter key support
        question_input.submit(
            fn=rag_system.query_documents
            if rag_system
            else lambda x: ("System not initialized", ""),
            inputs=[question_input],
            outputs=[answer_output, sources_output],
        )

    return interface


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()

    # Launch with appropriate settings
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_api=False,  # Disable API docs to avoid JSON schema issues
        quiet=True,  # Reduce verbose output
    )
