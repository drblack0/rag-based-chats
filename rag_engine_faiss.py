"""RAG Engine for FAISS Vector Search."""

import os
import pickle
from typing import List, Optional, Tuple
import numpy as np
import httpx
import faiss

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks

from config import Config
from document_processor import DocumentProcessor


class RAGEngineFAISS:
    """Main RAG engine using FAISS for vector storage."""

    def __init__(self):
        """Initialize the RAG engine."""
        # Validate configuration
        Config.validate()

        # Shared HTTP client to avoid proxies kwarg path
        self.http_client = httpx.Client()

        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            http_client=self.http_client,
        )

        # Fix Pydantic validation issue for ChatOpenAI by providing missing types
        ChatOpenAI.model_rebuild(_types_namespace={"BaseCache": BaseCache, "Callbacks": Callbacks})

        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            http_client=self.http_client,
        )

        # Initialize vector store
        self.vector_store = self._load_or_create_vector_store()

        # Initialize document processor
        self.document_processor = DocumentProcessor()

        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Initialize conversation chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": Config.SIMILARITY_SEARCH_K}
            ),
            memory=self.memory,
            verbose=True,
        )

    def _load_or_create_vector_store(self) -> FAISS:
        """Load existing FAISS index or create a new empty one without API calls."""
        faiss_path = Config.FAISS_INDEX_PATH
        metadata_path = Config.FAISS_METADATA_PATH

        # Try to load existing index (does not call embeddings API)
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            try:
                vector_store = FAISS.load_local(
                    folder_path=os.path.dirname(faiss_path),
                    embeddings=self.embeddings,
                    index_name=os.path.basename(faiss_path),
                    allow_dangerous_deserialization=True,
                )
                print(f"✅ Loaded existing FAISS index from {faiss_path}")
                return vector_store
            except Exception as e:
                print(f"⚠️ Could not load existing FAISS index: {e}")
                print("Creating new FAISS index...")

        # Create a new, empty FAISS index without embedding any text to avoid API usage
        # Dimension map for common OpenAI embedding models
        embedding_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        dimension = embedding_dims.get(getattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small"), 1536)

        index = faiss.IndexFlatL2(dimension)
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        print("✅ Created new empty FAISS index without calling embeddings API")
        return vector_store

    def _save_vector_store(self):
        """Save the FAISS index and metadata."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)
            
            # Save FAISS index
            self.vector_store.save_local(
                folder_path=os.path.dirname(Config.FAISS_INDEX_PATH),
                index_name=os.path.basename(Config.FAISS_INDEX_PATH)
            )
            
            # Save metadata separately for easier access
            with open(Config.FAISS_METADATA_PATH, "wb") as f:
                pickle.dump({"docstore": self.vector_store.docstore._dict}, f)
            
            print(f"✅ Saved FAISS index to {Config.FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"❌ Error saving FAISS index: {e}")

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        try:
            if not documents:
                print("No documents to add")
                return False

            # Add documents to FAISS
            self.vector_store.add_documents(documents)
            
            # Save the updated index
            self._save_vector_store()
            
            print(f"✅ Successfully added {len(documents)} documents to FAISS")
            return True

        except Exception as e:
            print(f"❌ Error adding documents to FAISS: {e}")
            return False

    def load_and_index_file(
        self, file_path: str, content_key: Optional[str] = None
    ) -> bool:
        """Load and index a single file."""
        try:
            documents = self.document_processor.load_file(file_path, content_key)
            if documents:
                return self.add_documents(documents)
            return False
        except Exception as e:
            print(f"❌ Error loading and indexing file {file_path}: {e}")
            return False

    def load_and_index_directory(self, directory_path: str) -> bool:
        """Load and index all files in a directory."""
        try:
            documents = self.document_processor.load_directory(directory_path)
            if documents:
                return self.add_documents(documents)
            return False
        except Exception as e:
            print(f"❌ Error loading and indexing directory {directory_path}: {e}")
            return False

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search and return raw documents."""
        try:
            k = k or Config.SIMILARITY_SEARCH_K
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"❌ Error during similarity search: {e}")
            return []

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """Query the RAG system with conversation memory."""
        try:
            # Get the response from the conversation chain
            response = self.conversation_chain({"question": question})
            answer = response.get("answer", "Sorry, I couldn't generate an answer.")

            # Also get the source documents for transparency
            source_docs = self.similarity_search(question)

            return answer, source_docs

        except Exception as e:
            print(f"❌ Error during query: {e}")
            return f"Error processing query: {e}", []

    def simple_query(self, question: str) -> str:
        """Simple query without conversation memory."""
        try:
            # Create a simple QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": Config.SIMILARITY_SEARCH_K}
                ),
            )

            response = qa_chain.invoke({"query": question})
            return response.get("result", "Sorry, I couldn't generate an answer.")

        except Exception as e:
            print(f"❌ Error during simple query: {e}")
            return f"Error processing query: {e}"

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        print("✅ Conversation memory cleared")

    def get_collection_stats(self) -> dict:
        """Get statistics about the document collection."""
        try:
            stats = {
                "total_documents": len(self.vector_store.docstore._dict),
                "index_path": Config.FAISS_INDEX_PATH,
                "vector_dimension": self.vector_store.index.d if hasattr(self.vector_store.index, 'd') else "Unknown",
                "storage_type": "FAISS (Local)"
            }
            return stats
        except Exception as e:
            print(f"❌ Error getting collection stats: {e}")
            return {}

    def delete_all_documents(self) -> bool:
        """Delete all documents from the collection (use with caution)."""
        try:
            # Create a new empty FAISS index
            empty_docs = [Document(page_content="", metadata={})]
            self.vector_store = FAISS.from_documents(
                documents=empty_docs,
                embedding=self.embeddings
            )
            # Remove the empty document
            self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])
            
            # Save the empty index
            self._save_vector_store()
            
            print("✅ All documents deleted from FAISS")
            return True
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            return False

    def get_document_count(self) -> int:
        """Get the current number of documents in the index."""
        try:
            return len(self.vector_store.docstore._dict)
        except Exception as e:
            print(f"❌ Error getting document count: {e}")
            return 0
