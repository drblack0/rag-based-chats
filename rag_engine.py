"""RAG Engine for MongoDB Atlas Vector Search."""

from typing import List, Optional, Tuple
from pymongo import MongoClient

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

from config import Config
from document_processor import DocumentProcessor


class RAGEngine:
    """Main RAG engine using MongoDB Atlas Vector Search."""

    def __init__(self):
        """Initialize the RAG engine."""
        # Validate configuration
        Config.validate()

        # Initialize MongoDB client
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DB_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]

        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings()

        # Fix Pydantic validation issue
        ChatOpenAI.model_rebuild()

        self.llm = ChatOpenAI(
            model_name=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE
        )

        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="vector_index",
        )

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

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        try:
            if not documents:
                print("No documents to add")
                return False

            # Add documents to MongoDB Atlas Vector Search
            self.vector_store.add_documents(documents)
            print(f"Successfully added {len(documents)} documents to the vector store")
            return True

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
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
            print(f"Error loading and indexing file {file_path}: {e}")
            return False

    def load_and_index_directory(self, directory_path: str) -> bool:
        """Load and index all files in a directory."""
        try:
            documents = self.document_processor.load_directory(directory_path)
            if documents:
                return self.add_documents(documents)
            return False
        except Exception as e:
            print(f"Error loading and indexing directory {directory_path}: {e}")
            return False

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search and return raw documents."""
        try:
            k = k or Config.SIMILARITY_SEARCH_K
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
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
            print(f"Error during query: {e}")
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
            print(f"Error during simple query: {e}")
            return f"Error processing query: {e}"

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        print("Conversation memory cleared")

    def get_collection_stats(self) -> dict:
        """Get statistics about the document collection."""
        try:
            stats = {
                "total_documents": self.collection.count_documents({}),
                "database_name": Config.DB_NAME,
                "collection_name": Config.COLLECTION_NAME,
            }
            return stats
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}

    def delete_all_documents(self) -> bool:
        """Delete all documents from the collection (use with caution)."""
        try:
            result = self.collection.delete_many({})
            print(f"Deleted {result.deleted_count} documents")
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
