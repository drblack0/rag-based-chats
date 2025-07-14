"""Document processing utilities for RAG Chat System."""

import json
import os
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)

from config import Config


class DocumentProcessor:
    """Handle document loading and processing."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_text_file(self, file_path: str) -> List[Document]:
        """Load a text file."""
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return []

    def load_pdf_file(self, file_path: str) -> List[Document]:
        """Load a PDF file."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error loading PDF file {file_path}: {e}")
            return []

    def load_json_file(
        self, file_path: str, content_key: Optional[str] = None
    ) -> List[Document]:
        """Load a JSON file and convert to documents."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            documents = []

            if content_key and content_key in data:
                # If specific key is provided, use that data
                data = data[content_key]

            if isinstance(data, list):
                # Handle list of items
                for i, item in enumerate(data):
                    content = str(item) if not isinstance(item, str) else item
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path, "chunk_id": i},
                    )
                    documents.append(doc)
            elif isinstance(data, dict):
                # Handle dictionary
                content = json.dumps(data, indent=2)
                doc = Document(page_content=content, metadata={"source": file_path})
                documents.append(doc)
            else:
                # Handle other types
                content = str(data)
                doc = Document(page_content=content, metadata={"source": file_path})
                documents.append(doc)

            return self.text_splitter.split_documents(documents)

        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return []

    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported files from a directory."""
        documents = []
        directory = Path(directory_path)

        if not directory.exists():
            print(f"Directory {directory_path} does not exist")
            return documents

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()

                if suffix == ".txt":
                    documents.extend(self.load_text_file(str(file_path)))
                elif suffix == ".pdf":
                    documents.extend(self.load_pdf_file(str(file_path)))
                elif suffix == ".json":
                    documents.extend(self.load_json_file(str(file_path)))
                else:
                    print(f"Unsupported file type: {file_path}")

        return documents

    def load_file(
        self, file_path: str, content_key: Optional[str] = None
    ) -> List[Document]:
        """Load a single file based on its extension."""
        path = Path(file_path)

        if not path.exists():
            print(f"File {file_path} does not exist")
            return []

        suffix = path.suffix.lower()

        if suffix == ".txt":
            return self.load_text_file(file_path)
        elif suffix == ".pdf":
            return self.load_pdf_file(file_path)
        elif suffix == ".json":
            return self.load_json_file(file_path, content_key)
        else:
            print(f"Unsupported file type: {suffix}")
            return []
