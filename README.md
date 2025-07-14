# 🔍 RAG Chat System

A production-ready **Retrieval-Augmented Generation (RAG)** system that enables intelligent document querying using AI-powered search and generation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange)
![Gradio](https://img.shields.io/badge/Gradio-Web%20Interface-yellow)

## ✨ Features

- 🤖 **AI-Powered Q&A**: Ask natural language questions about your documents
- 📄 **Multi-Format Support**: Process PDF, TXT, and JSON files
- 🔍 **Semantic Search**: MongoDB Atlas Vector Search for intelligent document retrieval
- 🎨 **Modern Web Interface**: Clean, responsive Gradio-based UI
- 📊 **Real-Time Statistics**: Monitor your document collection
- 🔒 **Secure**: Cloud-based storage with secure API access
- 🚀 **Easy Deployment**: One-command setup and launch

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │────│   RAG Engine     │────│  MongoDB Atlas  │
│                 │    │                  │    │  Vector Search  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                    ┌──────────────────┐
                    │   OpenAI API     │
                    │ (GPT-4 + Embeds) │
                    └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **MongoDB Atlas Cluster** with Vector Search enabled ([Setup guide](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-chat-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   
   Create a `.env` file with your credentials:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
   DB_NAME=your_database_name
   COLLECTION_NAME=your_collection_name
   OPENAI_MODEL=gpt-4
   EMBEDDING_MODEL=text-embedding-ada-002
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

4. **Launch the application**
   ```bash
   python run.py
   ```
   
   Or directly:
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:7860` and will provide a shareable public link.

## 📁 Project Structure

```
rag-chat-system/
├── 📄 app.py                    # Main Gradio web application
├── 🔧 config.py                 # Configuration management
├── 📝 document_processor.py     # Document processing utilities
├── 🚀 run.py                    # Startup script with checks
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                 # This file
├── 🗂️ sample_files/             # Sample documents for testing
├── 📄 load_sample_data.py       # Load sample data script
├── 🔒 .env                      # Environment variables (create this)
└── 📦 .venv/                    # Virtual environment (optional)
```

## 🖥️ Usage

### 1. **Upload Documents**
- Go to the "📁 Documents" tab
- Upload PDF, TXT, or JSON files
- Documents are automatically processed and indexed

### 2. **Ask Questions**
- Switch to the "💬 Chat" tab
- Type your question in natural language
- Get AI-generated answers with source citations

### 3. **Monitor System**
- View real-time statistics in the sidebar
- Track document count and system status

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `MONGODB_URI` | MongoDB connection string | Required |
| `DB_NAME` | Database name | `rag_chat_db` |
| `COLLECTION_NAME` | Collection name | `documents` |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-ada-002` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |

### MongoDB Atlas Setup

1. Create a MongoDB Atlas cluster
2. Create a database and collection
3. Set up a Vector Search index with these settings:
   ```json
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 1536,
         "similarity": "cosine"
       }
     ]
   }
   ```

## 📊 Sample Data

Load sample documents to test the system:

```bash
python load_sample_data.py
```

This will index sample files from the `sample_files/` directory.

## 🛠️ Development

### Running in Development Mode

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with debug logging
PYTHONPATH=. python app.py
```

### Adding New Document Types

Extend the `DocumentProcessor` class in `document_processor.py`:

```python
def process_new_format(self, file_path: str) -> List[Document]:
    # Your processing logic here
    pass
```

## 🔍 How It Works

1. **Document Processing**: Uploaded documents are split into chunks using LangChain text splitters
2. **Embedding Generation**: Each chunk is converted to vector embeddings using OpenAI's embedding model
3. **Vector Storage**: Embeddings are stored in MongoDB Atlas with metadata
4. **Query Processing**: User questions are embedded and used for similarity search
5. **Context Retrieval**: Most relevant document chunks are retrieved
6. **Answer Generation**: GPT-4 generates answers based on retrieved context

## 🔧 Troubleshooting

### Common Issues

**"Proxies error"**: This has been fixed with automatic compatibility patches.

**MongoDB connection issues**: 
- Verify your connection string
- Ensure your IP is whitelisted in Atlas
- Check that Vector Search is enabled

**OpenAI API errors**:
- Verify your API key
- Check your usage limits and billing

**Import errors**:
```bash
pip install --upgrade -r requirements.txt
```

### Logs

The application provides detailed logging. Check the console output for debugging information.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the MongoDB Atlas and OpenAI documentation
3. Open an issue on GitHub

---

**Built with ❤️ using LangChain, MongoDB Atlas, OpenAI, and Gradio**
