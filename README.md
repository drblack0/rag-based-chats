# RAG-Based Chat Application

A Retrieval-Augmented Generation (RAG) chat application built with FastAPI backend and Next.js frontend. This application allows users to upload documents and ask questions that are answered using AI-powered document search and generation.

## Features

- **Document Upload**: Support for PDF, TXT, and JSON files
- **AI-Powered Q&A**: Ask questions and get answers based on uploaded documents
- **Vector Search**: Uses FAISS for efficient similarity search
- **Modern UI**: Clean, responsive Next.js frontend
- **Real-time Chat**: Interactive chat interface with streaming responses

## Project Structure

```
rag-based-chats/
├── frontend/                 # Next.js frontend application
├── api.py                   # FastAPI main application
├── config.py                # Configuration management
├── rag_engine_faiss.py      # FAISS-based RAG engine
├── document_processor.py     # Document processing utilities
├── requirements.txt          # Python dependencies
├── run.py                   # Backend entry point
└── sample_files/            # Sample documents for testing
```

## Prerequisites

- Python 3.8+
- Node.js 18+
- OpenAI API key

## Backend Setup

### 1. Create Virtual Environment

```bash
cd rag-based-chats
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the `rag-based-chats` directory:

```bash
cp env_template.txt .env
```

Edit `.env` with your configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URI=sqlite:///./chat_history.db
FAISS_INDEX_PATH=./faiss_index/index.faiss
FAISS_METADATA_PATH=./faiss_index/metadata.pkl
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### 4. Run Backend

```bash
python run.py
```

The backend will start on `http://localhost:8000`

## Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd frontend
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Environment Configuration

Create a `.env.local` file in the `frontend` directory:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

**Important**: Do not include trailing slashes or `/api` in the URL.

### 4. Run Frontend

```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

## Usage

1. **Start both backend and frontend** in separate terminal windows
2. **Upload documents** using the Documents tab
3. **Ask questions** in the Chat tab
4. **View statistics** in the Stats tab

## API Endpoints

- `POST /api/ask` - Ask a question
- `POST /api/upload` - Upload a document
- `GET /api/stats` - Get system statistics
- `GET /api/health` - Health check

## Troubleshooting

### Common Issues

1. **"Upload failed" errors**: Check that the backend is running and the API URL is correct
2. **CORS errors**: Ensure `ALLOWED_ORIGINS` in backend `.env` includes your frontend URL
3. **FAISS index errors**: The index will be created automatically on first document upload

### Debug Steps

1. Check backend logs for Python errors
2. Verify API health using the "Check API Health" button in the Documents tab
3. Ensure environment variables are set correctly
4. Check that both services are running on the expected ports

## Development

### Backend Development

- The main FastAPI app is in `api.py`
- RAG engine logic is in `rag_engine_faiss.py`
- Configuration is managed in `config.py`

### Frontend Development

- Built with Next.js 14 and TypeScript
- Main component is in `frontend/src/app/page.tsx`
- Uses modern React patterns with hooks

## Dependencies

### Backend
- FastAPI - Web framework
- FAISS - Vector similarity search
- LangChain - LLM framework
- OpenAI - AI models and embeddings

### Frontend
- Next.js 14 - React framework
- TypeScript - Type safety
- Tailwind CSS - Styling

## License

This project is for educational and development purposes.
