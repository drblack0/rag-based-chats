"use client";

import { useState } from "react";
import React from "react"; // Added missing import for React.useEffect

const TABS = ["Chat", "Documents", "Stats", "About"];
let API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
API_BASE = API_BASE.replace(/\/+$/, "");
API_BASE = API_BASE.replace(/\/api$/, "");

export default function Home() {
  const [tab, setTab] = useState("Chat");

  return (
    <div className="font-sans min-h-screen bg-gray-50 text-gray-900 flex flex-col items-center p-4 sm:p-8">
      <header className="w-full max-w-3xl text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">🔍 RAG Chat System</h1>
        <p className="text-lg text-gray-600">Ask questions about your documents using AI-powered search and generation</p>
      </header>
      <nav className="flex gap-4 mb-8">
        {TABS.map((t) => (
          <button
            key={t}
            className={`px-4 py-2 rounded-t font-medium border-b-2 transition-colors ${tab === t ? "border-blue-500 text-blue-600 bg-white" : "border-transparent text-gray-500 bg-gray-100 hover:bg-white"}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </nav>
      <main className="w-full max-w-3xl bg-white rounded shadow p-6 min-h-[400px]">
        {tab === "Chat" && <ChatTab />}
        {tab === "Documents" && <DocumentsTab />}
        {tab === "Stats" && <StatsTab />}
        {tab === "About" && <AboutTab />}
      </main>
    </div>
  );
}

function ChatTab() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setIsLoading(true);
    try {
      const askUrl = `${API_BASE}/api/ask`;
      const response = await fetch(askUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error("Failed to get answer");
      }

      const data = await response.json();
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (error) {
      console.error("Error:", error);
      setAnswer("❌ Error: Could not get answer. Please check if the backend is running.");
      setSources("");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setQuestion("");
    setAnswer("");
    setSources("");
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">💬 Chat</h2>
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="mb-4">
          <label htmlFor="question" className="block text-sm font-medium mb-2">
            ❓ Ask a Question
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question about the documents..."
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={3}
            disabled={isLoading}
          />
        </div>
        <div className="flex gap-2">
          <button
            type="submit"
            disabled={isLoading || !question.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? "🔍 Searching..." : "🔍 Search & Answer"}
          </button>
          <button
            type="button"
            onClick={handleClear}
            className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600"
          >
            🗑️ Clear
          </button>
        </div>
      </form>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h3 className="text-lg font-medium mb-2">🤖 AI Answer</h3>
          <div className="p-4 bg-gray-50 rounded-md min-h-[200px] max-h-[400px] overflow-y-auto">
            {answer ? (
              <div className="prose prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: answer.replace(/\n/g, '<br/>') }} />
            ) : (
              <p className="text-gray-500">Ask a question to get an answer...</p>
            )}
          </div>
        </div>
        <div>
          <h3 className="text-lg font-medium mb-2">📚 Sources</h3>
          <div className="p-4 bg-gray-50 rounded-md min-h-[200px] max-h-[400px] overflow-y-auto">
            {sources ? (
              <div className="prose prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: sources.replace(/\n/g, '<br/>') }} />
            ) : (
              <p className="text-gray-500">Sources will appear here...</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function DocumentsTab() {
  const [uploadStatus, setUploadStatus] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [health, setHealth] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    if (!file) return;

    setIsUploading(true);
    setUploadStatus("📤 Uploading...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const uploadUrl = `${API_BASE}/api/upload`;
      const response = await fetch(uploadUrl, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const raw = await response.text();
        let message = response.statusText;
        try {
          const parsed = JSON.parse(raw);
          message = parsed.detail || parsed.status || raw || response.statusText;
        } catch {
          message = raw || response.statusText;
        }
        setUploadStatus(`❌ Upload failed (${response.status}) at ${uploadUrl}: ${message}`);
        return;
      }

      const data = await response.json();
      setUploadStatus(`✅ ${data.status}`);
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus("❌ Upload failed. Please check if the backend is running.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const checkHealth = async () => {
    setHealth("Checking...");
    try {
      const url = `${API_BASE}/api/stats`;
      const res = await fetch(url);
      const text = await res.text();
      setHealth(`${res.status} at ${url}: ${text.slice(0, 120)}${text.length > 120 ? "..." : ""}`);
    } catch (e: any) {
      setHealth(`Error: ${e?.message || String(e)}`);
    }
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">📁 Documents</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h3 className="text-lg font-medium mb-4">📤 Upload Document</h3>
          
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="space-y-4">
              <div className="text-4xl">📄</div>
              <div>
                <p className="text-lg font-medium mb-2">
                  {dragActive ? "Drop your file here" : "Drag and drop your file here"}
                </p>
                <p className="text-gray-500 mb-4">or</p>
                <label className="cursor-pointer">
                  <input
                    type="file"
                    accept=".pdf,.txt,.json"
                    onChange={handleFileInput}
                    className="hidden"
                    disabled={isUploading}
                  />
                  <span className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50">
                    Choose File
                  </span>
                </label>
              </div>
              <p className="text-sm text-gray-500">
                Supported formats: PDF, TXT, JSON
              </p>
            </div>
          </div>

          {uploadStatus && (
            <div className="mt-4 p-3 rounded-md bg-gray-50">
              <p className="text-sm">{uploadStatus}</p>
            </div>
          )}

          <div className="mt-4 flex items-center gap-2">
            <button
              type="button"
              onClick={checkHealth}
              className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              🔎 Check API Health
            </button>
            {health && <span className="text-xs text-gray-600">{health}</span>}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium mb-4">📋 Supported File Types</h3>
          <div className="space-y-3 text-sm">
            <div className="p-3 bg-gray-50 rounded-md">
              <h4 className="font-medium">📄 PDF</h4>
              <p className="text-gray-600">Text extraction from PDF documents</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-md">
              <h4 className="font-medium">📝 TXT</h4>
              <p className="text-gray-600">Plain text files</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-md">
              <h4 className="font-medium">📊 JSON</h4>
              <p className="text-gray-600">Structured data files</p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-50 rounded-md">
            <h4 className="font-medium text-blue-800 mb-2">💡 Tips</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Upload high-quality documents for best results</li>
              <li>• Text-heavy documents work better than image-heavy ones</li>
              <li>• JSON files should have text content in readable fields</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatsTab() {
  const [stats, setStats] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchStats = async () => {
    setIsLoading(true);
    setError("");
    
    try {
      const statsUrl = `${API_BASE}/api/stats`;
      const response = await fetch(statsUrl);
      
      if (!response.ok) {
        throw new Error("Failed to fetch stats");
      }
      
      const data = await response.json();
      setStats(data.stats);
    } catch (error) {
      console.error("Stats error:", error);
      setError("❌ Could not fetch system stats. Please check if the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch stats on component mount
  React.useEffect(() => {
    fetchStats();
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">📊 System Stats</h2>
        <button
          onClick={fetchStats}
          disabled={isLoading}
          className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
        >
          {isLoading ? "🔄 Refreshing..." : "🔄 Refresh"}
        </button>
      </div>
      
      <div className="space-y-4">
        {isLoading && (
          <div className="p-4 bg-blue-50 rounded-md">
            <p className="text-blue-700">🔄 Loading system statistics...</p>
          </div>
        )}
        
        {error && (
          <div className="p-4 bg-red-50 rounded-md">
            <p className="text-red-700">{error}</p>
          </div>
        )}
        
        {stats && !isLoading && (
          <div className="p-4 bg-gray-50 rounded-md">
            <div className="prose prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: stats.replace(/\n/g, '<br/>') }} />
          </div>
        )}
        
        {!stats && !isLoading && !error && (
          <div className="p-4 bg-gray-50 rounded-md">
            <p className="text-gray-500">No system statistics available.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function AboutTab() {
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">ℹ️ About</h2>
      <p className="mb-2">This is a <b>Retrieval-Augmented Generation (RAG)</b> system that allows you to:</p>
      <ul className="list-disc pl-6 mb-2 text-gray-700">
        <li>Upload your own documents</li>
        <li>Ask questions about those documents</li>
        <li>Get AI-powered answers based on the content</li>
      </ul>
      <p className="mb-2">Powered by FastAPI, OpenAI GPT-4, FAISS Vector Search, and Next.js.</p>
    </div>
  );
}
