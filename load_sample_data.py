"""Script to load and index sample data files."""

from rag_engine import RAGEngine
import os


def load_sample_data():
    """Load all sample data files into the vector store."""
    print("Initializing RAG Engine...")
    rag_engine = RAGEngine()

    # Clear existing data (optional - comment out if you want to keep existing data)
    print("Clearing existing documents...")
    rag_engine.delete_all_documents()

    sample_files = [
        ("sample_files/knowledge_base.json", "interview"),  # JSON with specific key
        ("sample_files/aerodynamics.txt", None),  # Text file
        ("sample_files/chat_conversation.txt", None),  # Text file
        ("sample_files/log_example.txt", None),  # Text file
        ("scalexi.txt", None),  # Text file in root
    ]

    # Load PDF if it exists (it's quite large, so optional)
    pdf_file = "sample_files/How-To-Win-Friends_Libtoon.com_.pdf"
    if os.path.exists(pdf_file):
        sample_files.append((pdf_file, None))
        print(f"Found PDF file: {pdf_file}")
    else:
        print(f"PDF file not found: {pdf_file}")

    total_loaded = 0

    for file_path, content_key in sample_files:
        if os.path.exists(file_path):
            print(f"\nLoading: {file_path}")
            if content_key:
                print(f"  Using content key: {content_key}")

            success = rag_engine.load_and_index_file(file_path, content_key)
            if success:
                total_loaded += 1
                print(f"  ✅ Successfully loaded!")
            else:
                print(f"  ❌ Failed to load!")
        else:
            print(f"\n⚠️  File not found: {file_path}")

    # Show final stats
    print(f"\n📊 Loading complete!")
    print(f"Files processed: {total_loaded}/{len(sample_files)}")

    stats = rag_engine.get_collection_stats()
    if stats:
        print(f"Total documents in collection: {stats['total_documents']}")

    print("\n🎉 Sample data loading finished!")
    print("You can now run the chat application with: python app.py")


if __name__ == "__main__":
    load_sample_data()
