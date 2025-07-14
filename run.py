#!/usr/bin/env python3
"""
🚀 RAG Chat System Launcher
===========================
Simple startup script for the RAG Chat System.
Checks dependencies and launches the application.
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_env_file():
    """Check if .env file exists with required variables."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ Error: .env file not found!")
        print("Please create a .env file with:")
        print("OPENAI_API_KEY=your_openai_api_key")
        print("MONGODB_URI=your_mongodb_connection_string")
        return False

    # Check if required variables are present
    with open(env_file, "r") as f:
        content = f.read()

    required_vars = ["OPENAI_API_KEY", "MONGODB_URI"]
    missing_vars = []

    for var in required_vars:
        if var not in content or f"{var}=" not in content:
            missing_vars.append(var)

    if missing_vars:
        print(
            f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}"
        )
        return False

    print("✅ Environment configuration found")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import gradio
        import openai
        import pymongo
        import langchain

        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Error: Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def launch_application():
    """Launch the RAG chat application."""
    print("\n🚀 Launching RAG Chat System...")
    print("=" * 50)

    try:
        # Import and run the main application
        from app import create_interface

        demo = create_interface()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=True,
            show_api=False,
            quiet=True,
        )

    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return False

    return True


def main():
    """Main launcher function."""
    print("🔍 RAG Chat System Launcher")
    print("=" * 30)

    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Environment Configuration", check_env_file),
        ("Dependencies", check_dependencies),
    ]

    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            print(
                f"\n❌ Setup incomplete. Please fix the {check_name.lower()} issue and try again."
            )
            sys.exit(1)

    print("\n✅ All checks passed!")

    # Launch the application
    if not launch_application():
        print("\n❌ Failed to launch application.")
        sys.exit(1)


if __name__ == "__main__":
    main()
