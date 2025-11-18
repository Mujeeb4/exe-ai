# Exe: Local-First Agentic Code Assistant

**Codename:** Code Archeologist  
**Version:** 1.4.0  
**Status:** Active Development

## Overview

Exe is a Python-first, local-first AI pair programmer that turns your terminal into an intelligent "second brain." Unlike cloud-based tools, Exe runs entirely on your machine, indexing your codebase into a high-performance vector database (LanceDB) and maintaining a live understanding of your project as you code.

## Key Features

- **Zero Latency & Privacy**: Everything runs locally. No uploading code to third parties.
- **Live Context**: The AI knows about your edits the moment you save a file.
- **Agentic Editing**: Safely patches files using unified diffs, avoiding destructive overwrites.
- **Python-First**: Optimized specifically for Python (Django/Flask/FastAPI) workflows in v1.
- **Persistent Memory**: Remembers your conversation history across sessions using SQLite.
- **Focus Mode**: Restrict AI context to specific directories for improved accuracy.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Exe-ai.git
cd Exe-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Initialize Exe

```bash
Exe init
```

You'll be prompted for:
- Your OpenAI API key
- Whether to enable auto-apply mode (skips confirmation prompts)

### 2. Start Exe

```bash
Exe start
```

This will:
- Index your Python codebase
- Start the file watcher
- Launch the interactive REPL

### 3. Chat with Your Code

```
You: Explain the authentication logic in auth.py
Exe: [Provides detailed explanation based on your code]

You: Refactor the login function to use async/await
Exe: [Shows a unified diff patch]
Apply this patch? [y/N]: y
✓ Patched src/auth.py
```

## Commands

- **`exe init`**: Initialize Exe in the current directory
- **`exe start`**: Start Exe and begin interactive session
- **`exe focus <path>`**: Set focus mode to a specific directory or file

## How It Works

1. **Intelligent Chunking**: Uses AST parsing to split Python code into semantic chunks (functions, classes)
2. **Vector Storage**: Stores chunks in LanceDB with embeddings for semantic search
3. **Router Agent**: Classifies your intent and retrieves relevant context
4. **Coder Agent**: Generates answers or unified diff patches
5. **Safe Patching**: Applies changes using diffs, never overwrites entire files
6. **Live Updates**: Watches for file changes and re-indexes automatically

## Project Structure

```
Exe/
├── src/
│   ├── core/          # Router and Coder agents
│   ├── memory/        # Vector DB, embeddings, chunking
│   ├── ingestion/     # File scanning, watching, editing
│   └── interface/     # CLI and REPL
├── tests/             # Unit tests
├── .Exe/              # Config and session data (auto-created)
└── pyproject.toml     # Dependencies
```

## ⚙️ Configuration

Configuration is stored in `~/.Exe/config.json`:

```json
{
  "api_key": "sk-...",
  "auto_apply": false,
  "model": "gpt-4",
  "focus_path": null
}
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## Roadmap (v2.0)

- [ ] **Multi-Language Support**: Add Tree-Sitter for JS/TS/Go chunking
- [ ] **Local LLM Support**: Integration with Ollama/llama.cpp
- [ ] **Auto-Commit**: `exe commit` - automatically git commit changes
- [ ] **Plugin System**: Extensible architecture for custom agents

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with:
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting
- [LanceDB](https://lancedb.com/) - Vector database
- [OpenAI](https://openai.com/) - LLM API
- [Watchdog](https://python-watchdog.readthedocs.io/) - File monitoring

---

