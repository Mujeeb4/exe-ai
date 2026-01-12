# Exe: Local-First Agentic Code Assistant

**Version:** 1.4.0  
**Python:** ≥3.9  
**License:** MIT

---

## What is Exe?

Exe is a **local-first AI pair programmer** that indexes your codebase into a high-performance vector database and maintains a live understanding of your project as you code. The vector database and conversation history are stored locally, while AI inference is handled by your chosen provider (Google, OpenAI, or Anthropic).

---

## Features

### 🔒 Local-First Architecture
- Vector database and codebase index stored locally
- Conversation history persisted on your machine
- API keys are stored locally in `~/.exo/config.json`
- Code context is sent to AI providers only when you make queries

### 🧠 Intelligent Code Understanding
- **AST-based Chunking**: Parses Python code into semantic chunks (functions, classes, methods) preserving logical boundaries
- **Vector Embeddings**: Converts code chunks to embeddings for semantic search
- **LanceDB Storage**: High-performance local vector database for instant retrieval

### 🤖 Multi-Agent Architecture
- **Router Agent**: Classifies user intent into four categories:
  - `question` - Asking about code/architecture
  - `code_edit` - Modifying/fixing code
  - `refactor` - Improving code structure
  - `explain` - Detailed code explanations
- **Coder Agent**: Generates code edits using unified diff format with tools:
  - `search_code` - Semantic search across codebase
  - `read_file` - Read file contents
  - `list_files` - List available files
  - `edit_file` - Apply unified diff patches

### ⚡ Real-Time File Watching
- Monitors file changes using Watchdog
- Auto-reindexes modified files with debouncing
- Loop prevention to avoid re-indexing self-applied patches

### 🛡️ Safe Code Editing
- Applies changes using **unified diff patches** only
- Never overwrites entire files
- Creates backups before modifications
- Full undo support to revert changes

### 🎯 Focus Mode
- Restrict AI context to specific directories or files
- Improves accuracy and relevance for large codebases

### 💾 Persistent Memory
- Conversation history saved in SQLite
- Session state preserved across restarts
- Remembers context from previous interactions

---

## Multi-Provider LLM Support

Exe supports multiple AI providers. Configure different models for routing (fast/cheap) and coding (advanced):

| Provider | Router Model | Coder Model | Embeddings |
|----------|--------------|-------------|------------|
| **Google** | `gemini-1.5-flash` | `gemini-2.0-flash-exp` | `text-embedding-004` |
| **OpenAI** | `gpt-4o-mini` | `gpt-4o` | `text-embedding-3-small` |
| **Anthropic** | `claude-3-haiku` | `claude-3-5-sonnet` | - |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Mujeeb4/exe-ai.git
cd exe-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e .
```

### Requirements
- Python ≥3.9
- API key from at least one provider (Google AI Studio, OpenAI, or Anthropic)

---

## Usage

### Initialize Exe

```bash
exe init
```

The initialization wizard will:
1. **Model Selection** - Choose router and coder models (recommended defaults provided)
2. **API Keys** - Enter keys only for providers you selected
3. **Auto-apply Mode** - Optionally skip confirmation prompts for patches

### Start Interactive Session

```bash
exe start
```

This will:
- Scan and index your Python codebase
- Start the real-time file watcher
- Launch the interactive REPL

### Interactive Commands

Once in the REPL:

```
You: What does the login function do?
Exe: [Searches codebase, provides explanation with file references]

You: Add error handling to the database connection
Exe: [Shows unified diff patch]
Apply this patch? [y/N]: y
✓ Patched src/db.py

You: /undo
✓ Reverted last change to src/db.py

You: /focus src/auth/
✓ Focus set to src/auth/

You: /help
[Shows all available commands]
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `exe init` | Initialize Exe with model and API key configuration |
| `exe start` | Start interactive REPL session |
| `exe focus <path>` | Set focus mode to a specific directory |
| `exe models` | Change router/coder model configuration |
| `exe apikeys` | Update API keys |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Interface                            │
│                    (Typer + Rich REPL)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      Router Agent                                │
│         Intent Classification (question/edit/refactor/explain)   │
│                   Uses: LlmAgent + Pydantic Schema              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      Coder Agent                                 │
│              Code Generation & Unified Diff Patches              │
│          Tools: search_code, read_file, edit_file               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     Memory Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Chunker    │  │  Embedder   │  │   VectorDB (LanceDB)    │  │
│  │ (AST-based) │  │(Google/OAI) │  │   Semantic Search       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   Ingestion Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Scanner    │  │  Watcher    │  │   Editor                │  │
│  │(Initial idx)│  │ (Live sync) │  │  (Unified diff apply)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
exe-ai/
├── src/
│   ├── main.py              # CLI entry point (init, start, focus, models, apikeys)
│   ├── config.py            # Configuration management with Pydantic
│   ├── session.py           # Session state management
│   ├── core/
│   │   ├── router_agent.py  # Intent classification agent
│   │   ├── coder_agent.py   # Code generation agent
│   │   ├── tools.py         # ADK function tools (search, read, edit)
│   │   ├── model_factory.py # LLM model instantiation
│   │   └── adk_adapters.py  # Google ADK integration adapters
│   ├── memory/
│   │   ├── chunker.py       # AST-based Python code chunker
│   │   ├── embedder.py      # Text-to-vector embeddings (Google/OpenAI)
│   │   ├── db.py            # LanceDB vector database wrapper
│   │   └── history.py       # SQLite conversation history
│   ├── ingestion/
│   │   ├── scanner.py       # Initial codebase indexing
│   │   ├── watcher.py       # Real-time file monitoring
│   │   └── editor.py        # Safe unified diff patching
│   ├── interface/
│   │   ├── repl.py          # Interactive REPL with undo support
│   │   └── console.py       # Rich console formatting
│   └── utils/
│       └── model_selector.py # Interactive model selection wizard
├── tests/                   # Unit tests (pytest)
├── pyproject.toml           # Dependencies and package config
└── README.md
```

---

## Configuration

Configuration is stored in `~/.exo/config.json`:

```json
{
  "google_api_key": "...",
  "openai_api_key": "...",
  "anthropic_api_key": "...",
  "router_model": "gemini-1.5-flash",
  "coder_model": "gemini-2.0-flash-exp",
  "embedding_model": "text-embedding-004",
  "embedding_provider": "google",
  "auto_apply": false,
  "focus_path": null
}
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `typer` | CLI framework |
| `rich` | Terminal formatting and REPL UI |
| `lancedb` | Local vector database |
| `watchdog` | File system monitoring |
| `pydantic` | Configuration and schema validation |
| `google-adk` | Google Agent Development Kit |
| `google-genai` | Google Generative AI SDK |
| `openai` | OpenAI API client |
| `anthropic` | Anthropic API client |
| `litellm` | Unified LLM interface |

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_chunker.py -v
```

---

## License

MIT License

