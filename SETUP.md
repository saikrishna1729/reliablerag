# Setup Guide

## Prerequisites

### 1. Python 3.13+

**macOS:**
```bash
brew install pyenv
pyenv install 3.13
pyenv local 3.13
```

**Windows:**
```powershell
# Install pyenv-win
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"

pyenv install 3.13
pyenv local 3.13
```

Verify:
```bash
python --version
```

### 2. uv

**macOS:**
```bash
brew install uv
```

**Windows:**
```powershell
winget install astral-sh.uv
```

### 3. Ollama

**macOS:**
```bash
brew install ollama
```

**Windows:** Download the installer from [ollama.com](https://ollama.com/download/windows).

Start the daemon:
```bash
ollama serve
```

Pull the required models:
```bash
ollama pull embeddinggemma:300m-qat-q4_0
ollama pull gemma3:4b-it-q4_K_M
```

---

## Install dependencies

```bash
uv sync
```

This creates a `.venv` in the project root and installs all dependencies (including dev deps like `ipykernel`).

---

## Environment configuration

```bash
cp .env.example .env
```

Edit `.env` to override any defaults:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `embeddinggemma:300m-qat-q4_0` | Ollama embedding model |
| `LLM_MODEL` | `gemma3:4b-it-q4_K_M` | Ollama generation model |
| `LLM_TEMPERATURE` | `0` | 0 = deterministic, higher = more creative |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | Local ChromaDB storage path |
| `RETRIEVER_TOP_K` | `2` | Chunks retrieved per query |

---

## IDE setup

### PyCharm Pro

1. Open the project folder.
2. Click **"Set up uv environment"** in the banner that appears at the top of any file — PyCharm will run `uv sync` and configure the interpreter automatically.
3. Alternatively: **Settings → Project → Python Interpreter → Add → Existing Environment** → select `.venv/bin/python`.

---

Make sure Ollama is running before executing any notebooks or scripts.
