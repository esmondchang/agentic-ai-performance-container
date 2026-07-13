# 🤖 Agentic AI Tutorial with Ollama

A hands-on tutorial for learning agentic AI patterns using **100% local, open-source models**. Build intelligent agents that can reason, retrieve information, use tools, and orchestrate complex workflows - all running on your own machine!

## 🎯 What You'll Learn

This tutorial teaches you four fundamental patterns in agentic AI:

1. **🧠 ReAct (Reasoning + Acting)** - How agents think step-by-step to solve problems
2. **📚 RAG (Retrieval Augmented Generation)** - How to ground AI responses in real data
3. **🛠️ Tool Use (MCP-style)** - How agents interact with external systems
4. **🔄 Workflow Orchestration** - How to combine everything into complex behaviors

## 📋 Prerequisites

- **Python 3.9+** 
- **8GB RAM** (16GB recommended for larger models)
- **10GB disk space** for models
- **macOS, Linux, or Windows** (with WSL2)


## 🐳 Container Quick Start

Use this path when you want the app and Ollama to run entirely in containers,
with an easy way to trigger multiple concurrent test users.

Container setup requires Docker Desktop. You do not need a local Python virtual
environment and you do not need to install Ollama on your machine.

### 1. Clone the repository

```bash
git clone https://github.com/esmondchang/agentic-ai-performance-container.git
cd agentic-ai-performance-container
```

### 2. Configure defaults

```bash
cp .env.example .env
```

Edit `.env` if you want different models, ports, request counts, or concurrency.

### 3. Start Ollama and download models

```bash
docker compose up -d ollama
docker compose --profile setup run --rm pull-models
```

This starts Ollama in a container and downloads the required models into the
Docker `ollama-data` volume. You only need to download them again when you
change model names or remove the volume.

### 4. Run the Streamlit app

```bash
docker compose up --build app
```

Open your browser to: **http://localhost:8501**

### 5. Rebuild after code changes

If you change Python code, rebuild the app image before testing again:

```bash
docker compose down
docker compose up -d ollama
docker compose --profile setup run --rm pull-models
docker compose up --build app
```

### 6. Health checks

Check that the containers are running:

```bash
docker compose ps
```

Check Ollama from your host machine:

```bash
curl http://localhost:11434/api/tags
```

Check Ollama from inside the app container network:

```bash
docker compose run --rm app python -c "import requests; print(requests.get('http://ollama:11434/api/tags').json())"
```

If the inside-container check works, the app can reach Ollama.

### 7. Trigger concurrent workflow users

```bash
REQUESTS=20 CONCURRENCY=5 docker compose --profile perf run --rm workflow-load-test
```

This runs `FinancialAgentWorkflow.analyze()` concurrently inside a container and
writes results to `results/workflow_performance_results.json`.

### 8. Trigger concurrent direct Ollama users

```bash
REQUESTS=20 CONCURRENCY=5 docker compose --profile perf run --rm ollama-load-test
```

This measures concurrent calls to Ollama `/api/generate` and writes results to
`results/ollama_performance_results.json`.

### Faster macOS option: container app with host Ollama

On macOS, Ollama running directly on the host can use Apple Metal acceleration,
while Ollama inside Docker runs in a Linux VM and can be much slower. If raw
Ollama tests are much slower in Docker, use this mode.

Start Ollama on your host:

```bash
ollama serve
ollama pull llama3.2
ollama pull qwen2.5
ollama pull nomic-embed-text
```

Run the containerized app against host Ollama:

```bash
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build app
```

Run the workflow load test against host Ollama:

```bash
REQUESTS=20 CONCURRENCY=5 docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml --profile perf run --rm workflow-load-test
```

Run the raw Ollama load test against host Ollama:

```bash
REQUESTS=20 CONCURRENCY=5 docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml --profile perf run --rm ollama-load-test
```

In this mode, the app and tests still run in containers, but model inference
uses `http://host.docker.internal:11434` so it reaches your host Ollama service.

### Optional: scale app containers behind a proxy

Streamlit itself can serve multiple sessions from one container. If you want
multiple app replicas, put them behind a reverse proxy or load balancer and
remove the fixed `APP_PORT:8501` host-port binding from the `app` service. The
default Compose file keeps one browser-facing app container mapped to
`localhost:8501`.

### Troubleshooting: `httpx.ConnectError: [Errno 111] Connection refused`

This usually means the app container cannot reach Ollama or the app image was
not rebuilt after code changes.

First rebuild and restart:

```bash
docker compose down
docker compose up -d ollama
docker compose --profile setup run --rm pull-models
docker compose up --build app
```

Then verify the app can reach Ollama:

```bash
docker compose run --rm app python -c "import requests; print(requests.get('http://ollama:11434/api/tags').json())"
```

Inside Docker, do not use `localhost:11434` from app code running in the
container, because `localhost` would point at the app container itself. Use
`http://ollama:11434` for the fully containerized setup, or
`http://host.docker.internal:11434` when using the host-Ollama override.

## 📁 Project Structure

```
agentic-ai-performance-container/
├── src/
│   ├── config.py           # Ollama configuration
│   ├── react_agent.py      # ReAct reasoning implementation
│   ├── rag_engine.py       # RAG system with vector storage
│   ├── tool_system.py      # Tool framework (MCP-style)
│   ├── workflow.py         # LangGraph workflow orchestration
│   ├── main.py            # Streamlit web interface
│   └── cli.py             # Command-line interface
├── data/                  # Data storage
│   ├── vector_store/      # FAISS vector database
│   └── traces/           # Execution traces (for debugging)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container image for the Streamlit app and tests
├── docker-compose.yml    # App, Ollama, model setup, and performance profiles
├── docker-compose.host-ollama.yml # Override for using host Ollama from containers
├── run.py               # Application launcher
├── test_ollama.py       # Ollama connection test
└── README.md           # This file
```

## 🎓 Tutorial Walkthrough

### Part 1: ReAct Pattern (Reasoning + Acting)

The ReAct pattern teaches agents to think before they act.

**Try it:**
1. Go to the **ReAct tab** in the web app
2. Ask: "What is Apple's P/E ratio and is it overvalued?"
3. Watch the reasoning steps unfold

**What's happening:**
```python
# The agent follows this loop:
Thought → Action → Observation → Thought → ...
```

**Key concepts:**
- **Thought**: Agent reasons about what to do next
- **Action**: Agent selects and executes a tool
- **Observation**: Agent sees the result
- **Iteration**: Process repeats until answer found

**Code to explore:** `src/react_agent.py`

### Part 2: RAG (Retrieval Augmented Generation)

RAG prevents hallucination by grounding responses in real documents.

**Try it:**
1. Go to the **RAG tab**
2. Click "Load Financial Documents" 
3. Ask: "What are the key risks?"
4. See retrieved documents and generated answer

**What's happening:**
```python
# RAG Pipeline:
Documents → Chunk → Embed → Store → Retrieve → Generate
```

**Key concepts:**
- **Chunking**: Split documents into manageable pieces
- **Embedding**: Convert text to vectors
- **Vector Store**: FAISS database for similarity search
- **Retrieval**: Find relevant chunks for a query
- **Generation**: Create answer using retrieved context

**Code to explore:** `src/rag_engine.py`

### Part 3: Tool System (MCP-style)

Tools extend what agents can do - like APIs for AI.

**Try it:**
1. Go to the **Tools tab**
2. Select `stock_data` tool
3. Enter ticker: `AAPL`
4. Click "Execute Tool"

**Available tools:**
- **stock_data**: Fetch real market data
- **calculator**: Perform calculations
- **web_search**: Search for information (mock)

**Key concepts:**
- **Tool Schema**: Define inputs/outputs
- **Validation**: Check parameters
- **Error Handling**: Graceful failures
- **Execution History**: Track tool use

**Code to explore:** `src/tool_system.py`

### Part 4: Complete Workflow

Combines all patterns into a comprehensive analysis system.

**Try it:**
1. Go to the **Workflow tab**
2. Enter ticker: `MSFT`
3. Click "Run Complete Analysis"
4. Explore all 5 result tabs

**What you'll see:**
- **Report**: Executive summary
- **Reasoning**: Step-by-step thought process
- **RAG Context**: Retrieved documents
- **Technical**: Market indicators
- **Sentiment**: Social media analysis

**Code to explore:** `src/workflow.py`

## 🛠️ Configuration

### Environment Variables

Create a `.env` file (optional):
```bash
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434

# Model selection
REASONING_MODEL=llama3.2:latest
ANALYSIS_MODEL=llama3.2:latest
EMBEDDING_MODEL=nomic-embed-text:latest

# Settings
LOG_LEVEL=INFO
VECTOR_STORE_TYPE=faiss
```

### Using Different Models

You can experiment with different models:

```bash
# Smaller, faster model
ollama pull phi3
# Update REASONING_MODEL=phi3:latest in .env

# Better at following instructions
ollama pull mistral
# Update REASONING_MODEL=mistral:latest in .env

# Larger, more capable
ollama pull llama3.2:70b  # Requires 40GB RAM!
```

## 🔍 CLI Usage

The command-line interface is great for learning and debugging:

### Test Individual Components

```bash
# Test ReAct reasoning
python src/cli.py react "What is 2+2 and why?"

# Test RAG system
python src/cli.py rag AAPL "What are the risks?"

# Test tool execution
python src/cli.py tool stock_data --params '{"ticker": "GOOGL"}'

# Run complete workflow
python src/cli.py workflow NVDA --query "Should I invest?"
```

### Check System Status

```bash
# List available models
python src/cli.py models

# Verify setup
python src/cli.py check
```

## 🐛 Troubleshooting

### Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| "Ollama not found" | Make sure `ollama serve` is running |
| "Model not found" | Run `ollama pull llama3.2` |
| No reasoning trace shown | Try `ollama pull mistral` (better at structured output) |
| Import errors | Check virtual environment is activated |
| Port 8501 in use | Run `streamlit run src/main.py --server.port 8502` |
| Out of memory | Use smaller model: `ollama pull phi3` |

### Debug Mode

Enable verbose logging:
```python
# In src/config.py
verbose: bool = True  # Set to True for debug output
```

Check console output while running for detailed logs.

## 📚 Understanding the Code

### Key Design Patterns

1. **State Management** - How data flows through the system
2. **Prompt Engineering** - Crafting effective LLM instructions  
3. **Error Recovery** - Graceful handling of failures
4. **Caching** - Preserving results across interactions
5. **Modular Design** - Each component works independently

### Learning Path

**Beginner:**
1. Run the web app and try each tab
2. Read the code comments in `src/react_agent.py`
3. Modify a prompt and see what changes

**Intermediate:**
1. Add a new tool in `src/tool_system.py`
2. Create custom documents for RAG
3. Modify the workflow to add a new step

**Advanced:**
1. Implement a new reasoning strategy
2. Add a different vector store
3. Create a production deployment

## 🎯 Educational Exercises

### Exercise 1: Add a New Tool
Create a weather tool that returns mock weather data.

**Hints:**
- Look at `StockDataTool` in `src/tool_system.py`
- Define schema with city parameter
- Return temperature and conditions

### Exercise 2: Improve RAG
Add PDF document loading to the RAG system.

**Hints:**
- Use `pypdf` library (already installed)
- Add method to `RAGEngine` class
- Handle text extraction and chunking

### Exercise 3: Custom Workflow
Create a news analysis workflow.

**Hints:**
- Combine sentiment analysis with RAG
- Add news-specific tools
- Generate summary report

## 📊 Performance Tips

- **Model Selection**: Start with smaller models for testing
- **Caching**: Results are cached in session state
- **Batch Processing**: Process multiple documents at once
- **GPU Acceleration**: Use `ollama run --gpu` if available

## 🤝 Contributing

This is an educational project! Contributions welcome:

- Add new examples
- Improve documentation
- Fix bugs
- Add new features

## 📖 Resources

### Documentation
- [Ollama Docs](https://ollama.ai/docs)
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Streamlit Docs](https://docs.streamlit.io/)

### Key Papers
- [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Toolformer](https://arxiv.org/abs/2302.04761)

### Videos & Tutorials
- [Building Agents with LangGraph](https://www.youtube.com/watch?v=example)
- [Local LLMs with Ollama](https://www.youtube.com/watch?v=example)

## 📄 License

MIT License - Use freely for learning!

## 🙏 Acknowledgments

- **Ollama** team for making local LLMs accessible
- **LangChain** for the excellent framework
- **Meta, Mistral, Qwen** teams for open-source models
- The open-source AI community

---

**Made for Learning** 🎓 | **100% Local** 🏠 | **No API Keys** 🔐 | **Open Source** 📖

## Support

If you found this tutorial helpful, please ⭐ star the repository!

Questions? Open an issue on GitHub.

Happy Learning! 🚀
