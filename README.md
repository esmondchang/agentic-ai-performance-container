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


## 🐳 Quick Start: Host Ollama + Containerized App

This is the recommended deployment for this repository. Ollama runs directly
on the host with `ollama serve`; Docker runs only the Streamlit app and optional
load-test clients. The host owns the models, CPU/GPU access, and port `11434`.
This Compose file uses host networking and is intended for Linux hosts.

Do not combine `docker-compose.local-ollama.yml` with `docker-compose.yml`. The
host file is standalone and intentionally contains no Ollama service.

### 1. Clone and configure the repository

```bash
git clone https://github.com/esmondchang/agentic-ai-performance-container.git
cd agentic-ai-performance-container
cp .env.example .env
```

### 2. Install and start Ollama on the host

Install Ollama using the instructions for your operating system, then start the
host service:

```bash
ollama serve
```

If Ollama is managed by systemd, verify it instead:

```bash
sudo systemctl status ollama
```

### 3. Pull models on the host

```bash
ollama pull llama3.2:latest
ollama pull qwen2.5:latest
ollama pull nomic-embed-text:latest
ollama list
```

There is no Compose `pull-models` step in this deployment because the Ollama
server and model store are outside Docker.

### 4. Verify host Ollama

Confirm Ollama works locally:

```bash
curl http://127.0.0.1:11434/api/tags
sudo ss -ltnp 'sport = :11434'
```

`docker-compose.local-ollama.yml` uses Linux host networking, so the app
container reaches Ollama at `http://127.0.0.1:11434`. Ollama can remain bound
to loopback; it does not need to be exposed on `0.0.0.0`.

### 5. Verify the standalone Compose file

```bash
docker compose -f docker-compose.local-ollama.yml config --services
```

Expected services:

```text
app
```

There should be no `ollama` service.

Test connectivity from the app image:

```bash
docker compose -f docker-compose.local-ollama.yml run --rm app \
  python -c "import requests; r=requests.get('http://127.0.0.1:11434/api/tags'); print(r.status_code, r.text[:500])"
```

A `200` response confirms that Docker can reach the host Ollama server.

### 6. Build and run the app

```bash
docker compose -f docker-compose.local-ollama.yml up --build app
```

Open **http://localhost:8501**. To run in the background:

```bash
docker compose -f docker-compose.local-ollama.yml up -d --build app
docker compose -f docker-compose.local-ollama.yml logs -f app
```

After Python code changes, run the same `up --build app` command. Restarting
Ollama or pulling models again is unnecessary.

### 7. Run concurrent workflow tests

```bash
docker compose -f docker-compose.local-ollama.yml run --rm \
  -v "$PWD/results:/app/results" app \
  python tests/performance_test1.py \
  --requests 20 --concurrency 5 --warmup 2 --ticker AAPL \
  --output /app/results/workflow_performance_results.json
```

Results are written to `results/workflow_performance_results.json`.

Run a tenant-isolated concurrency test:

```bash
docker compose -f docker-compose.local-ollama.yml run --rm \
  -v "$PWD/results:/app/results" app \
  python tests/multitenant_test.py \
  --tenants 3 --users-per-tenant 2 --requests-per-user 2 \
  --ticker AAPL \
  --output /app/results/multitenant_results.json
```

This example starts six concurrent simulated users and runs twelve complete
LangGraph analyses. Users belonging to a tenant share that tenant's FAISS
store, while every tenant receives a separate directory. The JSON report
includes latency, throughput, failures, tenant paths, and an isolation audit.
Add `--keep-data` to retain the generated tenant stores after the audit.

Test direct Ollama concurrency:

```bash
docker compose -f docker-compose.local-ollama.yml run --rm \
  -v "$PWD/results:/app/results" app \
  python tests/ollama_model_performance.py \
  --base-url http://127.0.0.1:11434 --model llama3.2:latest \
  --requests 20 --concurrency 5 --warmup 2 \
  --output /app/results/ollama_performance_results.json
```

Results are written to `results/ollama_performance_results.json`.

### Per-task CPU and GPU report

Rebuild the app after changing the workflow instrumentation:

```bash
docker compose -f docker-compose.local-ollama.yml build app
```

Run the resource monitor on the Linux host. It launches the test, samples the
host and GPUs, and correlates samples with each LangGraph node window:

```bash
python3 tests/task_resource_monitor.py \
  --trace-file "$PWD/results/task-events.jsonl" \
  --output "$PWD/results/task-resources.json" \
  --gpu-devices 0,1 \
  --interval 0.25 \
  -- docker compose -f docker-compose.local-ollama.yml run --rm \
    -e RESOURCE_TRACE_FILE=/app/results/task-events.jsonl \
    -v "$PWD/results:/app/results" \
    app python tests/multitenant_test.py \
    --tenants 1 \
    --users-per-tenant 1 \
    --requests-per-user 1 \
    --ticker AAPL \
    --output /app/results/multitenant-smoke.json
```

Inspect the RAG task:

```bash
jq '.tasks[] | select(.task == "RAG Analysis")' \
  results/task-resources.json
```

The report contains wall time, application CPU time, local Ollama CPU time,
host CPU capacity, GPU active time, average GPU utilization, peak GPU memory,
and sample coverage for every node. With concurrent users, overlapping nodes
share Ollama and GPU measurements; use one concurrent user for clean per-task
attribution.

### What `-f` means

`-f` means `--file`; it selects the Compose file to use:

```bash
docker compose -f docker-compose.local-ollama.yml up --build app
```

Always include it for this deployment. Running `docker compose up` without
`-f` selects `docker-compose.yml`, which defines a containerized Ollama service
and can conflict with host Ollama on port `11434`.

### Multi-user and multi-tenant testing

One Streamlit app container can serve multiple browser sessions. All users
share the host Ollama server, whose `OLLAMA_NUM_PARALLEL` setting controls how
many inference requests execute concurrently; extra requests wait in its queue.

Multiple app replicas can also share host Ollama. Because this file uses host
networking, each replica needs a unique Streamlit listen port or a reverse
proxy. For true tenant isolation, use a separate FAISS vector-store directory
or volume per tenant. The default shared `/app/data/vector_store` path is
appropriate for a demo, not isolated tenant data.

Select two GPUs for local Ollama with:

```bash
sudo scripts/switch-ollama-device.sh gpu 0,1
```

For an explicit multi-GPU list, the script sets `CUDA_VISIBLE_DEVICES`, enables
`OLLAMA_SCHED_SPREAD`, and configures one `OLLAMA_NUM_PARALLEL` slot per selected
GPU. Running `gpu` without a device list retains Ollama's automatic scheduling.

### Troubleshooting

If the app reports `Connection refused`:

```bash
curl http://127.0.0.1:11434/api/tags
sudo ss -ltnp 'sport = :11434'
docker compose -f docker-compose.local-ollama.yml run --rm app \
  python -c "import requests; print(requests.get('http://127.0.0.1:11434/api/tags').status_code)"
```

If Docker reports that host port `11434` is already in use, the wrong Compose
file was selected. Stop that attempt and use only:

```bash
docker compose -f docker-compose.local-ollama.yml up --build app
```

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
├── docker-compose.gpu.yml # NVIDIA GPU override for the Ollama service
├── docker-compose.local-ollama.yml # Standalone app stack using local Ollama
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
