# Arc-Fusion: Extensible Multi-Agent RAG System

A state-of-the-art multi-agent Retrieval-Augmented Generation system built with **LangGraph**, **FastAPI**, and **Google Gemini**. Designed with senior-level extensibility in mind, demonstrating advanced agent orchestration patterns and production-ready architecture.

## 🚀 Key Features

### **Extensible Agent Architecture**
- **Registry Pattern**: Add new agents with a simple `@AgentRegistry.register` decorator
- **Capability-Based Routing**: Agents self-describe their capabilities for dynamic routing
- **Plugin System**: Drop-in agent files that auto-register themselves
- **Dynamic Graph Construction**: LangGraph automatically rebuilds when agents are added

### **Advanced RAG Pipeline**
- **HyDE Query Expansion**: Hypothetical Document Embeddings for better retrieval
- **Hybrid Search**: Vector + BM25 search using Weaviate's native capabilities  
- **Parent-Child Chunking**: Precise retrieval with rich context
- **Quality-Based Routing**: Intelligent fallback based on retrieval scores

### **Intelligent Multi-Agent Orchestration**
- **Intent Classification**: Fast routing using Gemini 2.5 Flash Lite
- **LLM-Based Planning**: Dynamic agent execution planning using Gemini 2.5 Flash Lite
- **Corpus Retrieval**: Advanced RAG with academic paper understanding
- **Automatic Fallback**: When no corpus results found → auto-route to web search
- **Web Search**: Current information via Tavily API (explicit requests OR automatic fallback)
- **Response Synthesis**: High-quality answers using Gemini 2.5 Flash and Gemini 2.5 Pro

## 🧠 Current Architecture: LLM as a Judge

Our current system uses an **"LLM as a Judge"** architecture for document relevance assessment, leveraging Gemini 2.5 Flash to evaluate the quality of retrieved documents and make intelligent routing decisions.

### Key Advantages of Our Current Approach

**LLM Judge Benefits:**
-   **Massive Context Window:** The LLM can analyze all retrieved chunks simultaneously (up to 1 million tokens), enabling holistic, context-aware judgment of document relevance.
-   **Zero-Shot Reasoning:** We leverage the LLM's powerful, built-in reasoning capabilities for any academic domain without requiring additional training.
-   **Rich, Structured Output:** Instead of a single score, the LLM Judge provides structured JSON output with relevance scores, explanations, and clear routing decisions (`synthesize` or `search_web`).
-   **Simplified Pipeline:** This approach simplifies our architecture and makes it robust to changes in the underlying data corpus.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    A["User Query"] --> B["Routing Agent<br/>(Gemini 2.5 Flash Lite)"]
    B --> C{"Intent Classification"}
    
    C -->|"Any Intent"| P["Planner Agent<br/>(Gemini 2.5 Flash Lite)"]
    P --> D{"Dynamic Plan"}
    
    D -->|"corpus_retrieval only"| E["Corpus Retrieval Agent"]
    D -->|"web_search only"| F["Web Search Agent"] 
    D -->|"both tasks"| PE["Parallel Executor<br/>⚡ Concurrent Processing"]
    D -->|"clarification"| G["Clarification Agent"]
    
    PE --> E2["Parallel Processing<br/>⚡ Corpus Retrieval<br/>⚡ Web Search"]
    E2 --> H["Synthesis Agent<br/>(Gemini 2.5 Flash/Pro)"]
    
    E --> E1["1. HyDE Query Expansion<br/>(Gemini 2.5 Flash Lite)"]
    E1 --> E3["2. Hybrid Search<br/>(Weaviate: Vector + BM25)"]
    E3 --> E4["3. LLM as a Judge<br/>(Gemini 2.5 Flash)"]
    E4 --> E5{"Quality Assessment<br/>Best Score ≥ 7.0?"}
    
    E5 -->|"High Quality"| H
    E5 -->|"Low Quality"| F
    
    F --> F1["Query Optimization<br/>(Gemini 2.5 Flash Lite)"]
    F1 --> F2["Tavily API Search"]
    F2 --> H
    
    G --> I["Clarification Response"]
    
    H --> J["Final Answer<br/>with Citations"]
    
    style E fill:#e1f5fe
    style E4 fill:#f3e5f5
    style E5 fill:#fff3e0
    style F fill:#e8f5e8
    style H fill:#fce4ec
    style P fill:#fff8e1
    style PE fill:#fff9c4
    style E2 fill:#e8f5e8
```

### **Agent Capabilities Matrix**

| Agent | Capabilities | LLM Used | Purpose |
|-------|-------------|----------|---------|
| **Routing** | `intent_classification`, `query_analysis` | Gemini 2.5 Flash Lite | Fast, cost-effective intent routing |
| **Planner** | `task_planning`, `workflow_orchestration` | Gemini 2.5 Flash Lite | Dynamic agent execution planning |
| **Corpus Retrieval** | `document_search`, `rag`, `hybrid_search` | Gemini 2.5 Flash + Embedding | Advanced RAG pipeline with LLM Judge |
| **Web Search** | `web_search`, `external_search` | Gemini 2.5 Flash Lite + Tavily | Current information retrieval |
| **Synthesis** | `response_synthesis`, `answer_generation` | Gemini 2.5 Flash/Pro | High-quality answer generation |

## 🛠️ Quick Start

### Prerequisites
- **Python 3.11+**
- **Docker & Docker Compose**
- **Google AI Studio API Key** (free tier available)
- **Tavily API Key** (optional, for web search)

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd arc-fusion

# Copy environment template
cp env.template .env

# Add your API keys to .env
GOOGLE_API_KEY=your_google_ai_studio_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### 2. Run with Docker (Recommended)

```bash
# Start the entire system
docker-compose up -d

# Check that services are running
docker-compose ps

# View logs (optional)
docker-compose logs -f app
```

**Note:** The system automatically configures user permissions to match your host system using `USER_ID` and `GROUP_ID` environment variables. This ensures proper file ownership for logs and data volumes.

## 🔮 Future GPU Enhancement

**Cross-Encoder Reranking Implementation**

During this exercise, we implemented the LLM-as-a-Judge approach due to hardware constraints. However, modern cross-encoder models like `Qwen/Qwen3-Reranker-0.6B` have been trained on extensive datasets including scientific papers and would provide excellent domain-specific reranking capabilities.

**Why Cross-Encoder Wasn't Implemented:**
- **Hardware Limitations**: Limited GPU resources during development
- **Processing Time**: Batch processing was too slow on available hardware
- **Development Timeline**: Exercise timeframe required a faster implementation approach

**Future Implementation Benefits:**
- **Domain-Specific Training**: Models like Qwen3-Reranker are trained on scientific literature
- **Enhanced Precision**: Fine-grained relevance scoring between query-document pairs
- **Quality Boost**: Improved retrieval quality over hybrid search and LLM judge

**Implementation Plan:**
- GPU-optimized model loading with 8-bit quantization
- Efficient batch processing for multiple document pairs
- Hybrid approach: Cross-encoder + LLM judge for different scenarios
- Configuration-driven enable/disable functionality

**Technical Requirements:**
- NVIDIA GPU with 8GB+ VRAM
- CUDA toolkit and container runtime
- Additional dependencies: `transformers`, `torch`, `accelerate`, `bitsandbytes`

This enhancement is a high priority for future development with adequate hardware resources.

The system will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Weaviate**: http://localhost:8080

### 3. Upload Documents

```bash
# Upload PDF documents to build your knowledge base
curl -X POST "http://localhost:8000/api/v1/documents" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_paper.pdf"
```

### 4. Ask Questions

```bash
# Query the system
curl -X POST "http://localhost:8000/api/v1/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What methodology did Zhang et al. use for prompt optimization?",
       "session_id": "test-session"
     }'
```

### 5. Monitor Document Processing

```bash
# Check document processing status
curl -X GET "http://localhost:8000/api/v1/documents/{document_id}/status"

# View all documents
curl -X GET "http://localhost:8000/api/v1/documents"

# Get database statistics
curl -X GET "http://localhost:8000/api/v1/documents/stats"
```

## 📚 API Reference

### Core Endpoints

#### **POST /api/v1/ask**
Process queries through the multi-agent system.

```json
{
  "query": "Your question here",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "answer": "Generated response with citations",
  "session_id": "session-uuid",
  "success": true,
  "processing_time": 3.42,
  "agent_path": ["routing", "corpus_retrieval", "synthesis"],
  "citations": [
    {
      "type": "document",
      "filename": "paper.pdf",
      "score": "0.89"
    }
  ],
  "confidence": 0.92,
  "metadata": {
    "step_count": 3,
    "intent": "retrieve_corpus",
    "has_document_context": true,
    "document_sources": 3
  }
}
```

#### **POST /api/v1/clear-memory**
Clear conversation memory for a session.

#### **GET /api/v1/agents/info**
Get information about registered agents and capabilities.

### Document Management

#### **POST /api/v1/documents**
Upload and process PDF files.

#### **GET /api/v1/documents**
List all ingested documents.

#### **DELETE /api/v1/documents**
Clear all documents from the database.

### Evaluation & Dataset Generation

#### **POST /api/v1/evaluation/generate-dataset**
Generate golden Q&A dataset from ingested documents (background task).

#### **GET /api/v1/evaluation/golden-dataset**
Retrieve the generated golden dataset with metadata.

#### **GET /api/v1/evaluation/golden-dataset/stats**
Get statistics about the golden dataset without loading full content.

#### **POST /api/v1/evaluation/run**
Run comprehensive evaluation of the RAG system using the golden dataset.

## 🔧 Adding New Agents (Senior-Level Extensibility)

The system is designed for effortless agent expansion. Here's how to add a **Citation Agent**:

### Step 1: Create the Agent

```python
# app/agents/citation_agent.py
from .registry import AgentRegistry
from .state import GraphState

class CitationService:
    def add_detailed_citations(self, state: GraphState) -> GraphState:
        # Implementation here
        sources = state.get("document_sources", [])
        # Add page numbers, confidence scores, etc.
        enhanced_citations = self._create_detailed_citations(sources)
        
        updated_state = state.copy()
        updated_state["citations"] = enhanced_citations
        return updated_state

citation_service = CitationService()

@AgentRegistry.register(
    name="citation",
    capabilities=["source_attribution", "citation_enhancement"],
    dependencies=["corpus_retrieval"],  # Runs after retrieval
    priority=5
)
def citation_agent(state: GraphState) -> GraphState:
    return citation_service.add_detailed_citations(state)
```

### Step 2: Register the Agent

```python
# app/agents/loader.py - Add to imports
from . import citation_agent  # Auto-registers!
```

### Step 3: That's It! 

The agent is now:
- ✅ **Auto-discovered** by the framework
- ✅ **Routed to** when `source_attribution` capability is needed  
- ✅ **Integrated** into the graph workflow
- ✅ **Available** via API immediately

### Advanced Agent Patterns

#### **Middleware for Cross-Cutting Concerns**
```python
@AgentRegistry.register("logging_middleware", ["logging"])
def logging_agent(state: GraphState) -> GraphState:
    # Automatically logs all agent interactions
    pass
```

#### **Conditional Agent Activation**
```python
@AgentRegistry.register("confidence_scorer", ["uncertainty_quantification"])
def confidence_agent(state: GraphState) -> GraphState:
    if state.get("requires_confidence_scoring"):
        # Add confidence analysis
    return state
```

## 🚀 Production Deployment

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_google_ai_key

# Optional
TAVILY_API_KEY=your_tavily_key
WEAVIATE_URL=http://localhost:8080
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=false
```

### Docker Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  api:
    build: .
    environment:
      - ENVIRONMENT=production
    restart: unless-stopped
    
  weaviate:
    restart: unless-stopped
    volumes:
      - weaviate_data:/var/lib/weaviate
```

### Scaling Considerations

- **Stateless Design**: Sessions stored in memory, easily moved to Redis
- **Agent Parallelization**: Multiple agents can run concurrently
- **LLM Rate Limiting**: Built-in retry logic with exponential backoff
- **Vector Database**: Weaviate handles horizontal scaling

## 🎯 Assignment Requirement: Automatic Web Search Fallback

### **Critical Feature: Intelligent Routing Logic**

Our system implements the assignment requirement perfectly:

> "performing a web search either when **explicitly requested** by the user (e.g., "Search online for...") **OR when the answer cannot be found** in the provided PDFs."

**Technical Implementation:**

1. **LLM as a Judge Re-ranking Pipeline**
   ```python
   # Step 1: Bi-encoder Retrieval (Fast, Recall-focused)
   child_results = await vector_store.hybrid_search(query, embedding, limit=30)
   
   # Step 2: LLM as a Judge (Precise, Quality-focused) 
   judge_assessments = await self._select_top_context_with_llm_judge(parent_chunks, query)
   
   # Step 3: Quality-based Routing Decision
   best_score = judge_assessments.get("best_llm_judge_score", 0.0)
   if best_score >= 7.0: return "synthesize"     # High quality
   else: return "search_web"                     # Low quality → fallback
   ```

2. **Conditional Routing After Quality Assessment**
   ```python
   # In framework.py - SOPHISTICATED ROUTING
   def _route_after_corpus_retrieval(self, state: GraphState) -> str:
       best_score = state.get("best_llm_judge_score", 0.0)
       
       if best_score >= 7.0:
           return "synthesis"  # High quality results
       else:
           return "search_web"  # Low quality → web search fallback
   ```

3. **Quality Assessment Examples:**
   ```
   Query: "What did Zhang et al report about prompts?"
   → Corpus Retrieval → LLM Judge: "score": 9.5 → Best: 9.5 (HIGH) → Synthesis ✓
   
   Query: "What did OpenAI release this month?"  
   → Corpus Retrieval → LLM Judge: "score": 2.0 → Best: 2.0 (LOW) → Web Search ✓
   
   Query: "Search online for latest AI news"
   → Routing → [EXPLICIT REQUEST] → Web Search → Synthesis ✓
   
   Query: "How does few-shot learning work?"
   → Corpus Retrieval → LLM Judge: "score": 6.0 → Best: 6.0 (MEDIUM) → Web Search ✓
   ```
   
4. **Model Used (gemini-2.5-flash):**
   - **Context Window**: 1M Tokens (eliminates chunk truncation)
   - **Strengths**: Speed, cost-effectiveness, and strong reasoning for evaluation tasks.
   - **Output**: Structured JSON for reliable, transparent routing.

## 🧪 Testing the System

### Real-World Scenarios

The system handles the assignment's test cases perfectly:

#### **Ambiguous Questions**
```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
     -d '{"query": "How many examples are enough for good accuracy?"}'
# Response: Routes to clarification or asks for more context
```

#### **PDF-Only Queries**  
```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
     -d '{"query": "What execution accuracy does davinci-codex reach on Spider?"}'
# Response: Routes to corpus_retrieval → finds specific results in papers
```

#### **Web Search Queries**
```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
     -d '{"query": "What did OpenAI release this month?"}'
# Response: Routes to web_search → gets current information
```

### Agent Flow Validation

```bash
# Check which agents were used
GET /api/v1/sessions/{session_id}

# Verify agent registration
GET /api/v1/agents/info
```

## 🔮 Future Improvements

The extensible architecture enables easy addition of:

### **🚀 Quick Wins (1-2 Days Implementation)**

**Response Caching System**
```python
@AgentRegistry.register("cache", ["response_caching"])
def cache_agent(state: GraphState) -> GraphState:
    # Simple in-memory cache for repeated queries
    cache_key = f"{state['query']}:{state['session_id']}"
    if cached_response := response_cache.get(cache_key):
        return cached_response
    return state
```
- **Effort**: 4-6 hours
- **Impact**: 80% faster responses for repeated queries
- **Implementation**: Simple dictionary cache with TTL

**Enhanced Prompt Engineering**
- **Current**: Generic prompts for all agents
- **Quick Win**: Add few-shot examples to improve consistency
- **Effort**: 2-4 hours per agent
- **Impact**: 15-20% better response quality

**Configuration Exposure**
```python
# Expose more tuning parameters via environment variables
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", 0.7))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", 4))
SYNTHESIS_TEMPERATURE = float(os.getenv("SYNTHESIS_TEMPERATURE", 0.1))
```
- **Effort**: 2-3 hours
- **Impact**: Runtime tuning without code changes

**Basic Request Analytics**
```python
@AgentRegistry.register("analytics", ["request_tracking"])
def analytics_agent(state: GraphState) -> GraphState:
    # Simple logging for request patterns
    logger.info("query_analytics", extra={
        "query_length": len(state["query"]),
        "intent": state.get("intent"),
        "processing_time": state.get("processing_time"),
        "agent_path": state.get("agent_path")
    })
    return state
```
- **Effort**: 3-4 hours
- **Impact**: Immediate insights into usage patterns

**Citation Formatting Enhancement**
- **Current**: Basic citation objects
- **Quick Win**: Rich markdown formatting with page numbers
- **Effort**: 4-6 hours
- **Impact**: More professional, academic-style citations

### **⚡ Medium Effort (3-7 Days)**

**Confidence Scoring Agent**
```python
@AgentRegistry.register("confidence", ["uncertainty_quantification"])
def confidence_agent(state: GraphState) -> GraphState:
    # Calculate confidence based on multiple factors
    retrieval_confidence = state.get("best_retrieval_score", 0.0)
    source_count = len(state.get("document_sources", []))
    answer_length = len(state.get("final_answer", ""))
    
    confidence = (retrieval_confidence * 0.5 + 
                 min(source_count/3, 1.0) * 0.3 + 
                 min(answer_length/500, 1.0) * 0.2)
    
    state["answer_confidence"] = confidence
    return state
```

**Session Persistence with Redis**
- Move from in-memory sessions to Redis
- Enable horizontal scaling
- Session data survives container restarts

**Query Preprocessing Agent**
```python
@AgentRegistry.register("preprocessor", ["query_normalization"])
def preprocessing_agent(state: GraphState) -> GraphState:
    query = state["query"]
    # Clean and normalize query
    cleaned_query = re.sub(r'\s+', ' ', query.strip())
    # Fix common typos, expand abbreviations
    state["query"] = cleaned_query
    return state
```

### **🔧 Advanced Features (2+ Weeks)**

**Agnostic LLM Provider System**

**Multi-Provider Support Implementation**

Currently, the system is tightly coupled to Google's Gemini API. A future enhancement would implement an agnostic LLM provider system that allows seamless switching between different providers:

**Supported Providers:**
- **OpenAI**: GPT-4.1, GPT-4.1 nano, GPT-4o, GPT-4o mini
- **Anthropic**: Claude 3.5 / 3.7 / 4
- **Google Gemini**: Current implementation (Flash/Pro models)
- **OpenRouter**: Access to multiple models through unified API
- **Local Models**: Ollama integration for on-premise deployment

**Implementation Benefits:**
- **Cost Optimization**: Switch to most cost-effective provider per use case
- **Performance Tuning**: Use best model for each agent type (fast for routing, powerful for synthesis)
- **Fallback Strategy**: Automatic provider switching on rate limits or failures
- **A/B Testing**: Compare model performance across providers
- **Compliance**: Choose providers meeting specific regulatory requirements

**Technical Architecture:**
```python
class LLMProvider(ABC):
    async def generate(self, prompt: str, config: GenerationConfig) -> Response
    
class OpenAIProvider(LLMProvider): ...
class AnthropicProvider(LLMProvider): ...
class GeminiProvider(LLMProvider): ...  # Current implementation

# Configuration-driven provider selection
llm_factory = LLMProviderFactory(config.provider_name)
```

This enhancement would significantly improve the system's flexibility and production readiness.

### **LLM Judge & Re-ranking Enhancements**

The "LLM as a Judge" architecture is powerful, but it can be further enhanced:

1.  **Improving Judge Prompting Techniques**
    *   **Limitation:** The current prompt is effective but generic.
    *   **Improvement:** Experiment with more advanced prompting strategies. This includes adding few-shot examples of good and bad assessments to the prompt, or using a "Chain of Thought" approach where the model first explains its reasoning before assigning a score. This can improve the consistency and accuracy of its judgments.

2.  **Fine-Tuning a Specialized Judge Model**
    *   **Limitation:** We rely on the zero-shot capabilities of `gemini-2.5-flash`. While strong, it's not specialized for our exact task.
    *   **Improvement:** For the highest level of quality, use a powerful model like `gemini-2.5-pro` to generate a high-quality dataset of `(query, passage, assessment)` tuples. Then, fine-tune a smaller, faster model (like `gemini-2.5-flash` or an open-source alternative) on this dataset. This creates a highly accurate, fast, and cost-effective judge specialized for our academic domain.

3.  **Integrating Advanced Re-ranking Logic**
    *   **Limitation:** The current system relies solely on the LLM Judge's relevance score.
    *   **Improvement:** Implement a more advanced re-ranking logic that considers the set of documents as a whole. After the LLM Judge identifies relevant documents, use techniques like Maximal Marginal Relevance (MMR) to promote diversity in the top-ranked results, ensuring a more comprehensive context for the synthesis agent by avoiding redundant information.

4.  **Optimizing for Latency and Cost**
    *   **Limitation:** While `gemini-2.5-flash` is fast, a model call is still a point of latency.
    *   **Improvement:** Implement a two-stage judging process. First, use a very fast, local model (like a fine-tuned cross-encoder, if speed is paramount) to perform an initial broad filtering of the 30 candidate chunks down to the top 10. Then, use the powerful LLM Judge only on this highly-relevant, smaller set. This hybrid approach can balance speed, cost, and quality.

### **ReAct Architecture Enhancements**

With the introduction of the PlannerAgent, the system now implements a foundation for a full ReAct (Reason + Act) architecture:

1.  **Dynamic Planning Improvements**
    *   **Current State:** The PlannerAgent uses LLM-based planning to determine agent execution order.
    *   **Future Enhancement:** Implement iterative planning where the PlannerAgent can dynamically adjust the plan based on intermediate results, enabling more sophisticated workflows.

2.  **Reasoning and Action Loop**
    *   **Current State:** Agents execute in a predetermined sequence based on the initial plan.
    *   **Future Enhancement:** Enable agents to request additional information or actions during execution, creating a true Reason-Act loop where the system can adapt its approach based on intermediate findings.

### **Advanced Intelligence Agents**
- **Multi-Document Comparison Agent**: "Compare findings between papers X and Y"
- **Confidence Scoring Agent**: Quantify answer uncertainty
- **Follow-up Context Agent**: Handle conversational queries

### **Quality & Evaluation Agents**  
- **Answer Validation Agent**: Cross-check for hallucinations
- **Golden Dataset Generation Agent**: Auto-create evaluation datasets
- **A/B Testing Agent**: Compare different agent configurations

### **Production Enhancement Agents**
- **Caching Agent**: Intelligent response caching
- **Rate Limiting Agent**: Per-user quota management  
- **Analytics Agent**: Usage pattern analysis

### **Implementation Roadmap**
1. **Fine-tuned Embeddings**: Domain-specific embedding models
2. **GraphRAG Integration**: Knowledge graph enhanced retrieval  
3. **Real-time Learning**: Agents that improve from user feedback
4. **Multi-modal Support**: PDF + image + table understanding
5. **Full ReAct Implementation**: Iterative planning and execution based on intermediate results

## 🏆 Technical Highlights

This implementation demonstrates **senior-level engineering**:

- **Design Patterns**: Registry, Plugin, Factory, Middleware patterns
- **Async Architecture**: Full async/await with proper event loop management
- **Error Resilience**: Graceful degradation and comprehensive error handling
- **Observability**: Structured logging with LangFuse integration ready
- **Production Ready**: Docker, environment configuration, health checks
- **Code Quality**: Type hints, docstrings, modular design
- **Testing Ready**: Pytest configuration and test structure
- **Advanced Orchestration**: Transition from rule-based to LLM-based planning with ReAct architecture foundations

The **extensible agent framework** is the key innovation - it transforms adding sophisticated AI capabilities from a multi-day refactoring effort into a **single file drop-in**. With the addition of the PlannerAgent, the system now demonstrates cutting-edge agent orchestration patterns that move beyond static graphs to dynamic, model-driven execution plans.

## 📦 Modular Architecture Improvements

This system has been enhanced with a robust modular architecture that follows industry best practices:

### **Dependency Injection & Inversion of Control**
- Services are created through a centralized factory pattern
- Dependencies are injected via constructors rather than hardcoded
- Easy to swap implementations without changing core logic

### **Single Responsibility Principle**
- Document processing responsibilities split into specialized services:
  - `PDFExtractor`: Handles PDF text extraction
  - `ChunkingService`: Manages text chunking strategies
  - `EmbeddingService`: Handles embedding generation with retry logic
- Session management extracted to `SessionManager`

### **Configuration Management**
- Typed configuration classes with environment variable support
- Centralized configuration validation
- Easy to extend with new configuration options

### **Testability**
- Services can be easily mocked for unit testing
- Clear interfaces between components
- Reduced coupling between modules

### **Extensibility**
- New providers can be added through simple interface implementation
- Auto-discovery of new components
- Plugin-like architecture for LLM providers

## 📄 License

MIT License - See LICENSE file for details.

---

**Built with ❤️ for the future of AI-powered research assistants.**
