**Project Blueprint: A State-of-the-Art Multi-Agent RAG System**

This document outlines the project summary and the detailed technical steps required to build a production-grade, multi-agent Retrieval-Augmented Generation (RAG) system. The goal is to create an intelligent "Chat with PDF" assistant capable of answering complex questions over a corpus of academic papers, handling ambiguity, and performing web searches when necessary.

**1. Project Summary & Core Objectives**

The project's primary objective is to develop a sophisticated backend system that demonstrates a senior-level understanding of modern AI architectures. The system will function as a multi-agent framework orchestrated by LangGraph, where specialized agents collaborate to provide accurate, context-aware answers.

**Key Features:**
*   **Advanced RAG Pipeline:** Go beyond basic RAG by implementing a multi-stage retrieval process including advanced chunking, hybrid search, and re-ranking to ensure the highest quality context is provided to the language model.
*   **Multi-Agent Orchestration:** Use LangGraph to build a robust, stateful system where a router agent intelligently delegates tasks to specialized worker agents for document retrieval, web search, and response synthesis.
*   **Intelligent LLM Routing:** Employ a portfolio of LLMs, using a fast, cost-effective model for routing and classification, and a powerful, frontier model for final answer generation, all managed through a unified API layer.
*   **Production-Ready Architecture:** The entire system will be built using FastAPI, containerized with Docker, and designed with observability and evaluation at its core to ensure reliability and maintainability.

**2. Core Technology Stack**

| Component | Technology | Justification |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.11+ | Industry standard for AI/ML development with a rich ecosystem of libraries. |
| **Application Server** | FastAPI | High-performance, asynchronous framework with automatic data validation and API documentation, ideal for production services.¹ |
| **Orchestration Engine** | LangGraph | Provides explicit, fine-grained control over agent workflows, enabling the creation of robust, stateful, and debuggable systems suitable for production.² |
| **Vector Database** | Weaviate | Open-source and production-ready, with native support for hybrid search (BM25 + vector), which is critical for robust retrieval.⁵ |
| **Embedding Model** | Gemini Embedding (gemini-embedding-exp) | Google's latest text embedding model optimized for measuring text relatedness, with generous free tier limits and excellent retrieval performance. |
| **LLM Provider** | Google Gemini API | Direct access to Google's state-of-the-art models with generous free quotas: 2.5 Pro for complex reasoning, 2.5 Flash for high-volume routing tasks, offering superior price-performance compared to OpenAI alternatives. |
| **Web Search API** | Tavily Search API | A search API purpose-built for RAG, which handles searching, scraping, and content consolidation in a single call, reducing complexity and latency.⁸ |
| **Evaluation Framework** | RAGAs | An open-source framework for the automated evaluation of RAG pipelines using key metrics like faithfulness, answer relevance, and context precision/recall.¹¹ |
| **Observability** | LangFuse | Provides end-to-end tracing and debugging for LLM applications, offering deep visibility into agent behavior, costs, and performance bottlenecks.¹⁴ |
| **Deployment** | Docker & Docker Compose | Ensures a consistent, reproducible environment for local development and prepares the application for production deployment, as required by the assignment.¹ |

**3. Detailed Technical Implementation Steps**

This project will be executed in a series of structured steps, moving from foundational setup to advanced features and evaluation.

**Step 1: Environment Setup and Project Scaffolding**
1.  **Initialize Git Repository:** Create a new Git repository to track all project code and documentation.
2.  **Set Up Python Environment:** Use a virtual environment (e.g., venv) to manage project dependencies. Install initial libraries: fastapi, uvicorn, langgraph, langchain, python-dotenv, docker.
3.  **Structure Project Directory:** Organize the codebase logically with directories for the application (app/), agents (app/agents/), core logic (app/core/), and utilities (app/utils/).
4.  **Configure Docker:**
    *   Create a Dockerfile for the FastAPI application.
    *   Create a docker-compose.yml file to orchestrate the FastAPI service and a Weaviate database container.¹
5.  **Manage Configuration:** Use a .env file to store API keys (Google Gemini, Tavily, LangFuse) and other configuration variables, separating configuration from code.¹⁵

**Step 2: API-First Document Ingestion Pipeline**
1.  **FastAPI Document Endpoints:** Implement production-ready API endpoints for document management:
    *   `POST /api/v1/documents` - Upload and process PDF files with real-time ingestion
    *   `GET /api/v1/documents` - List all ingested document IDs  
    *   `GET /api/v1/documents/stats` - Database statistics and metrics
    *   `DELETE /api/v1/documents` - Clear all documents from database
2.  **Parent Document Retriever Strategy:** This advanced chunking method balances precision and context.¹⁶
    *   Split documents into large, contextually whole "parent" chunks (~1800 chars).
    *   Further divide these parent chunks into smaller, semantically precise "child" chunks (~500 chars).
3.  **Generate Embeddings:** Using the Gemini Embedding model (gemini-embedding-exp), generate vector embeddings for the **child chunks only.**
4.  **Store Data:**
    *   Store the child chunk embeddings and their metadata (including a pointer to their parent) in the Weaviate vector database.
    *   Store the full text of the parent chunks in a simple in-memory key-value store for rapid context retrieval.
5.  **Admin Utilities:** Provide `scripts/ingest_admin.py` for batch operations, maintenance tasks, and direct database access when needed.

**Step 3: Multi-Agent System Implementation with LangGraph**
1.  **Define Graph State:** Create a TypedDict to serve as the GraphState, which will manage the flow of data (e.g., question, conversation\_history, retrieved\_context, final\_answer) between nodes.⁴
2.  **Build Agent Nodes:** Implement each agent as a distinct function (a node in the graph).
    *   **Query Analysis & Routing Agent:** This entry-point agent will use a fast, cost-effective LLM (Gemini 2.5 Flash) to classify the user's intent and determine the next step (e.g., retrieve\_corpus, search\_web, clarify, end).²¹
    *   **Corpus Retrieval Agent:** This agent will execute the full RAG pipeline:
        *   **Query Expansion (HyDE):** Use an LLM to generate a hypothetical document from the user's query to create a more effective embedding.²³
        *   **Hybrid Search:** Query Weaviate using its native hybrid search to retrieve a large set of candidate child chunks.
        *   **Re-ranking:** Fetch the parent chunks and use a cross-encoder model (e.g., bge-reranker-large) to re-rank them for precision, selecting only the top 3-5 to pass as context.²⁸
    *   **Web Search Agent:**
        *   Use an LLM to transform the user's query into an optimized search term.³⁶
        *   Call the Tavily Search API to get a clean, RAG-optimized summary of web results.
    *   **Synthesis & Response Agent:** Use a powerful LLM (Gemini 2.5 Pro) to synthesize the final answer from the retrieved context.
3.  **Construct the Graph:**
    *   Instantiate a StatefulGraph.
    *   Add the agent functions as nodes.
    *   Define the conditional edges that connect the nodes based on the output of the routing agent.
    *   Compile the graph into a runnable application.

**Step 4: API Development and Service Integration**
1.  **Create FastAPI Endpoints:**
    *   Implement an async `POST /api/v1/ask` endpoint that accepts a session\_id and query.
    *   Implement a `POST /api/v1/clear-memory` endpoint to reset a user's session.
    *   Add health check endpoint at `/health` for monitoring and load balancing.
2.  **Integrate LangGraph:** Wire the compiled LangGraph application into the `/ask` endpoint, managing conversation state based on the session\_id.
3.  **Implement Streaming:** Use FastAPI's StreamingResponse to stream the final answer token-by-token and, optionally, intermediate agent thoughts for an enhanced user experience.
4.  **Error Handling & Validation:** Implement comprehensive error handling with proper HTTP status codes and Pydantic models for request/response validation.

**Step 5: Evaluation and Observability**
1.  **Integrate LangFuse:** Set the required environment variables to automatically trace every execution of the LangFuse application. This will provide deep visibility into agent steps, LLM inputs/outputs, and performance metrics.¹⁴
2.  **Build the Evaluation Framework:**
    *   **Generate a Golden Dataset:** Create a script that uses a powerful LLM to read the source PDFs and synthetically generate a high-quality dataset of question-answer pairs.⁴²
    *   **Automate Metrics with RAGAs:** Use the RAGAs library to run an evaluation pipeline on the golden dataset.
    *   **Track Key Metrics:** Measure and record the "RAG Triad": **Context Precision/Recall, Faithfulness,** and **Answer Relevance** to establish a quantitative benchmark of system performance.⁴⁷

**Step 6: Final Documentation and Submission**
1.  **Write the README.md:** Create a comprehensive README file that includes:
    *   An overview of the system architecture and a description of each agent's role.
    *   Clear, step-by-step instructions on how to set up the environment and run the application using docker-compose up.
    *   API documentation with example requests and responses.
    *   A section detailing potential future improvements, such as fine-tuning the embedding model, implementing GraphRAG, or building an A/B testing framework.
2.  **Final Review:** Ensure the code is clean, well-commented, and the Git repository is organized and ready for submission.

***
