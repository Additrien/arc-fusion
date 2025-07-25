### 1. Core RAG Pipeline Enhancements (Improving Answer Quality)

These improvements focus on making the core retrieval and generation process more robust and accurate.

*   **Implement a Re-ranking Stage:** The current system retrieves chunks and passes them directly to the synthesis agent. A significant improvement would be to add a re-ranking step.
    *   **How:** After the initial hybrid search retrieves a large set (`k=50`) of candidate chunks, use a more computationally expensive but highly accurate **Cross-Encoder model** (e.g., from the `sentence-transformers` library) to re-rank the top candidates (`top_n=10`). This ensures the final context provided to the LLM is of the highest possible relevance, drastically reducing the impact of "distractor" chunks.
    *   **Why it shows expertise:** It demonstrates an understanding of the trade-off between retrieval speed (vector search) and accuracy (cross-encoders) and how to build a multi-stage retrieval pipeline.

*   **Introduce Advanced Fusion for Merged Results:** When the system gets context from both the corpus and the web, it simply concatenates them. A more sophisticated approach is to merge and rank the combined results.
    *   **How:** Use **Reciprocal Rank Fusion (RRF)** to combine the ranked lists of documents from the corpus search and the web search into a single, more robustly ranked list before passing the top results to the synthesis agent.
    *   **Why it shows expertise:** It shows you can handle and intelligently merge results from heterogeneous data sources, which is a common real-world problem.

*   **Advanced Query Transformation:** Your `HyDE` implementation is a great start. To showcase deeper expertise, you can implement more advanced query understanding techniques.
    *   **How:**
        1.  **Query Decomposition:** For a question like "Compare the RAG methodology in paper X with the approach from paper Y", the system could break it down into two sub-queries: "What is the RAG methodology in paper X?" and "What is the approach from paper Y?".
        2.  **Step-Back Prompting:** For a specific query, the agent could first generate a more general, "step-back" question to retrieve broader context before retrieving documents for the specific query itself.
    *   **Why it shows expertise:** This demonstrates a proactive approach to handling complex user intent and shows you can design agents that reason about the query itself before acting.

---

### 2. Agent & Orchestration Intelligence (Making the System Smarter)

This is where you directly address the "multi-agent architecture" requirement at a senior level.

*   **Implement the `ClarificationAgent` (Bonus Point):** The assignment explicitly mentions this.
    *   **How:** Create a new agent that is triggered when the `routing_agent` outputs a `clarify` intent. This agent would analyze the query's ambiguity and respond to the user with a question asking for the necessary missing information (e.g., "When you say 'best method,' are you referring to accuracy, latency, or cost? For which dataset?").
    *   **Why it shows expertise:** It directly fulfills a bonus requirement and makes the system's conversational flow far more robust and user-friendly.

*   **Introduce a Self-Correction / Critique Agent:** This is a hallmark of an advanced, reliable agent system.
    *   **How:** After the `synthesis_agent` generates an answer, it doesn't immediately go to the user. Instead, it passes the `(question, context, answer)` tuple to a new `CritiqueAgent`. This agent's job is to check the answer against the provided context for factual consistency (hallucinations), completeness, and relevance. If the critique agent finds a flaw, it can send the result *back* to the synthesis agent with a critique note (e.g., "The answer failed to mention the key result from Document 2. Please revise.").
    *   **Why it shows expertise:** This is a highly advanced concept that demonstrates your ability to build self-correcting, reliable AI systems—a critical skill for production AI.

*   **Transition from Orchestrator to an LLM-based Planner:** Your current orchestrator is rule-based. The next evolution is to use an LLM for planning.
    *   **How:** Instead of hard-coding the `corpus_retrieval -> web_search -> synthesis` logic, you could have a `PlannerAgent`. This agent would receive the user's query and decide which "tools" (your other agents) to call, in what order, to satisfy the request. This is the foundation of a ReAct (Reason + Act) architecture.
    *   **Why it shows expertise:** It shows you're on the cutting edge of agent design, moving from static graphs to dynamic, model-driven execution plans.

---

### 3. Production Readiness & Scalability (Building for the Real World)

This is what truly separates a senior engineer from others. You're not just building a demo; you're building a service.

*   **Asynchronous Document Ingestion:** The `/upload` endpoint currently blocks until processing and embedding are complete. For large PDFs, this will time out.
    *   **How:** Convert the ingestion process to be fully asynchronous. The `/upload` endpoint should accept the file, place it in a queue (like **Celery** with a **Redis** or **RabbitMQ** broker), and immediately return a `task_id` to the user. A separate `/status/{task_id}` endpoint would allow the user to poll for the processing status.
    *   **Why it shows expertise:** This is standard practice for any long-running task in a production backend and demonstrates your ability to design robust, non-blocking APIs.

*   **Decouple State Management:** The current session memory (`sessions`) and parent chunk store (`parent_store`) are in-memory dictionaries. This will not work with multiple server instances (e.g., behind a load balancer) and state is lost on restart.
    *   **How:** Move the session memory and the parent chunk store to a persistent, external service like **Redis**. Redis is perfect for this kind of fast key-value access.
    *   **Why it shows expertise:** It shows you are thinking about scalability, state persistence, and building a stateless application layer, which is a fundamental principle of modern backend design.

*   **API Streaming for Responses:** The `/ask` endpoint waits for the full answer before returning. The user perceives this as high latency.
    *   **How:** Modify the `synthesis_agent` and the FastAPI endpoint to stream the response token-by-token using **Server-Sent Events (SSE)**. The user would start seeing the beginning of the answer almost immediately.
    *   **Why it shows expertise:** It demonstrates a focus on user experience and knowledge of modern API techniques for handling generative models.

---

### 4. Evaluation & System Reliability (Proving It Works)

Finally, to be in the top 1%, you must be able to prove your system's quality objectively.

*   **Implement a RAG Evaluation Framework (Bonus Point):** The assignment asks for a basic evaluation system.
    *   **How:**
        1.  Create a small "golden dataset" of 10-15 question/answer pairs based on the provided PDFs.
        2.  Build an evaluation script that runs these questions through your system.
        3.  Use a framework like **RAGAs** or a custom script to measure the core metrics of RAG:
            *   **Context Precision & Context Recall:** Is the retrieved context relevant?
            *   **Faithfulness:** Does the answer stay true to the context (i.e., not hallucinate)?
