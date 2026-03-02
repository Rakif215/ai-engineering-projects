# Project 1: Production-Grade RAG - Developer Log & Challenges

When discussing this project in interviews or on my profile, these are the key technical challenges I overcame to bring this system from a basic tutorial script to a production-ready application.

## 1. Navigating the Evolving LangChain Ecosystem
**The Challenge:** LangChain moves incredibly fast. During development, I encountered multiple `ModuleNotFoundError` issues because core components like `EnsembleRetriever` and `ContextualCompressionRetriever` were relocated across `langchain`, `langchain_community`, and `langchain_classic` packages in recent versions (0.2/0.3).
**The Solution:** I meticulously audited the package dependencies, explicitly managing versions in `requirements.txt` (`langchain-core`, `langchain-community`, `langchain-chroma`, etc.) rather than relying on the monolithic `langchain` wrapper. I mapped the deprecated imports to their new homes (e.g., `langchain_classic.retrievers.ensemble`) to ensure the pipeline was stable and resilient to upstream changes.

## 2. Hallucination Mitigation (The "I don't know" Problem)
**The Challenge:** Early iterations of the RAG system would attempt to guess answers when the retrieved context was sparse or irrelevant, which is unacceptable in enterprise environments.
**The Solution:** I implemented strict **Citation Enforcement** within the Prompt Management configuration (`config.yaml`). I engineered the system prompt to explicitly instruct Gemini/Groq to decline answering ("I cannot answer this question because...") if grounding was impossible, drastically improving reliability.

## 3. Optimizing Retrieval Precision vs. Recall
**The Challenge:** Standard semantic vector search (ChromaDB) sometimes missed keyword-heavy queries (e.g., specific acronyms or ID numbers), leading to poor recall.
**The Solution:** I implemented a **Hybrid Retrieval Strategy**. By combining BM25 (keyword search) with Semantic Search via an `EnsembleRetriever`, I maximized recall. To prevent the LLM from being overwhelmed by the resulting larger context window, I introduced a **Cohere Cross-Encoder Reranker** (`ContextualCompressionRetriever`) to rescore and compress the results, ensuring only the top 3 most relevant chunks made it to the generation phase.

## 4. Evaluating "Faithfulness" Automatically
**The Challenge:** Manually verifying if the LLM's answers were hallucinated is not scalable for CI/CD pipelines.
**The Solution:** I integrated the **RAGAS** evaluation framework. By curating a "Golden Dataset" and wiring RAGAS into a GitHub Actions workflow, I created an automated gating mechanism. The pipeline programmatically measures "Faithfulness" (claims supported by context) and will fail the build if the score drops below an 85% threshold, ensuring quality regressions never reach production.
