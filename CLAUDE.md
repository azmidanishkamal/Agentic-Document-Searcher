# Agentic-Document-Searcher

An agentic RAG (retrieval-augmented generation) system that benchmarks Pinecone
vs. Weaviate for retrieval quality. The project pairs a LangGraph/LangChain
agent with an evaluation harness that runs a golden Q&A dataset against both
vector stores to compare retrieval performance.

## Layout

- `src/ingestion/` — document loading, chunking, and indexing into vector stores
- `src/retrieval/` — retriever implementations for Pinecone and Weaviate
- `src/agent/` — the LangGraph agent that orchestrates retrieval and generation
- `src/eval/` — evaluation harness for scoring retrieval/answer quality
- `src/mcp_server/` — MCP server exposing the agent's tools
- `tests/` — unit and integration tests
- `eval_data/` — golden Q&A dataset used by the evaluation harness
