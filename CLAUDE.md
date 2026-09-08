# Agentic-Document-Searcher

## Project

An agentic RAG system that retrieves and answers questions over a document
corpus, built to demonstrate production-grade RAG engineering skills for
AI/ML engineering job applications.

## Goal

Benchmark retrieval quality, latency, and cost between Pinecone and Weaviate
as vector DB backends, and build a full evaluation harness (RAGAS + DeepEval)
with a golden Q&A dataset, wired into CI.

## Stack

- Python 3.11+
- LangGraph for agent orchestration
- LangChain for retrieval utilities
- Pinecone and Weaviate as vector DB backends
- RAGAS / DeepEval for evaluation
- GitHub Actions for CI
- MCP for exposing the system as a tool server

## Architecture

- `src/ingestion/` — chunking, embedding, and loading into both DBs
- `src/retrieval/` — hybrid search + reranking, with a swappable backend
- `src/agent/` — the LangGraph agent that decides when/what to retrieve
- `src/eval/` — the evaluation harness
- `src/mcp_server/` — exposes retrieval + eval as MCP tools

## Conventions

Standard Python conventions, type hints everywhere, ruff for linting/formatting,
pytest for testing.

## Current milestone

M1 — ingestion pipeline (not yet started).

## Corpus

Public financial filings (10-Ks and earnings reports). The initial dataset
will be SEC filings pulled via EDGAR's public API/full-text search, likely
starting with a handful of companies across a couple of sectors to keep the
eval set manageable.
