# Full RAG Chatbot Project

This repository contains a full RAG chatbot with:
- Data ingestion (PDF/TXT), chunking, and FAISS index building
- SentenceTransformer embeddings (configurable)
- Retriever + QA + generator pipelines
- FastAPI backend and Streamlit UI
- Dockerfile & Kubernetes manifests
- CI workflow (GitHub Actions)
- Tests and examples

## Quickstart (local)
1. Create virtualenv:
   python -m venv .venv
   .venv\Scripts\activate  # Windows
2. Install requirements:
   pip install -r requirements.txt
3. Ingest docs:
   python app/ingest.py
4. Run server:
   uvicorn app.main:app --reload
5. UI:
   streamlit run streamlit_app.py

## Config
Edit `config.yaml` to adjust model names, chunk size, vector store, etc.

