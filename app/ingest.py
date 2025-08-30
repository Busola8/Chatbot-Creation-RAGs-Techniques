import os
from pathlib import Path
import json
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from loguru import logger
import yaml
import math
from pypdf import PdfReader

# Loads config
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

EMBED_MODEL = cfg.get('embed_model', 'sentence-transformers/all-MiniLM-L6-v2')
CHUNK_SIZE = cfg.get('chunk_size', 500)
CHUNK_OVERLAP = cfg.get('chunk_overlap', 100)
DATA_DIR = cfg.get('data_dir', 'data')
FAISS_PATH = cfg.get('faiss_index_path', './embeddings/faiss_index.bin')

def read_documents(folder: str):
    docs = []
    p = Path(folder)
    for file in p.glob('**/*'):
        if file.suffix.lower() in ['.txt', '.md']:
            text = file.read_text(encoding='utf-8')
            docs.append({'source': str(file), 'text': text})
        elif file.suffix.lower() == '.pdf':
            try:
                reader = PdfReader(str(file))
                text = '\n'.join([p.extract_text() or '' for p in reader.pages])
                docs.append({'source': str(file), 'text': text})
            except Exception as e:
                logger.error(f'Failed to read PDF {file}: {e}')
    return docs

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

def build_faiss(docs: List[dict], index_path: str = FAISS_PATH):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    all_texts = []
    metadatas = []
    for d in docs:
        for c in chunk_text(d['text']):
            all_texts.append(c)
            metadatas.append({'source': d['source']})
    emb = embedder.encode(all_texts, show_progress_bar=True)
    import numpy as np
    emb = np.array(emb).astype('float32')
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb)
    index.add(emb)
    faiss.write_index(index, index_path)
    # save metadata
    import pickle
    with open(index_path + '.meta', 'wb') as f:
        pickle.dump(all_texts, f)
    logger.success(f'Built FAISS index with {len(all_texts)} chunks.')

if __name__ == '__main__':
    docs = read_documents(DATA_DIR)
    if not docs:
        print('No documents found in', DATA_DIR)
    else:
        build_faiss(docs)
