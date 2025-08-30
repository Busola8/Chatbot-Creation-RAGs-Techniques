import os, yaml, pickle, numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
from loguru import logger

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

EMBED_MODEL = cfg.get('embed_model')
GEN_MODEL = cfg.get('generator_model')
QA_MODEL = cfg.get('qa_model')
FAISS_PATH = cfg.get('faiss_index_path')

class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.generator = pipeline('text2text-generation', model=GEN_MODEL)
        self.qa = pipeline('question-answering', model=QA_MODEL)
        self.index = None
        self.texts = None
        if os.path.exists(FAISS_PATH):
            self._load_index()

    def _load_index(self):
        logger.info('Loading FAISS index...')
        self.index = faiss.read_index(FAISS_PATH)
        with open(FAISS_PATH + '.meta', 'rb') as f:
            self.texts = pickle.load(f)
        logger.success('Loaded index.')

    def retrieve(self, query, top_k=5):
        q_emb = self.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        hits = []
        for i, idx in enumerate(I[0]):
            hits.append({'text': self.texts[idx], 'score': float(D[0][i])})
        return hits

    def answer(self, query, top_k=5):
        hits = self.retrieve(query, top_k=top_k)
        context = '\n\n'.join([h['text'] for h in hits])
        # extractive answer
        try:
            extract = self.qa(question=query, context=context)
        except Exception as e:
            extract = {'error': str(e)}
        # generative answer
        prompt = f"""context: {context}\n\nquestion: {query}\nanswer:"""
        gen = self.generator(prompt, max_new_tokens=128, do_sample=False)[0]['generated_text']
        return {'query': query, 'hits': hits, 'extractive': extract, 'generative': gen}
