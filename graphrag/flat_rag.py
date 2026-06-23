import re
import numpy as np
from typing import List, Dict, Tuple


class FlatRAG:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.embeddings = None

    def _chunk(self, content: str, sz: int = 400) -> List[str]:
        sents = re.split(r'(?<=[.!?])\s+', content)
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) < sz:
                cur += " " + s
            else:
                if cur.strip():
                    chunks.append(cur.strip())
                cur = s
        if cur.strip():
            chunks.append(cur.strip())
        return chunks

    def index(self, docs: List[Dict]):
        total = len(docs)
        for idx, d in enumerate(docs):
            for c in self._chunk(d["content"]):
                self.chunks.append({"text": c, "source": d["title"], "id": d["id"]})
            if (idx + 1) % 20 == 0 or idx == total - 1:
                print(f"  Chunking: {idx+1}/{total} docs, {len(self.chunks)} chunks", flush=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("  Fitting TF-IDF vectorizer...", flush=True)
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        self.embeddings = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
        print(f"  Indexed {len(self.chunks)} chunks (TF-IDF)", flush=True)

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        q = self.vectorizer.transform([query])
        scores = (self.embeddings @ q.T).toarray().flatten()
        top = np.argsort(scores)[-k:][::-1]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]
