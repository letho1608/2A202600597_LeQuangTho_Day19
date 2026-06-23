import os
import re
import glob
from typing import List, Dict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")


def load_docs(data_dir: str = DATA_DIR) -> List[Dict]:
    docs = []
    pattern = os.path.join(data_dir, "doc_*.txt")
    all_files = sorted(glob.glob(pattern),
                       key=lambda x: int(re.search(r'doc_(\d+)', x).group(1)))
    total = len(all_files)
    for idx, fpath in enumerate(all_files):
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        lines = text.split("\n")
        doc = {
            "id": int(re.search(r'doc_(\d+)', os.path.basename(fpath)).group(1)),
            "query": "", "title": "", "link": "", "snippet": "", "content": ""
        }
        state = "header"
        cls = []
        for line in lines:
            if line.startswith("Query:"):
                doc["query"] = line[6:].strip()
            elif line.startswith("Title:"):
                doc["title"] = line[6:].strip()
            elif line.startswith("Link:"):
                doc["link"] = line[5:].strip()
            elif line.startswith("Snippet:"):
                doc["snippet"] = line[8:].strip()
            elif line.startswith("Full Content:"):
                state = "content"
            elif state == "content":
                cls.append(line)
        doc["content"] = "\n".join(cls).strip()
        docs.append(doc)
        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  Loading: {idx+1}/{total} docs", flush=True)
    return docs
