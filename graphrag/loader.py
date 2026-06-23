import os
import re
import glob
from typing import List, Dict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")


def load_docs(data_dir: str = DATA_DIR) -> List[Dict]:
    docs = []
    pattern = os.path.join(data_dir, "doc_*.txt")
    for fpath in sorted(glob.glob(pattern),
                        key=lambda x: int(re.search(r'doc_(\d+)', x).group(1))):
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
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
    return docs
