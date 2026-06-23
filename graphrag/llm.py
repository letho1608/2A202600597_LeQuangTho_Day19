import os
import re
import json
from typing import Optional, List, Dict, Any

from .config import get_config


def get_llm():
    cfg = get_config()
    if not cfg["api_key"] and cfg["provider"] != "ollama":
        print(f"  [WARN] LLM_API_KEY not set for '{cfg['provider']}'. Using rule-based.")
        return None

    kw = dict(temperature=cfg["temperature"],
              max_tokens=cfg["max_tokens"])

    if cfg["provider"] == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], **kw)

    elif cfg["provider"] == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=cfg["api_key"], **kw)

    elif cfg["provider"] == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=cfg["api_key"],
                          base_url=cfg["base_url"] or "https://api.deepseek.com", **kw)

    elif cfg["provider"] == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(api_key=cfg["api_key"], **kw)

    elif cfg["provider"] == "ollama":
        from langchain_openai import ChatOpenAI
        base = (cfg["base_url"] or "http://localhost:11434").rstrip("/") + "/v1/"
        ollama_kw = {k: v for k, v in kw.items() if k != "api_key"}
        ollama_kw["timeout"] = 300
        return ChatOpenAI(model=cfg["model"], api_key="ollama", base_url=base, **ollama_kw)

    elif cfg["provider"] == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(api_key=cfg["api_key"], **kw)

    elif cfg["provider"] == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(api_key=cfg["api_key"], **kw)

    else:
        print(f"  [WARN] Unknown provider '{cfg['provider']}'. Using rule-based.")
        return None


BATCH_PROMPT = """Extract entity-relation triples from MULTIPLE documents.
Each document is separated by "---DOC id=N---".

For each document, output a JSON object:
{{"doc_id": N, "triples": [{{"head": "Entity1", "relation": "REL", "tail": "Entity2"}}, ...]}}

Rules:
- Relation types: FOUNDED_BY, LOCATED_IN, PRODUCES, COMPETES_WITH, PARTNERS_WITH, INVESTED_IN, ACQUIRED, REGULATES, HAS_GOAL, RELATED_TO, PUBLISHED, AUTHORED, EMPLOYS
- Use consistent entity names (e.g. always "United States", never "US")
- Skip vague entities, focus on companies, organizations, people, locations, technologies
- Output ONLY valid JSON array. No other text.

Documents:
{docs_text}

JSON:"""


def extract_triples_llm(llm, docs: List[Dict], batch_size: int = 3) -> List[Dict]:
    if llm is None:
        return []

    all_triples = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        docs_text = ""
        for doc in batch:
            docs_text += f"\n---DOC id={doc['id']}---\n{doc['content'][:1000]}\n"

        prompt = BATCH_PROMPT.format(docs_text=docs_text)
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = _parse_json(content)
            for item in parsed:
                did = item.get("doc_id")
                for t in item.get("triples", []):
                    t["doc_id"] = did
                    all_triples.append(t)
        except Exception as e:
            print(f"    [LLM Error] batch {i}: {e}")

        pct = min(100, int((i + batch_size) / len(docs) * 100))
        print(f"    LLM extraction: {pct}% ({i + len(batch)}/{len(docs)} docs)")
    return all_triples


QA_PROMPT = """You are a helpful assistant answering questions about the US electric vehicle sector.

CONTEXT:
{context}

QUESTION: {question}

Answer concisely based ONLY on the context. If the context lacks information, say 'Khong du thong tin trong co so tri thuc.'"""


def answer_question(llm, question: str, context: str) -> str:
    if llm is None:
        return "LLM not configured."
    prompt = QA_PROMPT.format(context=context[:8000], question=question)
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"[LLM Error] {e}"


def _parse_json(content: str) -> List[Dict]:
    content = re.sub(r'```(?:json)?', '', content).strip()
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    results = []
    for line in content.split("\n"):
        try:
            obj = json.loads(line.strip().rstrip(","))
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results
