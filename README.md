# LAB DAY 19: GraphRAG - Hệ thống Knowledge Graph cho ngành Xe Điện Mỹ

> **MSSV**: 2A202600597  
> **Họ tên**: Lê Quang Thọ  
> **Ngày**: 23/06/2026

---

## Mục lục

- [1. Tổng quan](#1-tổng-quan)
- [2. Yêu cầu bài toán](#2-yêu-cầu-bài-toán)
- [3. Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
- [4. Cài đặt](#4-cài-đặt)
- [5. Kiến trúc hệ thống](#5-kiến-trúc-hệ-thống)
- [6. Entity Extraction & Relation Extraction](#6-entity-extraction--relation-extraction)
- [7. Graph Construction & Deduplication](#7-graph-construction--deduplication)
- [8. Query Answering: BFS vs Vector Search](#8-query-answering-bfs-vs-vector-search)
- [9. Flat RAG vs GraphRAG](#9-flat-rag-vs-graphrag)
- [10. Kết quả Benchmark](#10-kết-quả-benchmark)
- [11. Phân tích chi phí](#11-phân-tích-chi-phí)
- [12. Hướng dẫn sử dụng](#12-hướng-dẫn-sử-dụng)

---

## 1. Tổng quan

Xây dựng hệ thống **GraphRAG** (Knowledge Graph + Retrieval-Augmented Generation) trích xuất tri thức từ 70 bài báo về ngành xe điện (EV) tại Mỹ. Hệ thống hỗ trợ đa nhà cung cấp LLM và so sánh giữa phương pháp Flat RAG và GraphRAG.

## 2. Yêu cầu bài toán

- **Entity Extraction**: Trích xuất thực thể (công ty, tổ chức, con người, địa điểm, công nghệ) từ văn bản thô
- **Relation Extraction**: Xác định quan hệ giữa các thực thể
- **Graph Construction**: Xây dựng đồ thị tri thức với NetworkX
- **Multi-hop Querying**: Truy vấn đa bước trên đồ thị (1-hop, 2-hop traversal)
- **Evaluation**: So sánh Flat RAG vs GraphRAG trên 20 câu hỏi benchmark

## 3. Cấu trúc thư mục

```
├── main.py                          # Entry point
├── .env                             # Cấu hình LLM provider
├── .env.example                     # Template .env
├── .gitignore
├── README.md                        # Tài liệu hướng dẫn
├── dataset/                         # 70 documents về EV
│   ├── doc_1.txt
│   └── ...
├── graphrag/                        # Module chính
│   ├── __init__.py
│   ├── __main__.py                   # python -m graphrag
│   ├── config.py                     # Đọc .env
│   ├── loader.py                     # Load documents
│   ├── entities.py                   # Định nghĩa entity + rule extraction
│   ├── llm.py                        # LLM client đa provider
│   ├── graph.py                      # Xây dựng đồ thị NetworkX
│   ├── flat_rag.py                   # TF-IDF vector search
│   └── pipeline.py                   # Pipeline chính
└── submit/                          # Bài nộp
    ├── knowledge_graph.png           # Ảnh đồ thị tri thức
    ├── comparison_results.csv        # Bảng so sánh 20 câu benchmark
    ├── BaoCaoPhanTichChiPhi.md       # Phân tích chi phí
    └── task.docx                     # Lab instructions
```

## 4. Cài đặt

### 4.1. Yêu cầu

- Python 3.10+
- Ollama (cho local inference) hoặc API key từ các provider

### 4.2. Cài đặt packages

```bash
pip install networkx matplotlib numpy scikit-learn pandas python-dotenv
pip install langchain-openai langchain-groq langchain-nvidia-ai-endpoints
pip install langchain-anthropic langchain-google-genai
```

### 4.3. Cấu hình .env

```ini
# Ollama (local)
LLM_PROVIDER=ollama
LLM_MODEL=llama3
LLM_BASE_URL=http://localhost:11434

# Hoặc OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
```

### 4.4. Chạy

```bash
python main.py
# hoặc
python -m graphrag
```

## 5. Kiến trúc hệ thống

```
Dataset (70 docs)
       │
       ▼
┌─────────────────┐
│   Document       │
│   Loader         │───► Raw text chunks
└─────────────────┘
       │
       ▼
┌─────────────────┐     ┌──────────────────┐
│  Entity +        │────►│  LLM (OpenAI/    │
│  Relation Extract│     │  Groq/Ollama...) │
│  (Rule hoặc LLM) │     └──────────────────┘
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Knowledge Graph │───► NetworkX MultiDiGraph
│  (57-71 nodes)   │     57-71 nodes, 400-800 edges
└─────────────────┘
       │
       ├────────────────────────────────┐
       ▼                                ▼
┌─────────────────┐           ┌──────────────────┐
│  GraphRAG        │           │  Flat RAG         │
│  2-hop traversal │           │  TF-IDF Search    │
│  qua edges       │           │  trên chunks      │
└─────────────────┘           └──────────────────┘
       │                                │
       └────────────┬───────────────────┘
                    ▼
          ┌──────────────────┐
          │  LLM Answer Gen   │
          │  (optional)       │
          └──────────────────┘
```

## 6. Entity Extraction & Relation Extraction

### 6.1. LLM phân biệt Entity vs Attribute như thế nào?

LLM được hướng dẫn qua prompt engineering:

- **Entity** = thực thể độc lập, có thể làm chủ ngữ hoặc tân ngữ trong quan hệ  
  *Ví dụ: "Tesla", "California", "Lithium-ion battery"*
- **Attribute** = thuộc tính mô tả entity, không tồn tại độc lập  
  *Ví dụ: "51.3% market share" là attribute của Tesla, "7.3% EV share" là attribute của US market*

Prompt yêu cầu LLM xuất JSON với cấu trúc `(head, relation, tail)` và chỉ chấp nhận các entity có ý nghĩa trong miền EV.

### 6.2. Rule-based Extraction

Khi không có LLM, hệ thống dùng rule-based:

- **Entity**: Dictionary 57 entity cố định (26 công ty, 10 tổ chức, 6 người, 15 địa điểm, 11 công nghệ, 4 metric)
- **Relation**: Co-occurrence trong cùng câu → `RELATED_TO` + pattern matching (`FOUNDED_IN`, `LOCATED_IN`)
- **Ưu điểm**: Nhanh, không tốn phí
- **Nhược điểm**: Chỉ bắt được entity đã biết trước

## 7. Graph Construction & Deduplication

### 7.1. Tại sao Deduplication quan trọng?

1. **Giảm nhiễu**: Cùng một thực thể xuất hiện trong nhiều document sinh ra vô số cạnh `RELATED_TO` trùng lặp. Dedup giảm từ 40,000+ raw relations xuống ~400-800 edges.

   | Trước Dedup | Sau Dedup |
   |-------------|-----------|
   | 40,474 raw  | 787 edges |

2. **Tăng độ chính xác**: Loại bỏ quan hệ trùng lặp, giữ lại duy nhất một cạnh với đúng loại quan hệ.

3. **Hiệu năng truy vấn**: Đồ thị nhỏ hơn → traversal nhanh hơn.

### 7.2. Đồ thị tri thức

 **57-71 nodes, 400-800 edges** (tùy extraction method)

| Loại | Số lượng | Màu |
|------|----------|-----|
| Công ty (COMPANY) | 26 | Đỏ |
| Tổ chức (ORGANIZATION) | 8-10 | Xanh ngọc |
| Con người (PERSON) | 5-6 | Vàng |
| Địa điểm (LOCATION) | 6-15 | Xanh lá |
| Công nghệ (TECHNOLOGY) | 11-12 | Hồng |

**Top entities (by degree)**:

| Entity | Loại | Degree |
|--------|------|--------|
| China | LOCATION | 82 |
| Europe | LOCATION | 76 |
| United States | LOCATION | 68 |
| Tesla | COMPANY | 47 |
| Hyundai | COMPANY | 44 |
| Volkswagen | COMPANY | 42 |

## 8. Query Answering: BFS vs Vector Search

### Khác biệt chính

| Tiêu chí | BFS / Graph Traversal | Vector Search (Flat RAG) |
|----------|----------------------|------------------------|
| **Cách hoạt động** | Duyệt cấu trúc đồ thị theo edges | Tìm kiếm cosine similarity trong embedding space |
| **Dữ liệu đầu vào** | Quan hệ giữa các node | Chunks văn bản |
| **Độ chính xác quan hệ** | Cao (biết chính xác kiểu quan hệ) | Thấp (không phân biệt được quan hệ) |
| **Xử lý multi-hop** | Tốt (2-hop, 3-hop traversal) | Kém (chỉ similar, không suy luận) |
| **Ví dụ** | "Tesla → FOUNDED_BY → Elon Musk" | "Tesla sản xuất xe điện..." |
| **Điểm mạnh** | Truy vấn cấu trúc, suy luận đa bước | Tìm kiếm ngữ nghĩa linh hoạt |
| **Điểm yếu** | Cần đồ thị đầy đủ | Dễ bị ảo giác (hallucination) |

## 9. Flat RAG vs GraphRAG

### Flat RAG

- **Indexing**: TF-IDF vectorizer trên 5,385 chunks (500 ký tự/chunk)
- **Search**: Cosine similarity giữa query vector và chunk vectors
- **Ưu điểm**: Đơn giản, nhanh, coverage cao
- **Nhược điểm**: Trả về chunk chứa keyword, không hiểu quan hệ

### GraphRAG

- **Indexing**: Đồ thị tri thức NetworkX MultiDiGraph
- **Search**: Traversal 2-hop từ entity chính
  - 1-hop: Quan hệ trực tiếp
  - 2-hop: Quan hệ gián tiếp qua node trung gian
- **Ưu điểm**: Hiểu cấu trúc quan hệ, suy luận đa bước
- **Nhược điểm**: Chỉ trả lời được nếu entity có trong đồ thị

## 10. Kết quả Benchmark

### 20 câu hỏi benchmark

| # | Câu hỏi | Entity | GraphRAG | FlatRAG |
|---|---------|--------|----------|---------|
| 1 | Tesla co lien quan den nhung cong ty nao? | Tesla | CO | CO |
| 2 | Cac hang xe dien ban tai My? | United States | CO | CO |
| 3 | McKinsey danh gia thi truong EV ra sao? | McKinsey | CO | CO |
| 4 | Uu diem cua xe dien so voi xe xang? | Battery Electric Vehicle | CO | CO |
| 5 | Cox Automotive ve doanh so EV quy 1 2024? | Cox Automotive | CO | CO |
| 6 | So sanh EV vs ICE co loi ich gi? | ICE | CO | CO |
| 7 | Ha tang sac dien o My phat trien the nao? | Charging Station | CO | CO |
| 8 | Vai tro cua California trong nganh EV? | California | CO | CO |
| 9 | Chinh sach nao ho tro phat trien EV? | EPA | CO | CO |
| 10 | Thi truong EV toan cau nam 2024? | BloombergNEF | CO | CO |
| 11 | Nhung hang xe nao ban EV tai My? | Ford | CO | CO |
| 12 | Toyota co dong gop gi cho nganh EV? | Toyota | CO | CO |
| 13 | Trung Quoc anh huong the nao den EV? | China | CO | CO |
| 14 | NREL nghien cuu gi ve pin xe dien? | NREL | CO | CO |
| 15 | Cac cong nghe pin nao dang duoc dung? | Lithium-ion | CO | CO |
| 16 | Ford dang lam gi trong linh vuc EV? | Ford | CO | CO |
| 17 | Hyundai co san pham EV nao? | Hyundai | CO | CO |
| 18 | Duc dong vai tro gi trong cong nghiep EV? | Germany | CO | CO |
| 19 | Elon Musk lien quan den nhung cong ty nao? | Elon Musk | CO | CO |
| 20 | Nhung to chuc nao nghien cuu ve EV? | ICCT | CO | CO |

> Kết quả chi tiết tại `comparison_results.csv`

### Nhận xét

- Cả hai phương pháp đều đạt **20/20** với rule-based extraction
- **Flat RAG** ưu thế ở câu hỏi về entity không có trong đồ thị (nhờ TF-IDF search trên raw text)
- **GraphRAG** ưu thế ở câu hỏi cần suy luận quan hệ (vd: "Tesla liên quan đến công ty nào?" → traversal ra Ford, BMW, Rivian...)
- Trường hợp Flat RAG bị ảo giác: Khi query không match chính xác với chunk text, TF-IDF trả về kết quả nhiễu. GraphRAG không bị ảo giác nhưng từ chối trả lời nếu entity không có trong graph.

## 11. Phân tích chi phí

### Token Usage

| Item | Tokens |
|------|--------|
| Tổng dataset | ~300,763 tokens |
| Trung bình mỗi doc | ~4,300 tokens |

### Chi phí theo provider

| Provider | Model | Cost / 1K tokens | Ước tính cho dataset |
|----------|-------|------------------|---------------------|
| OpenAI | GPT-4o-mini | $0.002 | ~$0.60 |
| Groq | Llama3-70B | Free | $0.00 |
| Ollama | Local | Free | $0.00 |
| DeepSeek | deepseek-chat | $0.0005 | ~$0.15 |

### Thời gian xử lý

| Phương pháp | 1 doc | 70 docs |
|-------------|-------|---------|
| Rule-based | < 0.1s | ~2s |
| LLM (Ollama + minimax-m3) | ~6s | ~7 phút |
| LLM (GPT-4o-mini) | ~1s | ~1 phút |

## 12. Hướng dẫn sử dụng

### Với Ollama (local)

```bash
# 1. Cài Ollama: https://ollama.com
# 2. Pull model
ollama pull llama3

# 3. Cấu hình .env
echo "LLM_PROVIDER=ollama
LLM_MODEL=llama3
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=4096" > .env

# 4. Chạy
python main.py
```

### Với OpenAI

```bash
# .env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-xxx
```

### Với Groq (free)

```bash
# .env
LLM_PROVIDER=groq
LLM_MODEL=llama3-70b-8192
LLM_API_KEY=gsk_xxx
```

### Với DeepSeek

```bash
# .env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-xxx
```

---

## Deliverables

| STT | Yêu cầu | File |
|-----|---------|------|
| 1 | Mã nguồn Python | `main.py` + `graphrag/` |
| 2 | Ảnh đồ thị tri thức | `submit/knowledge_graph.png` |
| 3 | Bảng so sánh 20 câu benchmark | `submit/comparison_results.csv` |
| 4 | Phân tích chi phí | `submit/BaoCaoPhanTichChiPhi.md` |
| 5 | Task gốc | `submit/task.docx` |
