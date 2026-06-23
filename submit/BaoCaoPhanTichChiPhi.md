# BÁO CÁO PHÂN TÍCH CHI PHÍ
## LAB DAY 19: GraphRAG - US Electric Vehicle Sector

---

### 1. Token Usage

| Item | Giá trị |
|------|---------|
| Tổng số documents | 70 files |
| Tổng số tokens trong dataset | ~300,763 tokens |
| Trung bình mỗi document | ~4,300 tokens |
| Phương pháp extraction | Rule-based (không dùng LLM) |

### 2. Graph Construction

| Metric | Giá trị |
|--------|---------|
| Số nodes | 71 |
| Số edges | 775 |
| Số entity types | 6 (COMPANY, ORGANIZATION, PERSON, LOCATION, TECHNOLOGY, METRIC) |
| Thời gian xây dựng đồ thị | ~2 giây (rule-based) |

### 3. Chi phí theo Provider (nếu dùng LLM)

| Provider | Model | Cost / 1K tokens | Ước tính cho dataset |
|----------|-------|------------------|---------------------|
| OpenAI | GPT-4o-mini ($0.15/M input) | $0.00015 | ~$0.60 (4 calls) |
| Groq | Llama3-70B (free tier) | $0.00 | $0.00 |
| Ollama | minimax-m3:cloud (local) | $0.00 | $0.00 (chỉ tốn điện) |
| DeepSeek | deepseek-chat ($0.0005/K) | $0.0005 | ~$0.15 |
| Tổng LLM extraction (worst case) | 70 docs × ~4.3K tokens | - | ~$0.60 - $2.00 |

### 4. Thời gian xử lý

| Phương pháp | Thời gian 1 doc | Thời gian 70 docs |
|-------------|----------------|-------------------|
| Rule-based | < 0.1s | ~2s |
| Ollama + minimax-m3:cloud (batch 3) | ~6s | ~7 phút |
| GPT-4o-mini (batch 5) | ~1s | ~1 phút |

### 5. Flat RAG Indexing

| Metric | Giá trị |
|--------|---------|
| Số chunks | 5,385 |
| Phương pháp | TF-IDF (max_features=2000) |
| Thời gian index | ~3 giây |

### 6. Benchmark Results

| Hệ thống | Số câu đúng | Tổng số |
|----------|------------|---------|
| GraphRAG | 20 | 20 |
| FlatRAG | 19 | 20 |

### 7. Nhận xét

- **GraphRAG** hiệu quả với câu hỏi cần suy luận quan hệ (Tesla → Ford, Elon Musk)
- **FlatRAG** có coverage cao hơn nhưng dễ bị ảo giác khi query không match chính xác
- **Lỗ hổng**: FlatRAG miss câu Q15 về "Lithium-ion" vì TF-IDF không bắt được từ chuyên ngành
- **Khuyến nghị**: Hybrid approach: GraphRAG làm chính, FlatRAG làm fallback
