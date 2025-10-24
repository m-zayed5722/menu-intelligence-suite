# Menu Intelligence Suite - Local Demo

## 🚀 Quick Start

Run this command to start everything:

```bash
start_demo.bat
```

This will launch:
1. **API Server** on http://127.0.0.1:8080
2. **Streamlit UI** on http://localhost:8501

## 📊 What's Included

- ✅ **11,000 menu items** with multilingual embeddings (EN/AR)
- ✅ **FAISS vector store** with 384-dim embeddings
- ✅ **BM25 sparse retrieval** for keyword search
- ✅ **Hybrid search** combining semantic + keyword
- ✅ **Auto-tagging** for cuisine and diet labels
- ✅ **No Docker required** - runs entirely in Python

## 🔍 Try These Searches

### English:
- `chicken shawarma`
- `spicy pizza`
- `grilled fish`
- `vegetable burger`

### Arabic:
- `دجاج` (chicken)
- `شاورما` (shawarma)
- `بيتزا` (pizza)
- `سمك` (fish)

### Hybrid (mix keywords + meaning):
- `healthy grilled options`
- `spicy seafood`
- `vegetarian arabic food`

## 🎯 Features to Explore

### 1. Search Modes
- **Dense**: Pure semantic search using embeddings
- **Sparse**: BM25 keyword matching
- **Hybrid**: Best of both worlds (recommended)

### 2. Auto-Tagging
Go to the "Auto-Tagging" tab and try:
```
Grilled chicken shawarma with garlic sauce and tahini
```

It will automatically classify by:
- **Cuisine**: Middle Eastern, Arabic, Mediterranean
- **Diet**: High Protein, Halal, etc.

### 3. Performance Tuning
- **Alpha slider**: Adjust sparse vs dense weight (hybrid mode)
- **efSearch**: Trade speed for accuracy in vector search

## 📈 Performance

- **First query**: ~3 seconds (includes model loading)
- **Subsequent queries**: <100ms
- **Vector search**: FAISS IVFFlat index
- **Model**: intfloat/multilingual-e5-small (118M params)

## 🛠️ Architecture

```
┌─────────────┐
│  Streamlit  │  (Port 8501)
│     UI      │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   FastAPI   │  (Port 8080)
│     API     │
└──────┬──────┘
       │
       ├──► FAISS Vector Store (11K vectors)
       ├──► BM25 Index (sparse retrieval)
       └──► sentence-transformers (embeddings)
```

## 🔧 Manual Start (Alternative)

If the batch file doesn't work, start services manually:

### Terminal 1 - API Server:
```bash
venv\Scripts\activate
set PYTHONIOENCODING=utf-8
python -c "from app_simple import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8080)"
```

### Terminal 2 - Streamlit:
```bash
venv\Scripts\activate
streamlit run ui_simple.py --server.port 8501
```

## 📝 API Documentation

Once the API is running, visit:
- **Interactive Docs**: http://127.0.0.1:8080/docs
- **OpenAPI Schema**: http://127.0.0.1:8080/openapi.json

### Key Endpoints:

**POST /api/search**
```json
{
  "query": "chicken shawarma",
  "k": 10,
  "mode": "hybrid",
  "alpha": 0.4
}
```

**POST /api/tag**
```json
{
  "text": "Grilled chicken with rice",
  "top_n": 3,
  "threshold": 0.3
}
```

## 🎓 Technical Details

### Vector Store
- **Backend**: FAISS (CPU version)
- **Index Type**: IVFFlat with 100 clusters
- **Distance**: Cosine similarity
- **Dimensionality**: 384

### Text Processing
- **Normalization**: Arabic diacritic removal, Alef/Yaa normalization
- **Tokenization**: sentencepiece (multilingual)
- **Embedding Model**: intfloat/multilingual-e5-small

### Sparse Retrieval
- **Algorithm**: BM25 (Okapi variant)
- **Tokenization**: Space-split + lowercasing
- **Corpus**: 11K normalized menu items

### Hybrid Search
- **Formula**: `score = α * sparse + (1-α) * dense`
- **Default α**: 0.4 (60% semantic, 40% keyword)
- **Score normalization**: Min-max scaling

## 🐛 Troubleshooting

### "Port already in use"
```bash
# Kill Python processes
taskkill /F /IM python.exe
```

### "Module not found"
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### "FAISS index not loaded"
```bash
# Regenerate data
python generate_local_data.py
```

### Slow first query
- First query loads the ML model (~500MB)
- Subsequent queries are much faster
- Model is cached in memory after first use

## 📚 Learn More

- **Full README**: See `README.md` for Docker deployment
- **Documentation**: Check `QUICKSTART.md` for detailed guide
- **Architecture**: Review `src/` directory for implementation

## ✅ Verification

To verify everything is working:

1. Open http://localhost:8501
2. Search for "chicken"
3. You should see ~10 results in <100ms
4. Try Arabic: "دجاج"
5. Results should be similar to English search

---

**Built with FastAPI, Streamlit, FAISS, and sentence-transformers**
