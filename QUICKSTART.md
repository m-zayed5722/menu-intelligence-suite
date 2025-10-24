# Menu Intelligence Suite - Quick Start Guide

## 🚀 Getting Started in 3 Minutes

### 1. Initial Setup

```bash
# Navigate to project directory
cd c:\Users\mzaye\OneDrive\Documents\Projects\MIS

# Copy environment configuration
copy .env.example .env

# Start all services (Docker required)
make dev
```

Wait for services to start (~30 seconds). You'll see:
```
✓ Services started!
  API:        http://localhost:8000
  API Docs:   http://localhost:8000/docs
  Streamlit:  http://localhost:8501
```

### 2. Load Sample Data

```bash
# Generate and load 10,000 menu items + 300 labeled queries
make seed
```

This creates:
- 10k menu items (EN/AR) with embeddings
- ~1k near-duplicate items
- 300 labeled search queries
- Cuisine and diet label sets

### 3. Try the Demo

Open http://localhost:8501 in your browser.

**Search Tab:**
- Try: "chicken shawarma" or "شاورما دجاج"
- Toggle between sparse/dense/hybrid modes
- Adjust alpha slider to see how it affects results

**Metrics Tab:**
- Click "Run Evaluation"
- See Recall@5, MRR, per-query breakdown
- Compare sparse vs dense vs hybrid performance

## 📊 Expected Results

After loading seed data, you should see:

| Metric | Target | Description |
|--------|--------|-------------|
| Hybrid Recall@10 | ≥0.90 | 90% of relevant items in top-10 |
| Arabic Dense Improvement | +15% | Dense beats sparse on Arabic queries |
| Dedup Precision@20 | ≥0.90 | 90% of predicted pairs are true duplicates |
| Search Latency (p95) | <150ms | Fast enough for production |

## 🔧 Common Commands

```bash
# View API logs
make logs

# Run tests
make test

# Access database
make db-shell

# Stop all services
docker-compose down

# Full cleanup (removes data)
docker-compose down -v
```

## 🐛 Troubleshooting

**Services won't start:**
```bash
docker-compose ps  # Check service status
docker-compose logs api  # View API logs
```

**Seed data fails:**
```bash
# Check API health first
curl http://localhost:8000/health

# Manually run generator
docker-compose exec api python src/data/generate_data.py
```

**Slow search:**
- Reduce `ef_search` slider in UI
- Change `VECTOR_BACKEND=faiss` in .env for smaller datasets

## 📖 Next Steps

1. **Explore API**: http://localhost:8000/docs
2. **Read README**: Full documentation in README.md
3. **Run Tests**: `make test` to verify everything works
4. **Customize**: Edit `.env` to change model, parameters, etc.

## 🎯 Demo Script (for Interviews)

### 1. Multilingual Search (2 min)
- Search "shawarma" → show EN results
- Search "شاورما" → show AR results with normalization
- Toggle sparse/dense → explain trade-offs

### 2. Hybrid Tuning (1 min)
- Start with alpha=1.0 (pure sparse)
- Slide to alpha=0.0 (pure dense)
- Settle on alpha=0.4 (balanced hybrid)

### 3. Evaluation (2 min)
- Run Metrics tab evaluation
- Show Recall@5 improvement: Sparse (0.65) → Hybrid (0.88)
- Explain MRR and per-query breakdown

### 4. Deduplication (1 min)
- Run dedup for "Riyadh"
- Open a cluster → show near-duplicate items
- Explain similarity threshold tuning

### 5. Architecture Discussion (3 min)
- Show docker-compose.yml (services)
- Explain pgvector vs FAISS trade-offs
- Discuss production scaling (index sharding, caching)

---

**Total Time**: ~10 minutes to full working demo 🎉
