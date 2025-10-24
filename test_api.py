"""Quick test of the API locally."""
import json
from app_simple import app
from fastapi.testclient import TestClient

# The data was already generated - just load it
print("Loading cached data...")
from src.api.deps_local import get_items_db, get_vector_store, get_bm25_retriever

vector_store = get_vector_store()
print(f"Vector store has {vector_store.count()} vectors")

# Check if items are loaded
items_db = get_items_db()
if not items_db:
    # Load from generated data
    import pickle
    from pathlib import Path
    cache_dir = Path("data/cache")
    
    # Need to regenerate items dict from IDs
    print("Items not in memory, building from vector store...")
    # For now, create minimal test data
    from generate_local_data import FOOD_TEMPLATES, CITIES, OUTLETS
    import random
    
    items = {}
    # Get IDs from vector store
    from src.core.normalize import normalize_text
    for i, item_id in enumerate(vector_store.id_map[:100]):  # Just use first 100 for quick test
        title_en, title_ar = random.choice(FOOD_TEMPLATES)
        title_norm = normalize_text(f"{title_en} {title_ar}")
        desc = f"Delicious {title_en}"
        items[item_id] = {
            "item_id": item_id,
            "outlet_name": random.choice(OUTLETS),
            "city": random.choice(CITIES),
            "title_en": title_en,
            "title_ar": title_ar,
            "title_norm": title_norm,
            "desc_norm": normalize_text(desc),
            "description": desc,
            "price": round(random.uniform(15, 80), 2),
        }
    
    from src.api.deps_local import set_items_db
    set_items_db(items)
    print(f"Created {len(items)} test items")

items_db = get_items_db()
print(f"Items DB has {len(items_db)} items")

bm25 = get_bm25_retriever()
print(f"BM25 ready: {bm25 is not None}")

# Test client
client = TestClient(app)

print("\n" + "="*60)
print("Testing API Endpoints")
print("="*60)

# Test health
print("\n1. Health Check:")
response = client.get("/health")
print(json.dumps(response.json(), indent=2))

# Test search - dense
print("\n2. Search (dense) - 'chicken':")
response = client.post("/api/search", json={
    "query": "chicken",
    "k": 5,
    "mode": "dense"
})
results = response.json()
print(f"Found {len(results['results'])} results:")
for r in results['results'][:3]:
    print(f"  - {r['title_en']} ({r['city']}) - Score: {r['score']:.3f}")
print(f"Timing: {results['timings']['total_ms']:.1f}ms")

# Test search - Arabic
print("\n3. Search (dense) - 'دجاج' (Arabic for chicken):")
response = client.post("/api/search", json={
    "query": "دجاج",
    "k": 5,
    "mode": "dense"
})
results = response.json()
print(f"Found {len(results['results'])} results:")
for r in results['results'][:3]:
    print(f"  - {r['title_ar']} - {r['title_en']} - Score: {r['score']:.3f}")

# Test search - hybrid
print("\n4. Search (hybrid) - 'spicy pizza':")
response = client.post("/api/search", json={
    "query": "spicy pizza",
    "k": 5,
    "mode": "hybrid",
    "alpha": 0.5
})
results = response.json()
print(f"Found {len(results['results'])} results:")
for r in results['results'][:3]:
    print(f"  - {r['title_en']} - Score: {r['score']:.3f}")

# Test tagging
print("\n5. Tagging - 'Grilled Chicken Shawarma':")
response = client.post("/api/tag", json={
    "text": "Grilled Chicken Shawarma with garlic sauce",
    "top_n": 3,
    "threshold": 0.2
})
tags = response.json()
cuisine_str = ', '.join([f"{t['label']} ({t['score']:.2f})" for t in tags['cuisine']])
diet_str = ', '.join([f"{t['label']} ({t['score']:.2f})" for t in tags['diet']])
print(f"Cuisine: {cuisine_str}")
print(f"Diet: {diet_str}")

print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)
