"""
Generate sample data and load into in-memory store for local demo.
"""
import json
import random
from pathlib import Path

from tqdm import tqdm

from src.api.deps_local import (
    set_items_db, 
    set_query_labels,
    get_vector_store,
    build_sparse_index,
    save_vector_store,
)
from src.core.embeddings import encode_texts
from src.core.normalize import normalize_text


# Arabic translations
ARABIC_FOODS = {
    "Chicken": "دجاج",
    "Beef": "لحم بقر",
    "Fish": "سمك",
    "Shrimp": "روبيان",
    "Vegetable": "خضروات",
    "Pasta": "معكرونة",
    "Pizza": "بيتزا",
    "Burger": "برجر",
    "Sandwich": "ساندويش",
    "Salad": "سلطة",
    "Soup": "شوربة",
    "Rice": "أرز",
    "Noodles": "نودلز",
    "Kebab": "كباب",
    "Shawarma": "شاورما",
    "Falafel": "فلافل",
    "Hummus": "حمص",
    "Tabbouleh": "تبولة",
    "Fattoush": "فتوش",
    "Grilled": "مشوي",
    "Fried": "مقلي",
    "Spicy": "حار",
    "with": "مع",
    "and": "و",
}

FOOD_TEMPLATES = [
    ("Chicken Shawarma", "شاورما دجاج"),
    ("Beef Kebab", "كباب لحم"),
    ("Grilled Fish", "سمك مشوي"),
    ("Shrimp Pasta", "معكرونة بالروبيان"),
    ("Vegetable Pizza", "بيتزا خضروات"),
    ("Beef Burger", "برجر لحم"),
    ("Chicken Sandwich", "ساندويش دجاج"),
    ("Caesar Salad", "سلطة سيزر"),
    ("Tomato Soup", "شوربة طماطم"),
    ("Fried Rice", "أرز مقلي"),
    ("Spicy Noodles", "نودلز حار"),
    ("Mixed Grill", "مشاوي مشكلة"),
    ("Falafel Wrap", "ساندويش فلافل"),
    ("Hummus Plate", "طبق حمص"),
    ("Chicken Wings", "أجنحة دجاج"),
    ("Fish Tacos", "تاكو سمك"),
    ("Veggie Burger", "برجر نباتي"),
    ("Greek Salad", "سلطة يونانية"),
    ("Beef Steak", "ستيك لحم"),
    ("Seafood Platter", "طبق مأكولات بحرية"),
]

CITIES = [
    "Dubai", "Abu Dhabi", "Riyadh", "Jeddah", "Cairo", 
    "Doha", "Kuwait City", "Manama", "Muscat", "Amman",
    "Beirut", "Casablanca", "Tunis", "Sharjah", "Al Ain",
    "Dammam", "Mecca", "Medina", "Alexandria", "Giza"
]

OUTLETS = [
    "The Kitchen", "Spice Route", "Garden Bistro", "Ocean Grill",
    "Fire & Flame", "Green Leaf", "Golden Fork", "Silver Spoon",
    "Tasty Bites", "Flavor Street", "Urban Eats", "Fresh & Co",
    "Sizzle House", "Catch of the Day", "Veggie Delight", "Meat Masters",
    "Pasta Palace", "Pizza Paradise", "Burger Barn", "Salad Station",
    "Grill Express", "Shawarma King", "Kebab House", "Seafood Bay",
    "Spice Garden", "Fusion Kitchen", "Heritage Flavors", "Modern Plates"
]


def generate_items(n=10000):
    """Generate synthetic food items."""
    items = {}
    
    print(f"🔄 Generating {n} food items...")
    
    for i in tqdm(range(n), desc="Creating items"):
        # Pick template
        title_en, title_ar = random.choice(FOOD_TEMPLATES)
        
        # Add variations
        if random.random() < 0.3:
            prefix = random.choice(["Spicy", "Grilled", "Fried", "Fresh", "Special"])
            prefix_ar = ARABIC_FOODS.get(prefix, prefix)
            title_en = f"{prefix} {title_en}"
            title_ar = f"{title_ar} {prefix_ar}"
        
        # Create item
        item_id = f"item_{i+1:06d}"
        outlet = random.choice(OUTLETS)
        city = random.choice(CITIES)
        price = round(random.uniform(15, 120), 2)
        
        # Normalize
        title_norm = normalize_text(f"{title_en} {title_ar}")
        desc_norm = normalize_text(f"Delicious {title_en.lower()} from {outlet}")
        
        items[item_id] = {
            "item_id": item_id,
            "outlet_name": outlet,
            "city": city,
            "title_en": title_en,
            "title_ar": title_ar,
            "description": f"Delicious {title_en.lower()} from {outlet}",
            "price": price,
            "title_norm": title_norm,
            "desc_norm": desc_norm,
        }
    
    # Add some near-duplicates (10%)
    print("🔄 Adding near-duplicates...")
    n_dupes = int(n * 0.1)
    base_items = list(items.values())[:n_dupes]
    
    for i, base in enumerate(tqdm(base_items, desc="Creating duplicates")):
        dupe_id = f"dupe_{i+1:06d}"
        
        # Slight variation
        variations = ["Special", "Premium", "Deluxe", "Classic", "Original"]
        prefix = random.choice(variations)
        
        items[dupe_id] = {
            "item_id": dupe_id,
            "outlet_name": base["outlet_name"],
            "city": base["city"],
            "title_en": f"{prefix} {base['title_en']}",
            "title_ar": f"{prefix} {base['title_ar']}",
            "description": base["description"],
            "price": base["price"] + random.uniform(-5, 5),
            "title_norm": normalize_text(f"{prefix} {base['title_en']} {base['title_ar']}"),
            "desc_norm": base["desc_norm"],
        }
    
    return items


def generate_query_labels():
    """Generate labeled queries for evaluation."""
    queries = []
    
    # Exact matches
    for title_en, title_ar in FOOD_TEMPLATES[:10]:
        queries.append({
            "query": title_en,
            "relevant_items": [f"item_{FOOD_TEMPLATES.index((title_en, title_ar))+1:06d}"],
        })
        queries.append({
            "query": title_ar,
            "relevant_items": [f"item_{FOOD_TEMPLATES.index((title_en, title_ar))+1:06d}"],
        })
    
    # Keyword searches
    keywords = ["chicken", "beef", "fish", "pizza", "burger", "salad", "دجاج", "لحم", "سمك"]
    for kw in keywords:
        queries.append({
            "query": kw,
            "relevant_items": [],  # Will match multiple
        })
    
    return queries


def embed_items(items):
    """Generate embeddings for all items."""
    print("\n🤖 Generating embeddings...")
    
    texts = []
    ids = []
    for item_id, item in items.items():
        text = f"{item['title_norm']} {item['desc_norm']}"
        texts.append(text)
        ids.append(item_id)
    
    # Batch encode
    batch_size = 100
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch = texts[i:i+batch_size]
        embeddings = encode_texts(batch, normalize=True)
        all_embeddings.extend(embeddings)
    
    return ids, all_embeddings


def main():
    """Generate and load data."""
    print("=" * 60)
    print("📦 Menu Intelligence Suite - Data Generator")
    print("=" * 60)
    
    # Generate items
    items = generate_items(n=10000)
    print(f"✓ Generated {len(items)} items")
    
    # Generate embeddings
    ids, embeddings = embed_items(items)
    
    # Load into vector store
    print("\n🔄 Loading into vector store...")
    vector_store = get_vector_store()
    
    # Prepare metadata
    metadata = [{"city": items[item_id]["city"]} for item_id in ids]
    
    vector_store.add(ids, embeddings, metadata)
    print(f"✓ Added {len(ids)} vectors to FAISS")
    
    # Save vector store
    save_vector_store()
    
    # Set items in memory
    set_items_db(items)
    print(f"✓ Loaded {len(items)} items into memory")
    
    # Build BM25 index
    print("\n🔄 Building BM25 index...")
    build_sparse_index(items)
    print("✓ BM25 index ready")
    
    # Generate query labels
    queries = generate_query_labels()
    set_query_labels(queries)
    print(f"✓ Generated {len(queries)} labeled queries")
    
    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   Items: {len(items)}")
    print(f"   Vectors: {vector_store.count()}")
    print(f"   Queries: {len(queries)}")
    print(f"\n🚀 Ready to start the API server!")


if __name__ == "__main__":
    main()
