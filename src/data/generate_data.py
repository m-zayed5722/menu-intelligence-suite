"""Generate synthetic menu data with EN/AR items, duplicates, and query labels."""
import json
import os
import random
from pathlib import Path

import httpx
import pandas as pd

# Set seed for reproducibility
random.seed(42)

# Base data
CITIES = [
    "Riyadh", "Jeddah", "Dubai", "Abu Dhabi", "Doha", "Kuwait City",
    "Beirut", "Amman", "Muscat", "Manama", "Cairo", "Istanbul",
    "Casablanca", "Tunis", "Baghdad", "Damascus", "Khartoum", "Sana'a",
    "Algiers", "Tripoli"
]

CUISINES = [
    "Lebanese", "Saudi", "Turkish", "Indian", "Italian", "Seafood",
    "Dessert", "Fast Food", "Asian", "Mexican", "American", "Arabic",
    "Mediterranean", "Japanese", "Chinese", "Pakistani", "Egyptian"
]

DIET_TAGS = [
    "halal", "vegetarian", "vegan", "spicy", "gluten free", "keto", "healthy"
]

# Food items templates (EN, AR)
FOOD_TEMPLATES = [
    ("Chicken Shawarma", "شاورما دجاج"),
    ("Beef Kebab", "كباب لحم"),
    ("Falafel Wrap", "سندويش فلافل"),
    ("Mixed Grill Platter", "مشاويات مشكلة"),
    ("Hummus Bowl", "صحن حمص"),
    ("Tabbouleh Salad", "سلطة تبولة"),
    ("Margherita Pizza", "بيتزا مارغريتا"),
    ("Chicken Biryani", "برياني دجاج"),
    ("Lamb Chops", "ريش غنم"),
    ("Grilled Salmon", "سلمون مشوي"),
    ("Caesar Salad", "سلطة سيزر"),
    ("Burger and Fries", "برجر مع بطاطس"),
    ("Sushi Platter", "طبق سوشي"),
    ("Chicken Tikka", "تكا دجاج"),
    ("Mandi Rice", "رز مندي"),
    ("Kunafa Dessert", "كنافة"),
    ("Baklava", "بقلاوة"),
    ("Chocolate Cake", "كيك شوكولاتة"),
    ("Fresh Juice", "عصير طازج"),
    ("Turkish Coffee", "قهوة تركية"),
]

OUTLET_NAMES = [
    "Al Baik", "Shawarmer", "Mama Noura", "Al Tazaj", "Kudu",
    "Herfy", "Domino's", "Pizza Hut", "McDonald's", "KFC",
    "Subway", "Nando's", "Sushi Art", "PF Chang's", "Chili's",
    "Applebee's", "The Cheesecake Factory", "Shake Shack", "Five Guys",
    "Krispy Kreme", "Dunkin'", "Starbucks", "Tim Hortons", "Costa Coffee",
    "Al Romansiah", "Al Nakheel", "Al Faisaliah", "Al Khaima"
]


def generate_items(n=10000):
    """Generate synthetic menu items."""
    items = []
    
    for i in range(n):
        # Pick random base item
        title_en, title_ar = random.choice(FOOD_TEMPLATES)
        
        # Add variations
        if random.random() < 0.3:
            prefixes = ["Special", "Deluxe", "Premium", "Classic", "Traditional"]
            title_en = f"{random.choice(prefixes)} {title_en}"
        
        if random.random() < 0.2:
            suffixes = ["Combo", "Meal", "Platter", "Box", "Special"]
            title_en = f"{title_en} {random.choice(suffixes)}"
        
        # Generate description
        descriptions = [
            f"Delicious {title_en.lower()} prepared fresh daily",
            f"Our signature {title_en.lower()} with special spices",
            f"Authentic {title_en.lower()} made with quality ingredients",
            f"Popular {title_en.lower()} served hot and fresh",
        ]
        description = random.choice(descriptions) if random.random() < 0.7 else None
        
        # Pick outlet and city
        outlet_name = random.choice(OUTLET_NAMES)
        city = random.choice(CITIES)
        
        # Generate coordinates (rough estimates)
        lat = 20.0 + random.uniform(0, 20)
        lon = 30.0 + random.uniform(0, 25)
        
        # Price
        price = round(random.uniform(10, 150), 2)
        
        # Tags
        cuisine_tags = random.sample(CUISINES, k=random.randint(1, 2))
        diet_tags = random.sample(DIET_TAGS, k=random.randint(0, 2)) if random.random() < 0.5 else []
        
        item = {
            "outlet_id": hash(outlet_name) % 10000,
            "outlet_name": outlet_name,
            "city": city,
            "lat": lat,
            "lon": lon,
            "title_en": title_en,
            "title_ar": title_ar,
            "description": description,
            "price": price,
            "cuisine_tags": cuisine_tags,
            "diet_tags": diet_tags,
        }
        
        items.append(item)
    
    # Inject near-duplicates (10% of items)
    num_dupes = n // 10
    for _ in range(num_dupes):
        # Pick a random item to duplicate
        original = random.choice(items)
        
        # Create slight variation
        dupe = original.copy()
        dupe["outlet_id"] = original["outlet_id"] + random.randint(1, 100)
        dupe["outlet_name"] = original["outlet_name"] + " Branch"
        
        # Slight text variations
        if random.random() < 0.5:
            dupe["title_en"] = original["title_en"].replace("Chicken", "Grilled Chicken")
        if random.random() < 0.3:
            dupe["price"] = original["price"] + random.uniform(-5, 5)
        
        items.append(dupe)
    
    return items


def generate_queries(items, n=300):
    """Generate search queries with relevance labels."""
    queries = []
    
    # Extract unique item titles
    titles = list(set([item["title_en"] for item in items]))
    
    for i in range(n):
        if i < n // 2:
            # Exact match queries
            target_title = random.choice(titles)
            query = target_title
            
            # Find relevant items
            relevant_ids = [
                j for j, item in enumerate(items, start=1)
                if target_title.lower() in item["title_en"].lower()
            ]
        else:
            # Partial/keyword queries
            words = ["chicken", "shawarma", "burger", "pizza", "salad", "kebab",
                     "rice", "grill", "fresh", "special", "platter", "wrap"]
            ar_words = ["دجاج", "شاورما", "برجر", "بيتزا", "سلطة", "كباب"]
            
            if random.random() < 0.3:
                # Arabic query
                query = random.choice(ar_words)
                relevant_ids = [
                    j for j, item in enumerate(items, start=1)
                    if item["title_ar"] and query in item["title_ar"]
                ]
            else:
                # English query
                query = random.choice(words)
                relevant_ids = [
                    j for j, item in enumerate(items, start=1)
                    if query.lower() in item["title_en"].lower()
                ]
        
        if relevant_ids:
            queries.append({
                "query": query,
                "relevant_ids": relevant_ids[:20],  # Limit to top 20
            })
    
    return queries


def main():
    """Generate and save data."""
    print("Generating synthetic menu data...")
    
    # Create output directory
    output_dir = Path("src/data/seed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate items
    print("Generating 10,000 menu items...")
    items = generate_items(10000)
    
    # Save items to CSV
    df = pd.DataFrame(items)
    items_path = output_dir / "items.csv"
    df.to_csv(items_path, index=False)
    print(f"✓ Items saved to {items_path}")
    
    # Generate queries
    print("Generating 300 labeled queries...")
    queries = generate_queries(items, 300)
    
    # Save queries
    queries_path = output_dir / "queries.json"
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"✓ Queries saved to {queries_path}")
    
    # Create labels file
    labels = {
        "cuisine": CUISINES,
        "diet": DIET_TAGS,
    }
    
    labels_path = output_dir / "labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    print(f"✓ Labels saved to {labels_path}")
    
    # Load data into API
    print("\nLoading data via API...")
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    try:
        # Test health
        response = httpx.get(f"{api_url}/health", timeout=5.0)
        if response.status_code != 200:
            print("⚠ API not healthy. Data generated but not loaded.")
            return
        
        # Ingest items in batches
        batch_size = 100
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            response = httpx.post(
                f"{api_url}/ingest",
                json={"items": batch},
                timeout=60.0
            )
            if response.status_code == 200:
                print(f"  Loaded batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")
            else:
                print(f"  ✗ Batch {i//batch_size + 1} failed: {response.text}")
        
        print("✓ All items loaded into API")
        
        # Load query labels into database
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import Session
        
        db_url = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/mis")
        engine = create_engine(db_url)
        
        with Session(engine) as db:
            # Clear existing labels
            db.execute(text("DELETE FROM query_labels"))
            
            # Insert new labels
            for q in queries:
                db.execute(
                    text("INSERT INTO query_labels (query, relevant_ids) VALUES (:query, :relevant_ids)"),
                    {"query": q["query"], "relevant_ids": q["relevant_ids"]}
                )
            
            db.commit()
        
        print(f"✓ {len(queries)} query labels loaded into database")
        
    except Exception as e:
        print(f"⚠ Could not load data via API: {e}")
        print("  Data files are saved and can be loaded manually.")
    
    print("\n✓ Data generation complete!")


if __name__ == "__main__":
    main()
