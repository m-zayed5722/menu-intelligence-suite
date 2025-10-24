"""Streamlit demo UI for Menu Intelligence Suite (Local Version)."""
import streamlit as st
import pandas as pd
import requests
import json

# Point to local API (will start it if needed)
API_URL = "http://127.0.0.1:8080"

st.set_page_config(
    page_title="Menu Intelligence Suite",
    page_icon="🍽",
    layout="wide",
)

st.title("Menu Intelligence Suite")
st.markdown("**Multilingual semantic search for food delivery (Local Demo)**")

# Check API health
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.sidebar.success(f"✓ API Connected | {health_data['vector_store_count']} vectors")
    else:
        st.sidebar.error("API not responding")
except:
    st.sidebar.warning("⚠ API not running. Start with: python -c \"from app_simple import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8080)\"")

# Sidebar controls
st.sidebar.header("Configuration")

mode = st.sidebar.selectbox(
    "Search Mode",
    ["hybrid", "dense", "sparse"],
    index=0,
    help="Hybrid combines BM25 + semantic search"
)

alpha = st.sidebar.slider(
    "Alpha (Sparse Weight)",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Weight for sparse retrieval (1.0 = pure BM25, 0.0 = pure semantic)",
    disabled=(mode != "hybrid")
)

normalize_arabic = st.sidebar.checkbox(
    "Normalize Arabic",
    value=True,
    help="Remove diacritics and normalize Arabic text"
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search", "🏷️ Auto-Tagging", "📊 About"])

# ==================== TAB 1: SEARCH ====================
with tab1:
    st.header("Search Menu Items")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Try: 'chicken shawarma' or 'دجاج' (Arabic)",
            help="Enter English or Arabic text"
        )
    
    with col2:
        k = st.number_input("Results", min_value=1, max_value=50, value=10)
    
    if st.button("🔍 Search", type="primary"):
        if not query:
            st.warning("Please enter a search query")
        else:
            try:
                with st.spinner("Searching..."):
                    response = requests.post(
                        f"{API_URL}/api/search",
                        json={
                            "query": query,
                            "k": k,
                            "mode": mode,
                            "alpha": alpha,
                            "normalize_arabic": normalize_arabic
                        },
                        timeout=30
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    timings = data.get("timings", {})
                    
                    if results:
                        st.success(f"Found {len(results)} results in {timings.get('total_ms', 0):.1f}ms")
                        
                        # Display results as table
                        df = pd.DataFrame(results)
                        
                        # Format columns
                        if 'score' in df.columns:
                            df['score'] = df['score'].apply(lambda x: f"{x:.3f}")
                        if 'price' in df.columns:
                            df['price'] = df['price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        
                        # Reorder columns
                        display_cols = ['title_en', 'title_ar', 'outlet_name', 'city', 'price', 'score']
                        display_cols = [c for c in display_cols if c in df.columns]
                        
                        st.dataframe(
                            df[display_cols],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Show timing breakdown
                        with st.expander("⏱️ Performance Details"):
                            col1, col2, col3, col4 = st.columns(4)
                            if timings.get('encode_ms'):
                                col1.metric("Encoding", f"{timings['encode_ms']:.1f}ms")
                            if timings.get('sparse_ms'):
                                col2.metric("Sparse", f"{timings['sparse_ms']:.1f}ms")
                            if timings.get('dense_ms'):
                                col3.metric("Dense", f"{timings['dense_ms']:.1f}ms")
                            if timings.get('hybrid_ms'):
                                col4.metric("Hybrid", f"{timings['hybrid_ms']:.1f}ms")
                    else:
                        st.info("No results found. Try a different query.")
                else:
                    st.error(f"Search failed: {response.text}")
            
            except requests.exceptions.Timeout:
                st.error("Request timed out. The API might be processing...")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure it's running on port 8080.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== TAB 2: TAGGING ====================
with tab2:
    st.header("Auto-Tag Menu Items")
    st.markdown("Automatically assign cuisine and diet labels based on item description")
    
    text_input = st.text_area(
        "Item Description",
        placeholder="Enter a food item description... e.g., 'Grilled chicken shawarma with garlic sauce'",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Top N Labels", 1, 5, 3)
    with col2:
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    
    if st.button("🏷️ Generate Tags", type="primary"):
        if not text_input:
            st.warning("Please enter an item description")
        else:
            try:
                with st.spinner("Analyzing..."):
                    response = requests.post(
                        f"{API_URL}/api/tag",
                        json={
                            "text": text_input,
                            "top_n": top_n,
                            "threshold": threshold
                        },
                        timeout=10
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Cuisine Labels")
                        cuisine = data.get("cuisine", [])
                        if cuisine:
                            for tag in cuisine:
                                st.write(f"**{tag['label']}** - {tag['score']:.2f}")
                        else:
                            st.info("No cuisine labels above threshold")
                    
                    with col2:
                        st.subheader("Diet Labels")
                        diet = data.get("diet", [])
                        if diet:
                            for tag in diet:
                                st.write(f"**{tag['label']}** - {tag['score']:.2f}")
                        else:
                            st.info("No diet labels above threshold")
                else:
                    st.error(f"Tagging failed: {response.text}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== TAB 3: ABOUT ====================
with tab3:
    st.header("About Menu Intelligence Suite")
    
    st.markdown("""
    ### Features
    
    - **Multilingual Search**: Search in English and Arabic seamlessly
    - **Hybrid Retrieval**: Combines BM25 (keyword) + Dense (semantic) search
    - **Auto-Tagging**: Automatically classify items by cuisine and diet
    - **FAISS Vector Store**: Fast approximate nearest neighbor search
    - **No Database Required**: Runs entirely in-memory for this demo
    
    ### Architecture
    
    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    - **ML Model**: intfloat/multilingual-e5-small (384-dim embeddings)
    - **Vector Store**: FAISS (IVFFlat index)
    - **Sparse Retrieval**: BM25 (rank-bm25)
    
    ### Dataset
    
    - 11,000 synthetic menu items
    - 20 MENA cities
    - English + Arabic titles
    - Multiple cuisines and diet categories
    
    ### Performance
    
    - First query: ~3s (includes model loading)
    - Subsequent queries: <100ms
    - Vector similarity: Cosine distance on normalized embeddings
    """)
    
    st.markdown("---")
    st.markdown("**Built for production-scale food delivery platforms**")
