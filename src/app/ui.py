"""Streamlit demo UI for Menu Intelligence Suite."""
import os

import httpx
import pandas as pd
import streamlit as st

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Menu Intelligence Suite",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Menu Intelligence Suite")
st.markdown("**Multilingual semantic search and intelligence for food delivery**")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.selectbox(
    "Search Mode",
    ["hybrid", "sparse", "dense"],
    index=0,
    help="Hybrid combines sparse and dense retrieval"
)

alpha = st.sidebar.slider(
    "Alpha (Sparse Weight)",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Weight for sparse retrieval in hybrid mode (1.0 = pure sparse, 0.0 = pure dense)",
    disabled=(mode != "hybrid")
)

ef_search = st.sidebar.slider(
    "efSearch (ANN Quality)",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Higher values = better quality but slower search"
)

normalize_arabic = st.sidebar.checkbox(
    "Normalize Arabic",
    value=True,
    help="Remove diacritics and normalize Alef/Yaa variants"
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "🔗 Deduplication", "🏷️ Tagging", "📊 Metrics"])

# ============================================================================
# TAB 1: Search
# ============================================================================
with tab1:
    st.header("Search Menu Items")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter query (EN or AR)",
            placeholder="e.g., chicken shawarma, شاورما دجاج",
            key="search_query"
        )
    with col2:
        k = st.number_input("Results", min_value=1, max_value=50, value=10)
    
    if st.button("Search", type="primary"):
        if not query:
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching..."):
                try:
                    response = httpx.post(
                        f"{API_URL}/search",
                        json={
                            "query": query,
                            "k": k,
                            "mode": mode,
                            "alpha": alpha,
                            "ef_search": ef_search,
                            "normalize_arabic": normalize_arabic,
                        },
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display timings
                        timings = data["timings"]
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Time", f"{timings['total_ms']:.1f} ms")
                        if timings.get("sparse_ms"):
                            col2.metric("Sparse", f"{timings['sparse_ms']:.1f} ms")
                        if timings.get("dense_ms"):
                            col3.metric("Dense", f"{timings['dense_ms']:.1f} ms")
                        if timings.get("encode_ms"):
                            col4.metric("Encoding", f"{timings['encode_ms']:.1f} ms")
                        
                        st.divider()
                        
                        # Display results
                        if data["results"]:
                            st.success(f"Found {len(data['results'])} results")
                            
                            for i, item in enumerate(data["results"], 1):
                                with st.container():
                                    col1, col2, col3 = st.columns([0.5, 3, 1])
                                    
                                    with col1:
                                        st.metric("Rank", i)
                                        st.metric("Score", f"{item['score']:.3f}")
                                    
                                    with col2:
                                        st.markdown(f"**{item['title_en']}**")
                                        if item['title_ar']:
                                            st.markdown(f"*{item['title_ar']}*")
                                        st.caption(f"📍 {item['outlet_name']} • {item['city']}")
                                    
                                    with col3:
                                        if item['price']:
                                            st.metric("Price", f"${item['price']:.2f}")
                                        st.caption(f"ID: {item['item_id']}")
                                    
                                    st.divider()
                        else:
                            st.info("No results found")
                    else:
                        st.error(f"API Error: {response.text}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 2: Deduplication
# ============================================================================
with tab2:
    st.header("Item Deduplication")
    
    col1, col2 = st.columns(2)
    with col1:
        dedup_city = st.text_input("Filter by city (optional)", key="dedup_city")
    with col2:
        sim_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.82,
            step=0.02,
            help="Higher = stricter matching"
        )
    
    if st.button("Find Duplicates", type="primary"):
        with st.spinner("Clustering duplicates..."):
            try:
                response = httpx.post(
                    f"{API_URL}/dedup/cluster",
                    json={
                        "city": dedup_city if dedup_city else None,
                        "sim_threshold": sim_threshold,
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    stats = data["stats"]
                    
                    # Display stats
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Items", stats["total_items"])
                    col2.metric("Clusters Found", stats["num_clusters"])
                    col3.metric("Duplicate Items", stats["num_duplicates"])
                    
                    st.divider()
                    
                    # Display clusters
                    if data["clusters"]:
                        st.success(f"Found {len(data['clusters'])} duplicate clusters")
                        
                        for cluster in data["clusters"][:20]:  # Limit display
                            with st.expander(
                                f"Cluster {cluster['cluster_id']} ({len(cluster['item_ids'])} items)"
                            ):
                                st.write(f"Item IDs: {', '.join(map(str, cluster['item_ids']))}")
                    else:
                        st.info("No duplicate clusters found")
                else:
                    st.error(f"API Error: {response.text}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 3: Tagging
# ============================================================================
with tab3:
    st.header("Auto-Tagging")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        tag_text = st.text_area(
            "Enter item text to tag",
            placeholder="e.g., Grilled chicken shawarma with fresh vegetables",
            height=100
        )
    with col2:
        top_n = st.number_input("Top N labels", min_value=1, max_value=5, value=2)
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05
        )
    
    if st.button("Tag Item", type="primary"):
        if not tag_text:
            st.warning("Please enter text to tag")
        else:
            with st.spinner("Tagging..."):
                try:
                    response = httpx.post(
                        f"{API_URL}/tag",
                        json={
                            "text": tag_text,
                            "top_n": top_n,
                            "threshold": threshold,
                        },
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🍴 Cuisine")
                            if data["cuisine"]:
                                for label_data in data["cuisine"]:
                                    st.metric(
                                        label_data["label"],
                                        f"{label_data['score']:.3f}"
                                    )
                            else:
                                st.info("No cuisine labels above threshold")
                        
                        with col2:
                            st.subheader("🥗 Diet")
                            if data["diet"]:
                                for label_data in data["diet"]:
                                    st.metric(
                                        label_data["label"],
                                        f"{label_data['score']:.3f}"
                                    )
                            else:
                                st.info("No diet labels above threshold")
                    else:
                        st.error(f"API Error: {response.text}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 4: Metrics
# ============================================================================
with tab4:
    st.header("Offline Evaluation Metrics")
    
    col1, col2 = st.columns([2, 2])
    with col1:
        eval_k = st.number_input("Evaluate at k", min_value=1, max_value=20, value=5)
    with col2:
        eval_mode = st.selectbox("Mode", ["hybrid", "sparse", "dense"], key="eval_mode")
    
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating on labeled queries..."):
            try:
                response = httpx.get(
                    f"{API_URL}/metrics/search",
                    params={
                        "k": eval_k,
                        "mode": eval_mode,
                        "alpha": alpha,
                        "ef_search": ef_search,
                    },
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display aggregate metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"Recall@{eval_k}", f"{data['recall_at_k']:.3f}")
                    col2.metric("MRR", f"{data['mrr']:.3f}")
                    col3.metric(f"Precision@{eval_k}", f"{data['precision_at_k']:.3f}")
                    
                    st.divider()
                    
                    # Per-query table
                    if data["per_query"]:
                        st.subheader("Per-Query Results")
                        
                        df = pd.DataFrame(data["per_query"])
                        df = df.rename(columns={
                            "query": "Query",
                            "hit": "Hit",
                            "first_rank": "First Hit Rank",
                            "recall": "Recall"
                        })
                        
                        st.dataframe(
                            df,
                            use_container_width=True,
                            height=400
                        )
                        
                        # Summary
                        hit_rate = df["Hit"].mean()
                        st.info(f"Hit Rate: {hit_rate:.1%} ({df['Hit'].sum()}/{len(df)} queries)")
                else:
                    st.error(f"API Error: {response.text}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.sidebar.divider()
st.sidebar.caption(f"API: {API_URL}")
st.sidebar.caption("Version 0.1.0")
