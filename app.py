import streamlit as st
import pandas as pd
from model import load_and_process_data, recommend_restaurants
from utils import local_css

# Page Configuration
st.set_page_config(page_title="Zomato Restaurant Recommender", page_icon="🍔", layout="wide")

# Apply Custom CSS
local_css()

@st.cache_resource
def get_data():
    return load_and_process_data()

with st.spinner("🚀 Loading Zomato dataset and preparing recommendation engine..."):
    df_percent, cosine_similarities = get_data()

# --- HEADER SECTION ---
st.image("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
st.title("🍽️ Zomato Intelligent Restaurant Recommendation System")
st.markdown("Discover curated restaurant suggestions based on customer review patterns, food styles, and top ratings.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔍 Search Controls")
restaurant_list = sorted(df_percent.index.unique().tolist())
selected_restaurant = st.sidebar.selectbox("Choose or type a restaurant:", restaurant_list)

# Quick Metrics Dashboard
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Insights")
st.sidebar.metric(label="Total Restaurants Indexed", value=f"{df_percent.index.nunique():,}")
st.sidebar.metric(label="Average Platform Rating", value=f"{df_percent['rate'].mean():.2f} / 5.0")

# --- MAIN CONTENT AREA ---
if st.sidebar.button("✨ Get Recommendations", type="primary"):
    with st.spinner(fsw := f"Analyzing reviews to find spots similar to {selected_restaurant}..."):
        result_df = recommend_restaurants(selected_restaurant, df_percent, cosine_similarities)
        
        if result_df is not None and not result_df.empty:
            st.success(f"Top 10 Restaurants Similar to **{selected_restaurant}** based on Review Context:")
            
            # Display results in an appealing grid/card format
            for index, row in result_df.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        st.image("https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=300&q=80", use_container_width=True)
                    with col2:
                        st.subheader(f"🍴 {index}")
                        st.write(f"**Cuisines:** {row['cuisines']}")
                        st.write(f"📍 **City/Location:** {row['city']}")
                    with col3:
                        st.metric(label="Mean Rating", value=f"⭐ {row['Mean Rating']}")
                        st.write(f"💵 **Cost for Two:** ₹{row['cost']}")
                    st.markdown("---")
        else:
            st.warning("No matching recommendations found. Try selecting another popular restaurant.")
else:
    st.info("👈 Select a restaurant from the sidebar menu and click **Get Recommendations** to start exploring!")