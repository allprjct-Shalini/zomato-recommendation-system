import streamlit as st
import pandas as pd
from model import load_and_process_data, recommend_restaurants

st.set_page_config(page_title="Zomato Restaurant Recommender", layout="centered")

@st.cache_resource
def get_data():
    return load_and_process_data()

st.title("🍽️ Zomato Restaurant Recommendation System")
st.write("Search or select a restaurant below to discover similar restaurants based on reviews, sorted by highest rating!")

with st.spinner("Loading and processing restaurant data... Please wait."):
    df_percent, cosine_similarities = get_data()

restaurant_list = sorted(df_percent.index.unique().tolist())
selected_restaurant = st.selectbox("Select a restaurant:", restaurant_list)

if st.button("Recommend Similar Restaurants"):
    with st.spinner("Finding best recommendations..."):
        result_df = recommend_restaurants(selected_restaurant, df_percent, cosine_similarities)
        if result_df is not None and not result_df.empty:
            st.success(f"Top Restaurants Similar to **{selected_restaurant}**:")
            st.dataframe(result_df)
        else:
            st.error("No recommendations found.")