import streamlit as st

def local_css():
    st.markdown("""
    <style>
    /* Main background & font style */
    .main {
        background-color: #f8f9fa;
    }
    /* Card design for recommendations */
    .restaurant-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 5px solid #ff3008;
    }
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)