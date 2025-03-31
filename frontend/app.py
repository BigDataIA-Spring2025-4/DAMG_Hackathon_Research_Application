import json, time
import streamlit as st
import requests, os, base64
from io import StringIO
from dotenv import load_dotenv
load_dotenv()

def trigger_all_agents():
    exit()

def main():    
    st.title("Hospitalization Trends in the United States: A Multi-State Analysis (2000-2023)")
    
    st.sidebar.header("Main Menu") 
    state = st.sidebar.selectbox("Select Your State:", ["Massachusetts","Illinois"])
    tigger = st.sidebar.button("Begin Analysis", use_container_width=True, icon = "📄")
    st.header(f"Selected State : {state}")
    if tigger == "Begin Analysis":
        trigger_all_agents()

        
if __name__ == "__main__":
# Set page configuration
    st.set_page_config(
        page_title="Hospitalization Trends in the United States: A Multi-State Analysis (2000-2023)",
        layout="wide",
        initial_sidebar_state="expanded"
    )    
    main()