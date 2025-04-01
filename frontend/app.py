import json, time
import streamlit as st
import requests, os, base64
from io import StringIO
from dotenv import load_dotenv
load_dotenv()

# API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_URL = "https://fastapi-service-vclcprawja-ue.a.run.app"

# def trigger_all_agents():
#     exit()

def main():    
    st.title("Hospitalization Trends in the United States: A Multi-State Analysis (2000-2023)")
    
    st.sidebar.header("Main Menu") 
    state_list = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", 
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", 
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", 
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", 
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]
    state = st.sidebar.selectbox("Select Your State:", state_list)
    tigger = st.sidebar.button("Begin Analysis", use_container_width=True, icon = "📄")
    st.header(f"Selected State : {state}")
    if tigger:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/generate_research", json={"state": state})
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown(answer)
            else:
                st.error(f"Error: {response.text}")
        
if __name__ == "__main__":
# Set page configuration
    st.set_page_config(
        page_title="Hospitalization Trends in the United States: A Multi-State Analysis (2000-2023)",
        layout="wide",
        initial_sidebar_state="expanded"
    )    
    main()
