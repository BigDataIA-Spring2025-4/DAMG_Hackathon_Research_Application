import os
import pandas as pd
from pypdf import PdfReader
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, tool

# Define the directory containing the data files
DATA_DIRECTORY = "./agents/hospital_trends/data"

### 1️⃣ Agent: Analyze Hospital Bed Data (CSV)
@tool
def analyze_hospital_beds() -> str:
    """Analyzes hospital bed availability trends from a CSV file."""
    file_path = os.path.join(DATA_DIRECTORY, "DQS_Community_hospital_beds__by_state__United_States.csv")
    
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    data = pd.read_csv(file_path)

    if "ESTIMATE" not in data.columns or "TIME_PERIOD" not in data.columns or "SUBGROUP" not in data.columns:
        return "Error: Required columns not found in the dataset."

    data["ESTIMATE"] = pd.to_numeric(data["ESTIMATE"], errors="coerce")
    data["TIME_PERIOD"] = pd.to_numeric(data["TIME_PERIOD"], errors="coerce")

    # Compute trends
    data["Percent_Change"] = data.groupby("SUBGROUP")["ESTIMATE"].pct_change() * 100

    summary = data.groupby("SUBGROUP").agg(
        Average_Annual_Change=("Percent_Change", "mean"),
        Total_Change=("ESTIMATE", lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100 if len(x) > 1 else 0),
        Start_Year=("TIME_PERIOD", "min"),
        End_Year=("TIME_PERIOD", "max"),
        Start_Value=("ESTIMATE", "first"),
        End_Value=("ESTIMATE", "last")
    ).reset_index()

    return summary.to_string()


### 2️⃣ Agent: Analyze Emergency Department Visits (PDF)
@tool
def analyze_emergency_visits() -> str:
    """
    Reads the first three pages of a PDF file from the local directory and returns its content as a string.
    """
    file_path = os.path.join(DATA_DIRECTORY, "EmergencyDepartment_Visits.pdf")

    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found. Please check the directory."

    reader = PdfReader(file_path)
    
    # Check number of pages
    total_pages = len(reader.pages)
    print(f"Total pages in PDF: {total_pages}")

    pages_to_read = min(2, total_pages)
    content = ""
    
    for i in range(pages_to_read):
        page = reader.pages[i]
        content += page.extract_text() + "\n\n"

    return content if content else "Error: Unable to extract text from the PDF."



### 3️⃣ Agent: Extract Hospital Utilization Data (PDF)
@tool
def extract_hospital_utilization() -> str:
    """
    Reads the first three pages of a PDF file from the local directory and returns its content as a string.
    """
    file_path = os.path.join(DATA_DIRECTORY, "HospitalUtilization.pdf")

    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found. Please check the directory."

    reader = PdfReader(file_path)
    
    # Check number of pages
    total_pages = len(reader.pages)
    print(f"Total pages in PDF: {total_pages}")

    pages_to_read = min(2, total_pages)
    content = ""
    
    for i in range(pages_to_read):
        page = reader.pages[i]
        content += page.extract_text() + "\n\n"

    return content if content else "Error: Unable to extract text from the PDF."

### **🔹 Running the Agents to Generate the Final Report**
if __name__ == "__main__":
    # Initialize Model
    model = LiteLLMModel(model_id="xai/grok-2-1212", api_key=os.getenv("XAI_API_KEY"))

    # Create and run agents
    hospital_beds_agent = ToolCallingAgent(tools=[analyze_hospital_beds], model=model,name="hospital_beds_agent",description="Analyzes Community hospital bed availability trends")
    emergency_visits_agent = ToolCallingAgent(tools=[analyze_emergency_visits], model=model,name="emergency_visits_agent",description="Analyzes Emergency department visits trends for US")
    hospital_utilization_agent = ToolCallingAgent(tools=[extract_hospital_utilization], model=model,name="hospital_utilization_agent",description="Analyzes hospital utilization trends for US")

    print("\n🔍 **Final Research Summary:**")
    
    manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[hospital_beds_agent,emergency_visits_agent,hospital_utilization_agent],
    additional_authorized_imports=["time", "numpy", "pandas","pypdf","os"]
    )
    
    state="California"
    answer = manager_agent.run(f"""
    You are a data analyst tasked with generating a comprehensive report on how for the {state} in US community hospital beds trends follow.
    Follow the tool to gather any necessary information and data, then create a detailed markdown report:
    - Rules 
        - Use all three tools avaiable to build additional context and final report generation
        - Do not generate any data use only the mentioned csv file
        - Try and relate the data to the state and how it affected the state

    Parameters:
        hospital_beds (str): Summary of hospital bed trends.

    Returns:
        str: A structured summary report.
    """)