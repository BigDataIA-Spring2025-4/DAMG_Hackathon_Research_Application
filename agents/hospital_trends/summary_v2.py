import os
import pandas as pd
from pypdf import PdfReader
from smolagents import CodeAgent, LiteLLMModel, tool

# Define the directory containing the data files
DATA_DIRECTORY = "./prototype/hospitalization_trends_data"

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
    """Analyzes emergency department visits from a PDF file."""
    file_path = os.path.join(DATA_DIRECTORY, "EmergencyDepartment_Visits.pdf")

    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    reader = PdfReader(file_path)
    
    total_pages = len(reader.pages)
    pages_to_read = min(2, total_pages)
    
    content = "\n\n".join(reader.pages[i].extract_text() for i in range(pages_to_read) if reader.pages[i].extract_text())

    return content if content else "Error: Unable to extract text from the PDF."


from pypdf import PdfReader
import os

@tool
def extract_hospital_utilization() -> str:
    """Extracts key insights from the first two pages of a hospital utilization research paper (PDF)."""
    
    file_path = os.path.join(DATA_DIRECTORY, "HospitalUtilization.pdf")
    
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    # Reading the PDF
    reader = PdfReader(file_path)
    
    # Total pages in the PDF
    total_pages = len(reader.pages)
    
    # Ensure we are reading only the first 2 pages (or fewer if not available)
    pages_to_read = min(2, total_pages)
    
    # Initialize an empty string to store the content
    content = ""
    
    # Loop through the pages and extract the text
    for i in range(pages_to_read):
        page = reader.pages[i]
        page_text = page.extract_text()
        
        # If text is found, append it; otherwise, note that the page has no text
        if page_text:
            content += page_text + "\n\n"
        else:
            content += f"Error: No text found on page {i+1}.\n"
    
    # Return the content or a fallback message if no content was extracted
    return content if content else "Error: Unable to extract text from the PDF."



### 4️⃣ Agent: Generate Summary Report
@tool
def summarize_findings(
    hospital_beds_result: str,
    emergency_visits_result: str,
    hospital_utilization_result: str
) -> str:
    """
    Summarizes key findings from the three research analyses.

    Args:
        hospital_beds_result (str): The analysis result of hospital bed trends, including average annual change, total change, start year, end year, etc.
        emergency_visits_result (str): The analysis result of emergency department visits, typically extracted from a PDF, containing trends and insights.
        hospital_utilization_result (str): Extracted insights from the hospital utilization research paper, summarizing key data and findings.

    Returns:
        str: A compiled research summary based on the provided analyses, structured in a readable format.
    """

    summary = f"""
    ### 📊 Final Research Summary ###

    🏥 **Hospital Bed Trends (CSV Data)**:
    {hospital_beds_result}

    🚑 **Emergency Department Visits (PDF Data)**:
    {emergency_visits_result}

    📄 **Hospital Utilization Research (PDF Data)**:
    {hospital_utilization_result}

    📌 **Conclusion**:
    - Hospital bed availability varies across states.
    - Emergency department visits show demographic trends.
    - The research paper provides deeper insights into hospital utilization patterns.
    """
    return summary



### **🔹 Running the Agents to Generate the Final Report**
if __name__ == "__main__":
    # Initialize Model
    model = LiteLLMModel(model_id="xai/grok-2-1212", api_key=os.getenv("XAI_API_KEY"))

    # Create and run agents
    hospital_beds_agent = CodeAgent(tools=[analyze_hospital_beds], model=model)
    # emergency_visits_agent = CodeAgent(tools=[analyze_emergency_visits], model=model)
    # hospital_utilization_agent = CodeAgent(tools=[extract_hospital_utilization], model=model)

    hospital_beds_result = hospital_beds_agent.run("Analyze hospital bed trends.")
    # emergency_visits_result = emergency_visits_agent.run("Analyze emergency department visits.")
    # hospital_utilization_result = hospital_utilization_agent.run("Extract insights from the research paper.")

    # Generate the final summary report
    summary_agent = CodeAgent(tools=[summarize_findings], model=model)
    final_report = summary_agent.run(
        f"Summarize the research findings:\n\n{hospital_beds_result}"
    )

    print("\n🔍 **Final Research Summary:**")
    print(final_report)
