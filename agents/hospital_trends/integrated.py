import os
import json
import snowflake.connector
import requests
import pandas as pd
from pypdf import PdfReader
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, tool
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Set up Tavily client for web searches
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(TAVILY_API_KEY)

# Define directories
DATA_DIRECTORY = "./agents/hospital_trends/data"

# ---- HELPER FUNCTIONS ----

def get_snowflake_connection():
    """Creates and returns a Snowflake connection."""
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )

def close_connection(cursor, conn):
    """Closes a Snowflake cursor and connection."""
    if cursor:
        cursor.close()
    if conn:
        conn.close()

# ---- TOOLS FOR HISTORICAL HEALTHCARE DATA ----

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

@tool
def extract_hospital_utilization() -> str:
    """Extracts key insights from a hospital utilization research paper."""
    file_path = os.path.join(DATA_DIRECTORY, "HospitalUtilization.pdf")
    
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    reader = PdfReader(file_path)
    
    total_pages = len(reader.pages)
    pages_to_read = min(2, total_pages)
    
    content = ""
    
    for i in range(pages_to_read):
        page = reader.pages[i]
        page_text = page.extract_text()
        
        if page_text:
            content += page_text + "\n\n"
        else:
            content += f"Error: No text found on page {i+1}.\n"
    
    return content if content else "Error: Unable to extract text from the PDF."

# ---- TOOLS FOR COVID-19 DATA ANALYSIS ----

@tool
def query_covid_cases_by_year(state: Optional[str] = None) -> str:
    """
    Retrieves year-over-year COVID-19 cases and deaths data.
    
    Args:
        state: Optional state filter. If None, returns data for all states.
    
    Returns:
        JSON string with year-over-year COVID cases and deaths.
    """
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Build the query with optional state filter
        base_query = """
        WITH state_agg AS 
        (
            SELECT 
                EXTRACT(YEAR FROM date) AS year, 
                state, 
                MAX(cases) AS cases, 
                MAX(deaths) AS deaths 
            FROM COVID19_GLOBAL_DATA_ATLAS.HLS_COVID19_USA.COVID19_USA_CASES_DEATHS_BY_STATE_DAILY_NYT
        """
        
        if state:
            base_query += f"    WHERE state='{state}'\n"
        
        query = base_query + """
            GROUP BY EXTRACT(YEAR FROM date), state
            ORDER BY state, year)
        SELECT year, state,
            cases - LAG(cases, 1, 0) OVER (PARTITION BY state ORDER BY state, year) as cases,
            deaths - LAG(deaths, 1, 0) OVER (PARTITION BY state ORDER BY state, year) as deaths    
        FROM state_agg;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=column_names)
        
        close_connection(cursor, conn)
        return df.to_json(orient="records")
    
    except Exception as e:
        return f"Error executing COVID cases query: {str(e)}"

@tool
def query_vaccine_providers(state: Optional[str] = None) -> str:
    """
    Retrieves data about COVID-19 vaccination provider locations.
    
    Args:
        state: Optional state filter. If None, returns data for all states.
    
    Returns:
        JSON string with vaccination provider counts by state.
    """
    us_states = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
        "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
        "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
        "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
        "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
        "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
    }
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT LOC_ADMIN_STATE, COUNT(*) AS PROVIDER_COUNT 
        FROM COVID19_GLOBAL_DATA_ATLAS.HLS_COVID19_USA.COVID_19_US_VACCINATING_PROVIDER_LOCATIONS
        """
        
        if state:
            state_code = us_states.get(state)
            if state_code:
                query += f" WHERE LOC_ADMIN_STATE = '{state_code}'"
        
        query += " GROUP BY LOC_ADMIN_STATE ORDER BY LOC_ADMIN_STATE"
        
        cursor.execute(query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=column_names)
        
        close_connection(cursor, conn)
        return df.to_json(orient="records")
    
    except Exception as e:
        return f"Error executing vaccine providers query: {str(e)}"

@tool
def query_healthcare_access(state: Optional[str] = None) -> str:
    """
    Retrieves healthcare access data including emergency departments, physician visits, 
    and delayed healthcare due to cost.
    
    Args:
        state: Optional state filter. If None, returns national data.
    
    Returns:
        JSON string with healthcare access data.
    """
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Dictionary to store results
        results = {
            "emergency_dept_visits": [],
            "physician_visits": [],
            "delayed_healthcare_by_year": []
        }
        
        # Query 1: Emergency department visits
        query1 = """
        SELECT PANEL, UNIT, YEAR, ESTIMATE 
        FROM DIVERSITY_EQUITY_AND_INCLUSION__ACCESS_TO_HEALTHCARE.DEI_HEALTHCARE."Healthcare Visits by Age/Sex/Race - USA"
        WHERE AGE = 'All ages' AND ESTIMATE > 0 AND UNIT = 'Number of visits in thousands'
        AND PANEL = 'Hospital emergency departments'
        """
            
        query1 += " ORDER BY YEAR DESC"
        
        cursor.execute(query1)
        results1 = cursor.fetchall()
        column_names1 = [desc[0] for desc in cursor.description]
        df1 = pd.DataFrame(results1, columns=column_names1)
        results["emergency_dept_visits"] = json.loads(df1.to_json(orient="records"))
        
        # Query 2: Physician office visits
        query2 = """
        SELECT PANEL, UNIT, YEAR, ESTIMATE 
        FROM DIVERSITY_EQUITY_AND_INCLUSION__ACCESS_TO_HEALTHCARE.DEI_HEALTHCARE."Healthcare Visits by Age/Sex/Race - USA"
        WHERE AGE = 'All ages' AND ESTIMATE > 0 AND UNIT = 'Number of visits in thousands'
        AND PANEL = 'Physician offices'
        """
            
        query2 += " ORDER BY YEAR DESC"
        
        cursor.execute(query2)
        results2 = cursor.fetchall()
        column_names2 = [desc[0] for desc in cursor.description]
        df2 = pd.DataFrame(results2, columns=column_names2)
        results["physician_visits"] = json.loads(df2.to_json(orient="records"))
        
        # Query 3: Delayed healthcare by year
        query3 = """
        SELECT YEAR, COUNT(*) AS COUNT 
        FROM DIVERSITY_EQUITY_AND_INCLUSION__ACCESS_TO_HEALTHCARE.DEI_HEALTHCARE."Delayed Healthcare Due to Cost - USA"
        GROUP BY YEAR
        """
        
        cursor.execute(query3)
        results3 = cursor.fetchall()
        column_names3 = [desc[0] for desc in cursor.description]
        df3 = pd.DataFrame(results3, columns=column_names3)
        results["delayed_healthcare_by_year"] = json.loads(df3.to_json(orient="records"))
        
        close_connection(cursor, conn)
        return json.dumps(results)
    
    except Exception as e:
        return f"Error executing healthcare access query: {str(e)}"

# ---- WEB SEARCH TOOLS ----

@tool
def web_search(query: str) -> str:
    """
    Searches the web for COVID-related information using Tavily API.
    
    Args:
        query: The search query, related to COVID-19 or healthcare.
    
    Returns:
        JSON string containing search results.
    """
    try:
        response = tavily_client.search(query=query)
        return response
    
    except Exception as e:
        print(f"Error in web search: {str(e)}")
        return json.dumps({"results": []})

@tool
def fetch_web_content(url: list) -> str:
    """
    Fetches content from specific URLs for detailed information.
    
    Args:
        url: List of URLs to fetch content from.
    
    Returns:
        The extracted content from the webpages.
    """
    try:
        response = tavily_client.extract(urls=url)
        return response["results"][0]["raw_content"]
    
    except Exception as e:
        print(f"Error in fetch web content: {str(e)}")
        return f"This is mock content about COVID-19 research and data analysis."

@tool
def web_search_emergingchallanges(query: str) -> str:
    """
    Searches the web for Emerging Challanges using Tavily api.
   
    Args:
        query (str): The search query, should be related to Health Care Staffing Shortages and Potential National Hospital Bed Shortage
   
    Returns:
        str: JSON string containing search results.
    """
    try:
        response = tavily_client.search(
        query=query
        )
        return response
   
    except Exception as e:
        print(f"Error in web search: {str(e)}")
        # Return empty results on error
        return json.dumps({"results": []})
       
    except Exception as e:
        return f"Error performing web search: {str(e)}"
   
 

@tool
def fetch_web_content(url: list) -> str:
    """
    Fetches content from a specific URL for detailed information.
   
    Args:
        url (list): List of URL's to fetch content from.
   
    Returns:
        str: The extracted content from the webpage.
    """
    try:
        response = tavily_client.extract(
            urls=url
        )
        return response["results"][0]["raw_content"]
   
    except Exception as e:
        print(f"Error in fetch web content: {str(e)}")
        # Return empty results on error
        return f"This is mock content about COVID-19 research and data analysis."
       
    except Exception as e:
        return f"Error fetching web content: {str(e)}"
    

### 3️⃣ Agent: Extract Hospital Utilization Data (PDF)
DATA_DIRECTORY=".\\agents\\emerging_challenges\\data"
@tool
def extract_emergingchallenges_pdf() -> str:
    """
    Reads the first three pages of a PDF file from the local directory and returns its content as a string.
    """
    file_path = os.path.join(DATA_DIRECTORY, "Emerging Challenges.pdf")

    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found. Please check the directory."
    reader = PdfReader(file_path)
    # Check number of pages
    total_pages = len(reader.pages)
    print(f"Total pages in PDF: {total_pages}")
    pages_to_read = min(3, total_pages)
    content = ""
    for i in range(pages_to_read):
        page = reader.pages[i]
        content += page.extract_text() + "\n\n"
    return content if content else "Error: Unable to extract text from the PDF."


# ---- COVID ANALYSIS FUNCTION ----

def run_covid_analysis(state="Illinois"):
    """
    Runs the COVID-19 data analysis for a specified state.
    
    Args:
        state: The state to analyze (default: Illinois)
    
    Returns:
        Agent output containing the comprehensive report
    """
    # Initialize the model
    model_id = "xai/grok-2-1212"
    model = LiteLLMModel(model_id=model_id, api_key=os.getenv("XAI_API_KEY"))
    
    # Create the agent with all specialized tools
    agent = CodeAgent(
        tools=[
            query_covid_cases_by_year,
            query_vaccine_providers,
            query_healthcare_access,
            web_search,
            fetch_web_content
        ],
        model=model,
        max_steps=25, # Increased from 20 to allow for more comprehensive analysis
        additional_authorized_imports=['pandas', 'json', 'requests', 'matplotlib', 'seaborn'],
        verbosity_level=2,
    )
    
    # Run the agent with a comprehensive prompt
    agent_output = agent.run(f"""
    You are a COVID-19 data analyst tasked with generating a COMPREHENSIVE and COMPLETE report on how the COVID-19 pandemic transformed healthcare in {state}. This will be part of a larger 20-page report.

    CRITICAL GUIDELINES:
    - You MUST produce a complete markdown report with ALL sections fully developed
    - Each section should be detailed and evidence-based (minimum 4-5 substantial paragraphs per section - At least 1500 words)
    - DO NOT omit or abbreviate any sections
    - Include detailed quantitative analysis where possible
    - Your final output MUST be a COMPLETE REPORT, not notes or partial analysis
    - Only retry a tool call at most 3 times. If still not resolved, continue with existing data
    - Process data one section at a time to minimize token usage
    
    Follow this step-by-step approach:

    1. EXECUTIVE SUMMARY & INTRODUCTION:
       - Start by querying basic COVID data for context: query_covid_cases_by_year("{state}")
       - Search the web for an overview: web_search("{state} COVID-19 healthcare impact overview")
       - Write a comprehensive Executive Summary (1 full page) and Introduction (1-2 full pages) that contextualizes the entire report
       - ENSURE THIS SECTION IS COMPLETE before moving to the next section

    2. PANDEMIC TIMELINE AND HEALTHCARE RESPONSE:
       - Thoroughly analyze the COVID cases/deaths data
       - Create a detailed timeline with key events and healthcare system responses
       - Identify turning points and policy changes
       - Include at least 5 paragraphs with specific data points and dates
       - ENSURE THIS SECTION IS COMPLETE before moving to the next section

    3. COMPARATIVE ANALYSIS:
       - Query healthcare access data: query_healthcare_access("{state}")
       - Focus on emergency department and physician visits data
       - Search for additional context: web_search("{state} pre-pandemic vs pandemic healthcare expenditure")
       - Create detailed comparative analysis with pre-pandemic baseline vs. pandemic changes
       - Include at least 4-5 paragraphs with quantitative comparisons
       - ENSURE THIS SECTION IS COMPLETE before moving to the next section

    4. SOCIAL DETERMINANTS AND COVID-19 IMPACT:
       - Analyze the delayed healthcare data in depth
       - Search for relevant information: web_search("{state} social determinants health COVID hotspots")
       - Identify vulnerable populations and geographic/demographic patterns
       - Include at least 4-5 paragraphs discussing disparities and social factors
       - ENSURE THIS SECTION IS COMPLETE before moving to the next section

    5. HEALTHCARE PROVIDER AVAILABILITY:
       - Query vaccination provider data: query_vaccine_providers("{state}")
       - Search for context: web_search("{state} healthcare provider availability COVID impact")
       - Analyze provider distribution, shortages, and adaptations
       - Include at least 4-5 paragraphs with specific provider metrics and distribution data
       - ENSURE THIS SECTION IS COMPLETE before moving to the next section

    6. LONG-TERM IMPLICATIONS:
       - Search for forecasting information: web_search("{state} long-term effects COVID healthcare access")
       - Analyze potential lasting changes to healthcare delivery and access
       - Discuss policy implications and system transformations
       - Include at least 4-5 paragraphs with detailed projections and industry analysis
       - ENSURE THIS SECTION IS COMPLETE before moving to the next section

    THE FINAL REPORT MUST FOLLOW THIS EXACT STRUCTURE:
    
    # COVID-19 Impact on Healthcare in {state}: Comprehensive Analysis
    
    ## Executive Summary
    [Detailed overview of key findings - minimum 4-5 paragraphs, equivalent to 1 full page]
    
    ## Introduction
    [Context of COVID-19 in {state}, scope of the report - minimum 4-5 paragraphs, equivalent to 1-2 full pages]
    
    ## Pandemic Timeline and Healthcare Response
    [Detailed timeline using cases/deaths data - minimum 4-5 substantial paragraphs]
    
    ## Comparative Analysis: Pre-Pandemic vs. Pandemic Healthcare
    [Thorough comparison of healthcare metrics - minimum 4-5 substantial paragraphs]
    
    ## Social Determinants and COVID-19 Impact
    [Detailed analysis of vulnerable populations - minimum 4-5 substantial paragraphs]
    
    ## Healthcare Provider Availability
    [Comprehensive analysis of provider distribution and access - minimum 4-5 substantial paragraphs]
    
    ## Long-Term Implications
    [In-depth forecast of healthcare transformation - minimum 4-5 substantial paragraphs]
    
    IMPORTANT: Your output MUST be a COMPLETE REPORT with all sections fully developed. Do not leave any sections incomplete or with placeholder text. The final report should be comprehensive (equivalent to 9-10 pages), evidence-based, properly formatted in markdown, and suitable for presentation to healthcare policymakers.
    """)
    
    return agent_output

# Update the historical healthcare context prompt to generate more comprehensive content
def generate_integrated_report(state="California"):
    """
    Generates a comprehensive integrated report that combines COVID-19 impact analysis
    with historical healthcare system data, and adds recommendations and conclusion.
    
    Args:
        state: The state to analyze (default: California)
        
    Returns:
        Comprehensive integrated report
    """
    # Initialize Model
    model_id = "xai/grok-2-1212"
    model = LiteLLMModel(model_id=model_id, api_key=os.getenv("XAI_API_KEY"))
    
    # First, run COVID-19 analysis
    print("\n🔍 **Running COVID-19 Analysis**")
    covid_analysis_result = run_covid_analysis(state)
    emergingchallenges_pdf_agent = ToolCallingAgent(tools=[extract_emergingchallenges_pdf], model=model,name="emergingchallenges_pdf_agent",description="Analyzes potential isssues for emerging chanllenges in the health sector for US")
    
    web_search_agent = ToolCallingAgent(
        tools=[web_search_emergingchallanges], model=model,
        name="web_search_agent",
        description="Web-searching potential issues for emerging challenges in the health sector for US"
    )
    
    # Fetch Web Content Agent
    fetch_web_content_agent = ToolCallingAgent(
        tools=[fetch_web_content], model=model,
        name="fetch_web_content_agent",
        description="Fetches detailed content from web sources related to emerging healthcare challenges."
    )
    
    # Create and run specialized agents for historical healthcare data
    hospital_beds_agent = ToolCallingAgent(
        tools=[analyze_hospital_beds], 
        model=model,
        name="hospital_beds_agent",
        description="Analyzes Community hospital bed availability trends"
    )
    
    emergency_visits_agent = ToolCallingAgent(
        tools=[analyze_emergency_visits], 
        model=model,
        name="emergency_visits_agent",
        description="Analyzes Emergency department visits trends for US"
    )
    
    hospital_utilization_agent = ToolCallingAgent(
        tools=[extract_hospital_utilization], 
        model=model,
        name="hospital_utilization_agent",
        description="Analyzes hospital utilization trends for US"
    )
    
    # Create manager agent to generate historical healthcare context
    healthcare_emerging_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[hospital_beds_agent, emergency_visits_agent, hospital_utilization_agent,
                        emergingchallenges_pdf_agent, web_search_agent, fetch_web_content_agent],
        additional_authorized_imports=["time", "numpy", "pandas", "pypdf", "os"]
    )
    
    # Run the historical context agent to create a more comprehensive historical healthcare context section
    print("\n🔍 **Generating Historical Healthcare Context Section**")
    healthcare_emerging_context_section = healthcare_emerging_agent.run(f"""
    You are an expert healthcare data analyst tasked with creating a detailed section on historical healthcare system data for {state}. This will form a critical part of a 20-page comprehensive report.
    
    You will have access to:
    1. Historical hospital bed availability trends (via hospital_beds_agent)
    2. Emergency department visit patterns (via emergency_visits_agent)
    3. Hospital utilization research findings (via hospital_utilization_agent)
    4. Emerging Challenges (via emergingchallenges_pdf_agent, web_search_agent and fetch_web_content_agent)
    
    Follow these steps:
    
    1. First, use hospital_beds_agent to analyze hospital bed availability trends
    2. Next, use emergency_visits_agent to analyze emergency department visit patterns
    3. Then, use hospital_utilization_agent to analyze hospital utilization research findings
    4. Then use emergingchallenges_pdf_agent to research emerging challenges in healthcare
    5. Use web_search_agent with the query "{state} healthcare system historical trends and challenges" to find state-specific information
    6. Use fetch_web_content_agent to get more detail on the most relevant search results
    
    Create TWO detailed sections:
    
    1. A "Historical Healthcare System Context" section that synthesizes findings from steps 1-3
       - Include detailed analysis of hospital bed trends over time
       - Analyze emergency department utilization patterns
       - Examine overall hospital utilization trends
       - Discuss how these trends relate specifically to {state}
       - Minimum 4-5 substantial paragraphs (equivalent to 3-4 pages - at least 3000 words)
    
    2. An "Emerging Challenges" section that synthesizes findings from steps 4-6
       - Identify key emerging challenges in healthcare delivery
       - Analyze staffing shortages and their impacts
       - Examine potential hospital bed shortages
       - Discuss how these challenges specifically affect {state}
       - Minimum 4-5 substantial paragraphs (equivalent to 3 pages - - at least 2500 words)
    
    Your output should be formatted as markdown sections with the headers:
    "## Historical Healthcare System Context"
    "## Emerging Challenges"
    
    Each section should be extremely comprehensive, data-driven, and equivalent to 3-4 pages of a report.
    """)

    # Create the final agent to combine results and add recommendations and conclusion
    final_report_agent = CodeAgent(
        tools=[],
        model=model,
        additional_authorized_imports=["time", "numpy", "pandas"]
    )

    # Update the final report integration prompt to ensure correct structure and comprehensive recommendations
    print("\n🔍 **Generating Final Integrated Report with Recommendations and Conclusion**")
    integrated_report = final_report_agent.run(f"""
    You are an expert healthcare data analyst tasked with integrating a COVID-19 impact analysis with historical healthcare system data for {state}, and adding comprehensive recommendations and conclusion sections. The final report must be equivalent to a 20-page document.
    
    You have two main inputs:
    1. The COVID-19 analysis (approximately 9-10 pages)
    2. The Historical Healthcare System and Emerging Challenges context sections (approximately 6-8 pages)
    
    Your task is to:
    1. Combine these inputs into one cohesive report following the EXACT order specified below
    2. Add two new comprehensive sections: "Recommendations" and "Conclusion"
    
    CRITICAL: Follow this EXACT structure in your final report (section names must match exactly):
    
    # COVID-19 Impact on Healthcare in {state}: Comprehensive Analysis
    
    ## Executive Summary
    [Keep this exactly as provided in the COVID-19 analysis]
    
    ## Introduction
    [Keep this exactly as provided in the COVID-19 analysis]
    
    ## Pandemic Timeline and Healthcare Response
    [Keep this exactly as provided in the COVID-19 analysis]
    
    ## Comparative Analysis: Pre-Pandemic vs. Pandemic Healthcare
    [Keep this exactly as provided in the COVID-19 analysis]
    
    ## Social Determinants and COVID-19 Impact
    [Keep this exactly as provided in the COVID-19 analysis]
    
    ## Healthcare Provider Availability
    [Keep this exactly as provided in the COVID-19 analysis]
    
    ## Long-Term Implications
    [Keep this exactly as provided in the COVID-19 analysis]
    
    ## Historical Healthcare System Context
    [Insert this content exactly as provided in the historical context section]
    
    ## Emerging Challenges
    [Insert this content exactly as provided in the emerging challenges section]
    
    ## Recommendations
    [Create a new comprehensive section with detailed, actionable recommendations based on all previous sections. This should be a minimum of 5-6 substantial paragraphs, equivalent to 2-3 pages.]
    
    ## Conclusion
    [Create a new comprehensive conclusion that synthesizes all key findings and reinforces the most critical points. This should be a minimum of 3-4 substantial paragraphs, equivalent to 1-2 pages.]
    
    The COVID-19 analysis is:
    {covid_analysis_result}
    
    The Historical Healthcare System and Emerging Challenges Context sections are:
    {healthcare_emerging_context_section}
    
    IMPORTANT GUIDELINES:
    - Maintain ALL section headings exactly as specified
    - Do not modify the content of the provided sections
    - Ensure your new Recommendations and Conclusion sections are comprehensive and data-driven
    - Format everything in proper markdown with clear heading hierarchy
    - Your final report should be equivalent to approximately 20 pages (Executive Summary through Conclusion)
    - Ensure seamless transitions between all sections
    """)
    
    # [rest of the code remains the same]
    
    return integrated_report

if __name__ == "__main__":
    state = "Massachusetts"  # Can be changed to any US state
    report = generate_integrated_report(state)