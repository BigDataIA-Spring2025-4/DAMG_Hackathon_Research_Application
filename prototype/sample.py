
import os
import json
import snowflake.connector
import requests
from dotenv import load_dotenv
from smolagents import CodeAgent, tool
from smolagents.agents import ActionStep
from smolagents import LiteLLMModel

import pandas as pd
from typing import Dict, Any, List, Optional
from tavily import TavilyClient

# Load environment variables
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


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
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY"
}
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT LOC_ADMIN_STATE, COUNT(*) AS PROVIDER_COUNT 
        FROM COVID19_GLOBAL_DATA_ATLAS.HLS_COVID19_USA.COVID_19_US_VACCINATING_PROVIDER_LOCATIONS
        """
        
        if state:
            state = us_states[state]
            print("State is : --> ", state)
            query += f" WHERE LOC_ADMIN_STATE = '{state}'"
        
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
        
        # Set the database
        # cursor.execute("USE DATABASE DIVERSITY_EQUITY_AND_INCLUSION__ACCESS_TO_HEALTHCARE")
        
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
        
        # if state:
        #     query1 += f" AND STATE = '{state}'"
            
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
        
        # if state:
        #     query2 += f" AND STATE = '{state}'"
            
        query2 += " ORDER BY YEAR DESC"
        
        cursor.execute(query2)
        results2 = cursor.fetchall()
        column_names2 = [desc[0] for desc in cursor.description]
        df2 = pd.DataFrame(results2, columns=column_names2)
        results["physician_visits"] = json.loads(df2.to_json(orient="records"))
        
        # # Query 3: Delayed healthcare due to cost
        # query3 = "SELECT * FROM DIVERSITY_EQUITY_AND_INCLUSION__ACCESS_TO_HEALTHCARE.DEI_HEALTHCARE.\"Delayed Healthcare Due to Cost - USA\""
        
        # # if state:
        # #     query3 += f" WHERE STATE = '{state}'"
        
        # cursor.execute(query3)
        # results3 = cursor.fetchall()
        # column_names3 = [desc[0] for desc in cursor.description]
        # df3 = pd.DataFrame(results3, columns=column_names3)
        # results["delayed_healthcare"] = json.loads(df3.to_json(orient="records"))
        
        # Query 4: Delayed healthcare by year
        query4 = """
        SELECT YEAR, COUNT(*) AS COUNT 
        FROM DIVERSITY_EQUITY_AND_INCLUSION__ACCESS_TO_HEALTHCARE.DEI_HEALTHCARE."Delayed Healthcare Due to Cost - USA"
        """
        
        # if state:
        #     query4 += f" WHERE STATE = '{state}'"
            
        query4 += " GROUP BY YEAR"
        
        cursor.execute(query4)
        results4 = cursor.fetchall()
        column_names4 = [desc[0] for desc in cursor.description]
        df4 = pd.DataFrame(results4, columns=column_names4)
        results["delayed_healthcare_by_year"] = json.loads(df4.to_json(orient="records"))
        
        close_connection(cursor, conn)
        return json.dumps(results)
    
    except Exception as e:
        return f"Error executing healthcare access query: {str(e)}"
    


    client = TavilyClient(TAVILY_API_KEY)
@tool
def web_search(query: str) -> str:
    """
    Searches the web for COVID-related information using Tavily api.
    
    Args:
        query (str): The search query, should be related to COVID-19, or healthcare in general.
    
    Returns:
        str: JSON string containing search results.
    """
    try:
        response = client.search(
        query=query
        )
        return response
    
    except Exception as e:
        print(f"Error in web search: {str(e)}")
        # Return empty results on error
        return json.dumps({"results": []})
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"
    

# urls = ["https://www.nytimes.com/interactive/2021/us/covid-cases.html"]
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
        response = client.extract(
            urls=url
        )
        return response["results"][0]["raw_content"]
    
    except Exception as e:
        print(f"Error in fetch web content: {str(e)}")
        # Return empty results on error
        return f"This is mock content about COVID-19 research and data analysis."
        
    except Exception as e:
        return f"Error fetching web content: {str(e)}"
    


model_id = "xai/grok-2-1212"
model = LiteLLMModel(model_id=model_id, api_key=os.getenv("XAI_API_KEY"))

def run_covid_analysis(state="Illinois"):
    """
    Main function to run the COVID-19 data analysis for a specified state.
    
    Args:
        state: The state to analyze (default: Illinois)
    
    Returns:
        Agent output containing the comprehensive report
    """
    # Initialize the model
    # model_id = "xai/grok-beta"
    # model = LiteLLMModel(model_id=model_id, api_key=os.getenv("XAI_API_KEY"))
    
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
        max_steps=20,  # Increased for more complex workflow
        additional_authorized_imports=['pandas', 'json', 'requests', 'matplotlib', 'seaborn'],
        verbosity_level=2,
    )
    
    # Run the agent with a comprehensive prompt
    agent_output = agent.run(f"""
    You are a COVID-19 data analyst tasked with generating a comprehensive report on how the COVID-19 pandemic transformed healthcare in {state}.

    IMPORTANT GUIDELINES:
    - Only retry a tool call at most 3 times. If still not resolved, continue with existing data.
    - Process data one section at a time to minimize token usage.
    - For each section of the report, collect and analyze only the data needed for that section.
    - Clearly indicate when you're moving to a new section of the report.
    
    Follow this step-by-step approach:

    1. EXECUTIVE SUMMARY & INTRODUCTION:
       - Start by querying basic COVID data for context: query_covid_cases_by_year("{state}")
       - Search the web for an overview: web_search("{state} COVID-19 healthcare impact overview")
       - Write the Executive Summary and Introduction sections.

    2. PANDEMIC TIMELINE AND HEALTHCARE RESPONSE:
       - Analyze the COVID cases/deaths data already collected
       - Create this section based on the yearly trends.

    3. COMPARATIVE ANALYSIS:
       - Query healthcare access data: query_healthcare_access("{state}")
       - Focus on emergency department and physician visits data
       - Search for additional context: web_search("{state} pre-pandemic vs pandemic healthcare expenditure")
       - Complete this section using the collected data.

    4. SOCIAL DETERMINANTS AND COVID-19 IMPACT:
       - Analyze the delayed healthcare data from the healthcare_access query
       - Search for relevant information: web_search("{state} social determinants health COVID hotspots")
       - Complete this section with the findings.

    5. HEALTHCARE PROVIDER AVAILABILITY:
       - Query vaccination provider data: query_vaccine_providers("{state}")
       - Search for context: web_search("{state} healthcare provider availability COVID impact")
       - Complete this section with the collected data.

    6. LONG-TERM IMPLICATIONS:
       - Search for forecasting information: web_search("{state} long-term effects COVID healthcare access")
       - Complete this section based on search results.

    7. RECOMMENDATIONS AND CONCLUSION:
       - Draw on all previous analysis to formulate data-driven recommendations
       - Summarize key findings in the conclusion.

    The final report should follow this structure:
    
    # COVID-19 Impact on Healthcare in {state}: Comprehensive Analysis
    
    ## Executive Summary
    [Concise overview of key findings]
    
    ## Introduction
    [Context of COVID-19 in {state}, scope of the report]
    
    ## Pandemic Timeline and Healthcare Response
    [Use cases/deaths data to outline pandemic waves and response]
    
    ## Comparative Analysis: Pre-Pandemic vs. Pandemic Healthcare
    [Compare healthcare metrics before and during pandemic]
    
    ## Social Determinants and COVID-19 Impact
    [Identify vulnerable populations and impact patterns]
    
    ## Healthcare Provider Availability
    [Analyze vaccination provider distribution and access]
    
    ## Long-Term Implications
    [Forecast effects on healthcare access and policy implications]
    
    ## Recommendations
    [Data-driven recommendations for healthcare system resilience]
    
    ## Conclusion
    [Summarize key insights and future outlook]
    
    Make the report comprehensive (equivalent to 2-3 pages), evidence-based, and insightful for healthcare policymakers. Format with proper markdown, including headings, subheadings, bullet points, and emphasis where appropriate.
    """)
    
    return agent_output



output = run_covid_analysis()