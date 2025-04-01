import json
import os
import pandas as pd
from pypdf import PdfReader
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, tool
from tavily import TavilyClient


# Define the directory containing the data files
DATA_DIRECTORY = "./agents/emerging_challenges/data"

## 1️⃣ Agent: Websearch tool
client = TavilyClient(os.getenv("TAVILY_API_KEY"))
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
    

### 3️⃣ Agent: Extract Hospital Utilization Data (PDF)
DATA_DIRECTORY="./agents/emerging_challenges/data"
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

### **🔹 Running the Agents to Generate the Final Report**
if __name__ == "__main__":
    # Initialize Model
    model = LiteLLMModel(model_id="xai/grok-2-1212", api_key=os.getenv("XAI_API_KEY"))

    # Create and run agents
    emergingchallenges_pdf_agent = ToolCallingAgent(tools=[extract_emergingchallenges_pdf], model=model,name="emergingchallenges_pdf_agent",description="Analyzes potential isssues for emerging chanllenges in the health sector for US")
    
    web_search_agent = ToolCallingAgent(
        tools=[web_search_emergingchallanges], model=model,
        name="web_search_agent",  # ✅ Changed name to avoid duplicates
        description="Web-searching potential issues for emerging challenges in the health sector for US"
    )

    # Fetch Web Content Agent
    fetch_web_content_agent = ToolCallingAgent(
        tools=[fetch_web_content], model=model,
        name="fetch_web_content_agent",
        description="Fetches detailed content from web sources related to emerging healthcare challenges."
    )
    manager_agent = CodeAgent(
    model=model,
    tools=[],
    managed_agents=[web_search_agent,emergingchallenges_pdf_agent,fetch_web_content_agent],
    additional_authorized_imports=["time", "numpy", "pandas","pypdf","os"]
    )
    
    state="California"
    answer = manager_agent.run(f"""
    You are a data analyst tasked with generating a comprehensive report on how for the {state} in US 
    Research on Health Care Staffing Shortages and Potential National Hospital Bed Shortage provide insights into healthcare staffing shortages and the looming hospital bed crisis:
    Follow the tool to gather any necessary information and data, then create a detailed markdown report:
    - Rules 
        - Use all three tools avaiable to build additional context and final report generation
        - Do not generate any data use only use the mentioned resources
        - Try and relate the data to the state and how it affected the state
    Parameters:
        hospital_beds (str): Summary of hospital bed trends.

    Returns:
        str: A structured summary report.
    """)