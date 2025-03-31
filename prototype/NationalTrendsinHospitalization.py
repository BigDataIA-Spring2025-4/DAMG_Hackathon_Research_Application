import os
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader
from smolagents import CodeAgent, tool
from smolagents import LiteLLMModel

# Define the directory containing the PDF file
DATA_DIRECTORY = "./prototype/hospitalization_trends_data"

@tool
def read_pdf_HospitalUtilizationfile() -> str:
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


# Run agent
agent_output = agent.run("""
I have provided a local PDF file in the directory. 
Extract the content from the first three pages and summarize key insights.
""")

print("Extracted PDF Content:")
print(agent_output)

@tool
def summarize_findings() -> str:
    """Compiles insights from the three analysis agents into a research summary."""
    summary = (
        f"### Final Research Summary ###\n\n"
        f"1️⃣ **Hospitalization Trends:**\n{result1}\n\n"
        f"2️⃣ **ICU Admissions Analysis:**\n{result2}\n\n"
        f"3️⃣ **Mortality Trends:**\n{result3}\n\n"
        f"📌 Conclusion: Based on the analysis, hospitalization rates are fluctuating, ICU admissions have a notable trend, and mortality rates vary across states."
    )
    return summary

model = LiteLLMModel(model_id="xai/grok-2-1212", api_key=os.getenv("XAI_API_KEY"))
final_agent = CodeAgent(
    tools=[summarize_findings],
    model=model,
    max_steps=5,
    verbosity_level=2,
)

final_report = final_agent.run("Summarize the research findings.")
print(final_report)
