import os
from pypdf import PdfReader
from smolagents import CodeAgent, tool
from smolagents import LiteLLMModel

# Define the directory containing the PDF file
DATA_DIRECTORY = "./agents/hospital_trends/data/"

@tool
def read_pdf_emergencydepartment_visitsfile() -> str:
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

# Initialize the model
model = LiteLLMModel(model_id="xai/grok-2-1212", api_key=os.getenv("XAI_API_KEY"))

# Create the agent
agent = CodeAgent(
    tools=[read_pdf_emergencydepartment_visitsfile],
    model=model,
    max_steps=10,
    additional_authorized_imports=['os', 'pypdf'],
    verbosity_level=2,
)

# Run agent
agent_output = agent.run("""
I have provided a local PDF file in the directory. 
Extract the content from the first three pages and summarize key insights.
""")

# Print the extracted content
print("Extracted PDF Content:")
print(agent_output)
