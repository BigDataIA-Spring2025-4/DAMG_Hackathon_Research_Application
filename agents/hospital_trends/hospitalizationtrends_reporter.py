import os
import pandas as pd
from smolagents import CodeAgent, tool
from smolagents import LiteLLMModel

# Define the directory
DATA_DIRECTORY = "./agents/hospital_trends/data/"

@tool
def analyze_file_dqs_community_hospitalbeds() -> str:
    """
    Analyzes the CSV file and provides insights on hospital beds per 1,000 residents by state.
    Data Source: American Hospital Association (AHA) Annual Survey of Hospitals.
    """
    file_path = os.path.join(DATA_DIRECTORY, "DQS_Community_hospital_beds__by_state__United_States.csv")
    
    if not os.path.exists(file_path):
        return f"File not found. Update the path to: {file_path}"

    # Read the CSV file
    data = pd.read_csv(file_path)

    # Filter out national data, keep state-level insights
    state_data = data[data['SUBGROUP'] != 'United States']

    # Ensure numeric values
    state_data['ESTIMATE'] = pd.to_numeric(state_data['ESTIMATE'], errors='coerce')
    state_data['TIME_PERIOD'] = pd.to_numeric(state_data['TIME_PERIOD'], errors='coerce')

    # Compute percentage change in hospital beds over time
    state_data['Percent_Change'] = state_data.groupby('SUBGROUP')['ESTIMATE'].pct_change() * 100

    # Summary table
    summary = state_data.groupby('SUBGROUP').agg(
        Average_Annual_Change=('Percent_Change', 'mean'),
        Total_Change=('ESTIMATE', lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100 if len(x) > 1 else 0),
        Start_Year=('TIME_PERIOD', 'min'),
        End_Year=('TIME_PERIOD', 'max'),
        Start_Value=('ESTIMATE', 'first'),
        End_Value=('ESTIMATE', 'last')
    ).reset_index()

    # Filter for Illinois only (optional)
    parameter_state = "Illinois"
    state_summary = summary[summary['SUBGROUP'] == parameter_state]

    if state_summary.empty:
        return "No data found for Illinois."
    
    return state_summary.to_string(index=False)

# Initialize the model
model = LiteLLMModel(model_id="xai/grok-2-1212", api_key=os.getenv("XAI_API_KEY"))

# Create the agent
agent = CodeAgent(
    tools=[analyze_file_dqs_community_hospitalbeds],
    model=model,
    max_steps=10,
    additional_authorized_imports=['pandas', 'os'],
    verbosity_level=2,
)

# Run agent
agent_output = agent.run(""""
"I have provided a csv file and the analysis that needs to be done over for the selected state via the same code in the mentioned tool.
STRICTLY Follow the same code and provide a small anlaysis report for 1 page from the provided dataset:
""")

# Print the output
print("Report:")
print(agent_output)