import os
import snowflake.connector
from dotenv import load_dotenv
from smolagents import CodeAgent, tool
from smolagents.agents import ActionStep
from smolagents import LiteLLMModel

# Load environment variables
load_dotenv()

@tool
def query_snowflake(query: str) -> str:
    """
    Executes a SQL query against Snowflake and returns the result.
    
    Args:
        query (str): The SQL query to execute.
    
    Returns:
        str: The query result in text format.
    """
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )
        
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()
        
        return f"Query executed successfully. Results:\n{results}"
    
    except Exception as e:
        return f"Error executing query: {str(e)}"

# Initialize the model
model_id = "xai/grok-beta"
model = LiteLLMModel(model_id=model_id, api_key=os.getenv("XAI_API_KEY"))
parameter_state="Illinois"

# Create the agent
agent = CodeAgent(
    tools=[query_snowflake],
    model=model,
    max_steps=10,
    additional_authorized_imports=['pandas'],
    verbosity_level=2,
)


# Run the agent with the query
agent_output = agent.run(f"""
I have provided you my snowflake connectivity credentials use them and execute query in Snowflake Table 
Use this query to get the details about the numbers of year-over-year COVID-19 cases and deaths for U.S. states
Understand the numbers and provide me a comprehensive detail report for each year's dataset from the output:
WITH state_agg AS 
(
    SELECT 
        EXTRACT(YEAR FROM date) AS year, 
        state, 
        MAX(cases) AS cases, 
        MAX(deaths) AS deaths 
    FROM COVID19_GLOBAL_DATA_ATLAS.HLS_COVID19_USA.COVID19_USA_CASES_DEATHS_BY_STATE_DAILY_NYT
    WHERE state='{parameter_state}'
    GROUP BY EXTRACT(YEAR FROM date), state
    ORDER BY state, year)
SELECT year, state,
    cases - LAG(cases, 1, 0) OVER (PARTITION BY state ORDER BY state, year) as cases,
    deaths - LAG(deaths, 1, 0) OVER (PARTITION BY state ORDER BY state, year) as deaths,    
FROM state_agg;""")

# Print the output
print("Final output:")
print(agent_output)

