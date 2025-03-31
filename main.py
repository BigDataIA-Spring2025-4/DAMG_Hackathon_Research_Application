import os
from smolagents import CodeAgent, LiteLLMModel
from agents.hospital_trends.hospitalutilization import read_pdf_HospitalUtilizationfile
from agents.hospital_trends.hospitalizationtrends_reporter import analyze_file_dqs_community_hospitalbeds
from agents.hospital_trends.emergencydepartment_visits import read_pdf_emergencydepartment_visitsfile
from agents.hospital_trends.summary_agent import summarize_findings

# Initialize model
model = LiteLLMModel(model_id="xai/grok-2-1212", api_key=os.getenv("XAI_API_KEY"))

# Create and run individual agents
community_hospitalbeds_agent = CodeAgent(tools=[analyze_file_dqs_community_hospitalbeds], model=model)
emergencydepartment_visitsfile_agent = CodeAgent(tools=[read_pdf_emergencydepartment_visitsfile], model=model)
HospitalUtilizationfilepdf_agent = CodeAgent(tools=[read_pdf_HospitalUtilizationfile], model=model)

dqs_community_hospitalbeds_result = community_hospitalbeds_agent.run("Analyze the hospital beds dataset.")
emergencydepartment_visitsfile_agent_result = emergencydepartment_visitsfile_agent.run("Analyze emergency department visits.")
HospitalUtilizationfilepdf_agent_result = HospitalUtilizationfilepdf_agent.run("Extract key insights from the research paper.")

# Run the summary agent
summary_agent = CodeAgent(tools=[summarize_findings], model=model)
final_report = summary_agent.run(f"Summarize the research findings using:\n{dqs_community_hospitalbeds_result}\n{emergencydepartment_visitsfile_agent_result}\n{HospitalUtilizationfilepdf_agent_result}")

print("\n🔍 **Final Research Summary:**")
print(final_report)
