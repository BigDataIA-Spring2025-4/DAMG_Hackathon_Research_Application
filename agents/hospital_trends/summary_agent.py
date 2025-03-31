from smolagents import tool

@tool
def summarize_findings(
    dqs_community_hospitalbeds_result: str, 
    emergencydepartment_visitsfile_agent_result: str, 
    HospitalUtilizationfilepdf_agent_result: str
) -> str:
    """
    Compiles insights from hospitalization trends, emergency department visits, and hospital utilization reports.

    Args:
        dqs_community_hospitalbeds_result (str): Analysis of community hospital beds.
        emergencydepartment_visitsfile_agent_result (str): Analysis of emergency department visits.
        HospitalUtilizationfilepdf_agent_result (str): Summary of the hospital utilization research paper.

    Returns:
        str: A consolidated research summary combining all insights.
    """
    summary = f"""
    ### Final Research Summary ###
    
    🏥 **Hospital Bed Trends (CSV Data)**:
    {dqs_community_hospitalbeds_result}
    
    🚑 **Emergency Department Visits (Excel Data)**:
    {emergencydepartment_visitsfile_agent_result}
    
    📄 **Hospital Utilization Research (PDF Data)**:
    {HospitalUtilizationfilepdf_agent_result}
    
    📌 **Conclusion**:
    - Hospital bed availability varies significantly by state.
    - Emergency department visits show trends across demographics.
    - The research paper provides additional insights into hospitalization utilization.
    """
    return summary
