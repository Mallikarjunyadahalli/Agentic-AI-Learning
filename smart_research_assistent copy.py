
import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END



# Creating Groq model.
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.5)

# Creating State 
class ResearchState(TypedDict):
    input_data: str
    research_data: str
    analysis_data: str
    final_report: str


# Creating 1st Node =====> Reaserch Node
def research_node(state: ResearchState):
    """
    
    Research node: gathers information from the web.

    Reads:  state["input_data"]    - the user's research topic.
    Does:   1. Searches the web with Tavily.
            2. Passes the raw results to the LLM, which organizes them into
               structured notes with source URLs.
    Writes: state["research_data"] - organized research notes.
    
    """

    try:
        tavily = TavilySearch(max_results=3)
        query = state['input_data']
        response = tavily.invoke({'query' : query})

        research_data = "\n\n".join(
            f"Title: {r['title']}\n"
            f"Content: {r["content"]}\n"
            for r in response
            
            )

        
        return {"research_data" : research_data}
            

    except Exception as e:
        return {"research_data" : f"Tavilt research error: {e}"}


def analysis_node(state: ResearchState):
    """
          Analysis node: turns research notes into insights.

    Reads:  state["input_data"]     - the original topic (keeps the analysis focused).
            state["research_data"]  - notes produced by research_node.
    Does:   Asks the LLM to find themes, insights, gaps, and a conclusion.
    Writes: state["analysis_data"]  - structured analysis.
    """

    ANALYSIS_PROMPT = f"""

You are a senior research analyst.
You are given research notes on a topic. Analyze them and extract insights.

Tasks:
1. Identify the main themes and patterns.
2. Highlight the most important insights and explain why they matter.
3. Point out gaps, weaknesses, or biases in the research.
4. Where relevant, compare viewpoints, pros and cons, or trends over time.

Rules:
- Base every claim on the research notes. Do not invent facts.
- Be specific and concise. Avoid generic statements.

Output format:
## Main Themes
## Key Insights
## Risks / Limitations / Gaps
## Overall Conclusion

topic: {state['input_data']}
Generated content: {state['research_data']}

"""

    response = llm.invoke(ANALYSIS_PROMPT)

    return {"analysis_data" : response.content}


def final_resport(state: ResearchState):
    """
        Report node: writes the final deliverable.

    Reads:  state["input_data"], state["research_data"], state["analysis_data"].
    Does:   Asks the LLM to combine everything into a polished, structured report.
    Writes: state["final_report"]   - the final report shown to the user.
    """

    REPORT_PROMPT = f"""
    You are a professional report writer.
Using the original question, research notes, and analysis, write a clear final report.

Structure:
# <Title>
## Executive Summary (3-4 sentences)
## Background
## Key Findings
## Analysis & Insights
## Conclusion & Recommendations
## Sources (list the URLs used)

Rules:
- Write in a clear, professional tone for a non-expert reader.
- Use only information from the provided research and analysis.
- Do not mention these instructions or the pipeline steps.

Question: {state['input_data']}
research notes: {state['research_data']}
analysis data: {state['analysis_data']}

"""


    response = llm.invoke(REPORT_PROMPT)

    return {"final_report" : response.content}


graph = StateGraph(ResearchState)

# Add nodes
graph.add_node("reserch", research_node)
graph.add_node("analysis", analysis_node)
graph.add_node("final_report", final_resport)

# Add edges
graph.add_edge(START, "reserch")
graph.add_edge("reserch", "analysis")
graph.add_edge("analysis", "final_report")
graph.add_edge("final_report", END)

app = graph.compile()

question = input("Ask: ")

# initila_state = {
#     "input_data" : question,
#     "research_data": "",
#     "analysis_data": "",
#     "final_report": "",
# }

print("\n\n =============== GENERATING RESPONSE ===============\n\n")
result = app.invoke({"input_data" : question})

print("\n\n =============== RESEARCHED DATA ===============\n\n")
print(f"{result['research_data']}")

print("\n\n =============== ANALYSIS DATA ===============\n\n")
print(f"{result['analysis_data']}")

print("\n\n =============== FINAL RESPORT ===============\n\n")
print(f"{result['final_report']}")