import os
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

def merge_score(existing: dict, new_update: str) -> dict:
    if existing is None:
        return new_update

    return {**existing, **new_update}

class AnalyserState(TypedDict):
    raw_text: str
    safety_Score: Annotated[dict[str, int], merge_score]


def toxicity_node(state: AnalyserState):
    print("\n 😁 [Branch 1] Analysing Toxicity.... \n")
    prompt = f""" 
            Analyse the following text for profenity, aggresion, hate speech, or toxicity. Provide a score from 0 to 100,  where 0 means perfectly clean and 100 means highlt toxic.

            Rturn only the plain integer, nothing else. \n\n
            Text: \n' f{state['raw_text']}
 """
    

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())

    except ValueError:
        score = 0

    return {"safety_Score" : {"toxicity_level" : score}}


def copywrite_node(state: AnalyserState):
    print("\n 😁 [Branch 2] Analysing Copywrite Risks.... \n")
    prompt = f""" 
            Analyse the following text Judge if it sounds havily plagiarized, unoriginal, or presents a cotparate trademark risk. Provide from 0 to 100,
            where 0 means entirely original and 100 means high risk.

            Return Onlt the plain integer number, nothing else
            Text: \n{state['raw_text']}
 """

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())

    except ValueError:
        score = 0

    return {"safety_Score" : {"copyright_score" : score}}


def culture_node(state: AnalyserState):
    print("\n 😁 [Branch 3] Analysing Riginal and cultural Sensitivity.... \n")
    prompt = """ 
            Analyse the following text for reginal sensitivity, Political landmiss, or cultural insensitivity that might offerd a global audience. Provide a score from 0 to 100. 

            where 0 means completely safe and 100 means highly offecive.


            Return ONLY the plain integer number, nothing else.

            Text: \n{state['raw_text']}
 """

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())

    except ValueError:
        score = 0

    return {"safety_Score" : {"culture_node" : score}}


graph = StateGraph(AnalyserState)

graph.add_node("toxicity_node", toxicity_node)
graph.add_node("copywrite_node", copywrite_node)
graph.add_node("culture_node", culture_node)

graph.add_edge(START, "toxicity_node")
graph.add_edge(START, "copywrite_node")
graph.add_edge(START, "culture_node")

graph.add_edge("toxicity_node", END)
graph.add_edge("copywrite_node", END)
graph.add_edge("culture_node", END)

final_graph = graph.compile()

sample_scipt = """ 
Yo guys! Welcome back to the stream. Today I am going to show you how to hack into your friend's system using a script I copied directly from an online forum. Honestly, traditional security protocols are absolute garbage and anyone still using them is an absolute idiot. Let's dive into the code !.

 """

initial_sate = {
    "raw_text" : sample_scipt,
    "safety_score" : {}
}


final_state = final_graph.invoke(initial_sate)

print(final_state['safety_Score'])