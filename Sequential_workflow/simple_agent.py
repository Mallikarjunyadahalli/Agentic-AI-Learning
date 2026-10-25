from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from random import randint

class ReviewState(TypedDict):
    text: str = ""
    score: int = 0
    comment: str = ""

def score(state: ReviewState) -> ReviewState:
    state["score"] = randint(1, 10)

    return state

def comment(state: ReviewState) -> ReviewState:
    if state["score"] > 5:
        state['comment'] = "Well done."

    else:
        state['comment'] = "Improve yourself."

    return state


graph = StateGraph(ReviewState)

graph.add_node("score", score)
graph.add_node("comment", comment)

graph.add_edge(START, "score")
graph.add_edge("score", "comment")
graph.add_edge("comment", END)

final_graph = graph.compile()

print(final_graph)

response = final_graph.invoke({})

print(response)