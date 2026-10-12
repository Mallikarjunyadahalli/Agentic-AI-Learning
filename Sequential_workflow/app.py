
from typing import TypedDict

class pipelenestate(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)


#* Creating notes - Nodes is just like a python function
def editor_node(state: pipelenestate):
    """ stage 1 : Cleans up a grammar, removes typos, and refines the tone. """

    prompt = f""" 

           Clean the given raw text.

Rules:
- Remove extra spaces, blank lines, HTML tags, and scraping noise.
- Fix broken line breaks and formatting.
- Remove obvious duplicate text.
- Preserve headings, lists, names, dates, numbers, URLs, code, and important facts.
- Do not summarize, rewrite, or add information.
- Do not change the original meaning.

Return ONLY the cleaned text.

Raw text:

 {state['raw_input']}

 """


    response = llm.invoke(prompt)

    return {"edited_text" : response.content.strip()}


def scriptwritter_node(state: pipelenestate):
    """ Stage 2 : Formates the clean text into an engaging video script style. """
    print("\n ----- [Stage 2 Executing Script writter Node ----]")

    prompt = f""" 

          Convert the given clean text into a short, engaging video script.

Rules:
- Start with a strong hook.
- Use simple, natural spoken language.
- Organize it logically: Hook → Introduction → Main Content → Conclusion.
- Keep all important facts and meaning.
- Do not invent information.
- Do not unnecessarily repeat content.
- Write for narration, not like an article.

Return ONLY the final video script.

Clean text:

{state['edited_text']}    
    

 """

    response = llm.invoke(prompt)
    return {"script_text" : response.content.strip()} 

def translater_node(state: pipelenestate):
    """ Stage 3 : Trnslate the script text into natural Hinglish. """

    print("\n ---- [stage 3] Executing english translator node ----")

    prompt = f"""

            Convert the given text into natural, conversational Hinglish.

Rules:
- Use a natural mix of Hindi and English, like people commonly speak in India.
- Keep important English technical terms in English.
- Preserve the original meaning, facts, names, numbers, and examples.
- Do not add or remove information.
- Do not translate word-by-word.
- Make it sound natural when spoken aloud.
- Keep the language simple and easy to understand.
- Do not use overly formal Hindi.

Return ONLY the final Hinglish text.

Text:
{state['script_text']}

"""


    response = llm.invoke(prompt)

    return {"final_output" : response.content.strip()}



# * now our state and nodes are ready and now time to create graph.


from langgraph.graph import StateGraph, START, END

#* Creating the graph
graph = StateGraph(pipelenestate)

# * add nodes in our graph
graph.add_node("editor", editor_node)
graph.add_node("scriptwritter", scriptwritter_node)
graph.add_node("translator", translater_node)


#* Add edges (Sequantial - one after one)
graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwritter")
graph.add_edge("scriptwritter", "translator")
graph.add_edge("translator", END)



# compile the graph

app = graph.compile()

result = app.invoke({"raw_input" : """   Artificial Intelligence (AI) is   changing the way we live and work.

AI can analyze large amounts of data, identify patterns, and make predictions.  For example, AI is used in healthcare to help doctors detect diseases, in banking to detect fraud, and in recommendation systems to suggest movies and products.

  However, AI is not perfect. It can sometimes make mistakes or produce incorrect results. That's why human supervision is still important.

AI is becoming a powerful tool, but the goal is not to replace humans completely. Instead, it can help humans work faster, make better decisions, and solve complex problems. """})


print("Your result are : - \n\n")
print(result['final_output'])

