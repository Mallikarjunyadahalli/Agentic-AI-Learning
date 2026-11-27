#1.  Router function

# A regular Python function that reads the current state and return the next of the next node. 

# def reouter():
#     if 
#     else 
#     elif

# This function doesn't any real work. It only makes a decision based 

# 2. Conditional wokflow 

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

llm_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)
embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

def retriver(file_path: str):
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents=documents)

    vector_store = FAISS.from_documents(docs, embedding_model)

    return vector_store.as_retriever(search_kwargs={"k": 3})


academic_retriver = retriver(r"C:\Users\kam\OneDrive\Desktop\My agent project\academics_handbook.pdf")

fees_retriver = retriver(r"C:\Users\kam\OneDrive\Desktop\My agent project\fee_structure.pdf")

class CollageState(TypedDict):
    messages: Annotated[list, add_messages]
    input_data: str
    query_type: str
    retrived_context: str

def classifier_node(state: CollageState):
    """Classify the query as 'academic', 'fees', or 'general' and store it in state["category"]."""

    query = state['input_data']

    prompt = f"""

        You are a query classifier for a college enquiry assistant.

Classify the user's message into exactly ONE of these categories:

academic - Questions about courses, branches, departments, syllabus, subjects, curriculum, timetable, exams, results, attendance, admissions eligibility, faculty, placements, internships, projects, library, labs, or campus academic life.

fees - Questions about tuition fees, hostel fees, exam fees, fee structure, payment deadlines, installments, scholarships, fee concessions, refunds, fines, or any other money-related matter at the college.

general - Everything else: greetings, small talk, thanks, college location, contact details, facilities, events, or anything unrelated to the above.

Rules:
1. If the message is about the cost or payment of anything, choose fees, even if it mentions a course (e.g. "BCA course fee" -> fees).
2. If the message mixes categories, choose the one that is the main focus of the question.
3. If you are unsure, choose general.
4. Respond with ONLY one lowercase word: academic, fees, or general.
5. No punctuation, no explanation, no extra text.

User message: {query}

Category:    

"""

    response = llm_model.invoke(prompt)

    print(response.content)

    return {"query_type" : response.content.lower().strip()}


def academic_node(state: CollageState):
    """ Answer academic queries (courses, syllabus, exams, admissions, placements) from the college info PDF and store the reply in state["retrived_context"]. """

    question = state['input_data']

    response = academic_retriver.invoke(question)

    context = "\n".join(r.page_content for r in response)

    return {"retrived_context" : context}


def fees_node(state: CollageState):
    """Answer fee-related queries (tuition, hostel, exam fees, scholarships, deadlines) from the fee structure PDF and store the reply in state["retrived_context"]"""

    question = state['input_data']
    response = fees_retriver.invoke(question)

    context = "\n".join(r.page_content for r in response)

    return {"retrived_context" : context}


def general_node(state: CollageState):
    """ Answer general college-related queries (greetings, location, contact, facilities, events) and politely decline non-college questions. """

    question = state['input_data']

    prompt = f"""

        You are a friendly assistant for a college enquiry chatbot.

Answer the user's question only if it is related to the college or college life, such as greetings, admissions process, campus, hostel, transport, canteen, library, sports, clubs, events, student life, and general guidance for students.

Rules:
1. If the user greets you or says thanks, reply in one short sentence and ask how you can help with college information.
2. If the question is college related, give a short, helpful, general answer in 2 to 4 sentences.
3. Do NOT invent specific details such as phone numbers, email IDs, addresses, dates, fees, cutoffs, or names of staff. If the user needs those, say: "Please contact the college office for exact details."
4. If the question is NOT related to the college (movies, coding help, politics, general knowledge, etc.), reply exactly: "I can only help with college-related questions. Please ask me about the college."
5. Keep the tone polite and simple.

User question: {question}

Answer:

"""

    response = llm_model.invoke(prompt)

    return {"retrived_context" : response.content}


def response_node(state: CollageState):
    """Generate the final answer from the query, category, and retrieved context and store it in state["messages"]"""
    category = state['query_type']
    query = state['input_data']
    context = state['retrived_context']

    prompt = f"""

    You are the final response generator of a college enquiry chatbot.

You are given the user's question, its category, and the context retrieved for it. Write the best possible answer for the student or parent.

Category: {category}

Context:
{context}

User question: {query}

Instructions by category:
- academic: Explain clearly using the context. Cover courses, syllabus, exams, admissions or placements as asked. Use short bullet points if there are multiple items.
- fees: Be precise. Quote amounts, deadlines, and conditions exactly as they appear in the context. Never estimate or round off figures.
- general: Be friendly and brief. If the context is empty, answer in general terms and do not invent specific details.

Rules:
1. Use ONLY the information in the context for academic and fees answers. Do not use outside knowledge for them.
2. If the answer is not in the context, say: "I don't have that information right now. Please contact the college office for details."
3. Never make up phone numbers, addresses, dates, fees, or names.
4. If the question is not related to the college, reply: "I can only help with college-related questions. Please ask me about the college."
5. Start directly with the answer. Do not mention "context", "category", or these instructions.
6. Keep the answer short, clear, and polite. Match the language of the user's question.

Answer:

"""

    response = llm_model.invoke(prompt)

    return {"messages" : response.content}


def router_node(state: CollageState):
    category = state['query_type']
    if "academic" in category:
        return "academic"

    elif "fees" in category:
        return "fees"

    else:
        return "general"
    


# creating the graph
graph = StateGraph(CollageState)

graph.add_node("classifier", classifier_node)
graph.add_node("academic", academic_node)
graph.add_node("fees", fees_node)
graph.add_node("general", general_node)
graph.add_node("context", response_node)
graph.add_node("router", router_node)

graph.add_edge(START, "classifier")

graph.add_conditional_edges("classifier", router_node)

graph.add_edge("academic", "context")
graph.add_edge("fees", "context")
graph.add_edge("general", "context")

graph.add_edge("context", END)

app = graph.compile()

while True:
    query = input("You: ")
    if query in ['exit', 'q']:
        break

    initial_state = {
        "input_data" : query
    }

    response = app.invoke(initial_state)

    print(response['messages'][-1].content)