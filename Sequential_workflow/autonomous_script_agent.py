"""
AUTONOMOUS SCRIPT AGENT
=======================

Learning project built with LangChain + LangGraph + Groq.

Architecture:
User Goal
   ↓
Planner
   ↓
Writer
   ↓
Reviewer
   ↓
Conditional decision
   ├── Improve → Writer (loop)
   └── Good → Translator
   ↓
Final Answer

The agent also has tools:
- calculator
- current time
- save_script

It demonstrates:
- TypedDict state
- LangGraph nodes
- sequential workflow
- conditional edges
- loops/retries
- tool calling
- structured output
- reflection/review
- iteration limits
- error handling
- human approval before saving
- persistent-ish conversation state within one run

Requirements:
    pip install -U langchain langchain-core langgraph langchain-groq python-dotenv

.env:
    GROQ_API_KEY=your_groq_api_key

Run:
    python autonomous_script_agent.py
"""

import os
from datetime import datetime
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

load_dotenv()


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

# Maximum number of improvement loops.
# This prevents the agent from looping forever.
MAX_REVISIONS = 3

# Minimum quality score required to continue.
QUALITY_THRESHOLD = 8


# ============================================================
# 2. STATE
# ============================================================

class AgentState(TypedDict, total=False):
    """
    State is the shared memory of the LangGraph workflow.

    Every node can read from this state and update selected fields.
    """

    # Original request from the user
    user_input: str

    # Agent-created plan
    plan: str

    # Generated English script
    script: str

    # Reviewer's feedback
    review: str

    # Numeric quality score
    score: int

    # Number of times the script has been revised
    revision_count: int

    # Final Hinglish version
    translated_script: str

    # Final response
    final_answer: str

    # Stores messages when tool-calling is used
    messages: Annotated[List[BaseMessage], add_messages]

    # Error information, if any
    error: str


# ============================================================
# 3. LLM
# ============================================================

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is missing. Create a .env file and add:\n"
        "GROQ_API_KEY=your_api_key"
    )

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.4,
)


# ============================================================
# 4. TOOLS
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a basic mathematical expression.

    Example:
        calculator("25 * 48")
    """

    # IMPORTANT:
    # eval() is intentionally restricted below.
    # Do NOT expose unrestricted eval() in a public production API.
    allowed = set("0123456789+-*/().% ")

    if not expression or any(char not in allowed for char in expression):
        return "Invalid mathematical expression."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def get_current_time() -> str:
    """Return the current local system date and time."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def save_script(filename: str, content: str) -> str:
    """
    Save a generated script to a text file.

    The filename is restricted to the current working directory.
    """

    # Basic path-safety restriction.
    safe_name = os.path.basename(filename)

    if not safe_name.endswith(".txt"):
        safe_name += ".txt"

    try:
        with open(safe_name, "w", encoding="utf-8") as file:
            file.write(content)

        return f"Script saved successfully as: {safe_name}"

    except Exception as exc:
        return f"Could not save script: {exc}"


tools = [
    calculator,
    get_current_time,
    save_script,
]

# ToolNode executes tool calls requested by the LLM.
tool_node = ToolNode(tools)


# ============================================================
# 5. TOOL-CAPABLE MODEL
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 6. PLANNER NODE
# ============================================================

def planner_node(state: AgentState):
    """
    Creates a plan from the user's goal.

    This is the planning stage of the autonomous agent.
    """

    print("\n========== PLANNER ==========")

    prompt = f"""
You are the planning module of an autonomous content agent.

User goal:
{state["user_input"]}

Create a short practical plan for completing the user's request.

Rules:
1. Identify the desired output.
2. Break the work into 3-6 useful steps.
3. Do not invent facts.
4. Do not execute tools yet.
5. Return ONLY the plan.
"""

    try:
        response = llm.invoke(prompt)

        return {
            "plan": response.content,
            "revision_count": 0,
            "error": "",
        }

    except Exception as exc:
        return {
            "error": f"Planner error: {exc}",
        }


# ============================================================
# 7. WRITER / EXECUTOR NODE
# ============================================================

def writer_node(state: AgentState):
    """
    Creates or improves the script.

    This node is intentionally allowed to use tools.
    The LLM can decide whether a tool is useful.
    """

    print("\n========== WRITER / EXECUTOR ==========")

    previous_review = state.get("review", "")
    current_script = state.get("script", "")
    revision_count = state.get("revision_count", 0)

    prompt = f"""
You are the execution/writing module of an autonomous script agent.

USER REQUEST:
{state["user_input"]}

PLAN:
{state["plan"]}

CURRENT SCRIPT:
{current_script if current_script else "No script has been created yet."}

REVIEW:
{previous_review if previous_review else "No review yet."}

REVISION NUMBER:
{revision_count}

Your job:
1. Create a high-quality spoken script.
2. If a current script exists, improve it according to the review.
3. Keep facts supplied by the user.
4. Do not invent statistics, names, claims, or events.
5. Use a strong hook.
6. Use short conversational sentences.
7. Include a clear ending/call to action when appropriate.
8. Use tools only when they are genuinely useful.
9. Return the script when you have enough information.
"""

    try:
        response = llm_with_tools.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an autonomous content-writing agent. "
                        "Use tools only when necessary."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )

        # If the model requests a tool, preserve the message.
        if getattr(response, "tool_calls", None):
            return {
                "messages": [response],
            }

        return {
            "script": response.content,
            "messages": [response],
        }

    except Exception as exc:
        return {
            "error": f"Writer error: {exc}",
        }


# ============================================================
# 8. REVIEWER NODE
# ============================================================

def reviewer_node(state: AgentState):
    """
    Reviews the generated script.

    This is the reflection/self-correction stage.
    """

    print("\n========== REVIEWER ==========")

    prompt = f"""
You are a strict quality reviewer.

Review this script:

{state.get("script", "")}

Evaluate:

1. Hook
2. Clarity
3. Flow
4. Grammar
5. Conversational style
6. Repetition
7. Factual faithfulness to the user's request
8. Strong ending

Give a score from 1 to 10.

Return EXACTLY in this format:

SCORE: <number>
FEEDBACK: <short useful feedback>
"""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()

        # Extract score safely.
        score = 0

        for line in text.splitlines():
            if line.upper().startswith("SCORE:"):
                try:
                    score = int(
                        line.split(":", 1)[1].strip().split()[0]
                    )
                except (ValueError, IndexError):
                    score = 0

        score = max(0, min(score, 10))

        return {
            "review": text,
            "score": score,
        }

    except Exception as exc:
        return {
            "review": f"Review error: {exc}",
            "score": 0,
            "error": f"Reviewer error: {exc}",
        }


# ============================================================
# 9. ROUTER AFTER REVIEW
# ============================================================

def review_router(state: AgentState):
    """
    Decides whether the agent should:
        - improve the script
        - continue to translation
        - stop because of an error
    """

    score = state.get("score", 0)
    revision_count = state.get("revision_count", 0)

    print(
        f"\nReview score: {score}/10 | "
        f"Revision: {revision_count}/{MAX_REVISIONS}"
    )

    # Stop if something failed.
    if state.get("error"):
        return "end"

    # Good enough → continue.
    if score >= QUALITY_THRESHOLD:
        return "translate"

    # Prevent infinite loops.
    if revision_count >= MAX_REVISIONS:
        print("Maximum revisions reached.")
        return "translate"

    # Otherwise revise.
    return "revise"


# ============================================================
# 10. REVISION NODE
# ============================================================

def revision_node(state: AgentState):
    """
    Increments the revision counter before sending the script
    back to the writer.
    """

    print("\n========== REVISION ==========")

    return {
        "revision_count": state.get("revision_count", 0) + 1
    }


# ============================================================
# 11. TRANSLATOR NODE
# ============================================================

def translator_node(state: AgentState):
    """
    Converts the final English script into natural Hinglish.
    """

    print("\n========== TRANSLATOR ==========")

    prompt = f"""
Convert the following script into natural Hinglish.

Script:
{state["script"]}

Rules:
1. Hindi must be written in Roman script.
2. Keep common technical English words in English.
3. Sound natural and conversational for young Indian audiences.
4. Preserve meaning and facts.
5. Do not add new information.
6. Do not use Devanagari.
7. Return ONLY the Hinglish script.
"""

    try:
        response = llm.invoke(prompt)

        return {
            "translated_script": response.content,
            "final_answer": response.content,
        }

    except Exception as exc:
        return {
            "error": f"Translator error: {exc}",
        }


# ============================================================
# 12. FINAL NODE
# ============================================================

def final_node(state: AgentState):
    """
    Final safety/output node.

    This ensures the graph always has a final answer.
    """

    if state.get("error"):
        return {
            "final_answer": (
                "The agent could not complete the task.\n\n"
                f"Error: {state['error']}"
            )
        }

    final_answer = state.get("translated_script")

    if not final_answer:
        final_answer = state.get("script", "No output generated.")

    return {
        "final_answer": final_answer
    }


# ============================================================
# 13. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)

# Add nodes.
graph.add_node("planner", planner_node)
graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("revision", revision_node)
graph.add_node("translator", translator_node)
graph.add_node("final", final_node)

# ------------------------------------------------------------
# START → PLANNER → WRITER
# ------------------------------------------------------------

graph.add_edge(START, "planner")
graph.add_edge("planner", "writer")

# ------------------------------------------------------------
# WRITER → TOOL OR REVIEWER
#
# tools_condition checks whether the model generated a tool call.
# If yes → tools.
# If no  → reviewer.
# ------------------------------------------------------------

graph.add_conditional_edges(
    "writer",
    tools_condition,
    {
        "tools": "tools",
        END: "reviewer",
    },
)

# ------------------------------------------------------------
# TOOL → WRITER
#
# After using a tool, the result goes back to the writer.
# This creates the agent's action/observation loop.
# ------------------------------------------------------------

graph.add_edge("tools", "writer")

# ------------------------------------------------------------
# REVIEWER → ROUTER
# ------------------------------------------------------------

graph.add_conditional_edges(
    "reviewer",
    review_router,
    {
        "revise": "revision",
        "translate": "translator",
        "end": "final",
    },
)

# Revision → Writer
graph.add_edge("revision", "writer")

# Translator → Final
graph.add_edge("translator", "final")

# Final → END
graph.add_edge("final", END)


# Compile the graph.
app = graph.compile()


# ============================================================
# 14. OPTIONAL GRAPH VISUALIZATION
# ============================================================

def show_graph():
    """
    Prints the Mermaid representation if available.
    Useful for understanding the architecture.
    """

    try:
        print("\n========== GRAPH ==========")
        print(app.get_graph().draw_mermaid())
    except Exception:
        print("Graph visualization is not available.")


# ============================================================
# 15. HUMAN-IN-THE-LOOP SAVE
# ============================================================

def ask_to_save(state: AgentState):
    """
    Ask the user before writing a file.

    This is an important production pattern:
    not every external side effect should be automatic.
    """

    answer = input(
        "\nDo you want to save the final script to a file? (y/n): "
    ).strip().lower()

    if answer != "y":
        return

    filename = input(
        "Enter filename (example: my_script.txt): "
    ).strip()

    if not filename:
        print("No filename supplied.")
        return

    result = save_script.invoke(
        {
            "filename": filename,
            "content": state["final_answer"],
        }
    )

    print(result)


# ============================================================
# 16. MAIN APPLICATION
# ============================================================

def main():
    print("=" * 60)
    print("       AUTONOMOUS SCRIPT AGENT")
    print("=" * 60)

    print(
        "\nThis agent can:\n"
        "1. Plan the task\n"
        "2. Write the script\n"
        "3. Use tools when needed\n"
        "4. Review its own output\n"
        "5. Revise weak output\n"
        "6. Translate to Hinglish\n"
        "7. Ask before saving the file\n"
    )

    user_input = input("\nEnter your goal/script request:\n> ").strip()

    if not user_input:
        print("Please provide a request.")
        return

    initial_state: AgentState = {
        "user_input": user_input,
        "messages": [],
        "plan": "",
        "script": "",
        "review": "",
        "score": 0,
        "revision_count": 0,
        "translated_script": "",
        "final_answer": "",
        "error": "",
    }

    print("\nStarting autonomous agent...")

    try:
        result = app.invoke(initial_state)

    except Exception as exc:
        print("\nAgent crashed:")
        print(exc)
        return

    print("\n" + "=" * 60)
    print("FINAL HINGLISH SCRIPT")
    print("=" * 60)
    print(result.get("final_answer", "No final answer."))

    print("\n" + "=" * 60)
    print("AGENT SUMMARY")
    print("=" * 60)
    print(f"Quality score: {result.get('score', 'N/A')}/10")
    print(f"Revisions: {result.get('revision_count', 0)}")

    if result.get("review"):
        print("\nReview:")
        print(result["review"])

    ask_to_save(result)


if __name__ == "__main__":
    main()
