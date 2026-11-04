"""
app.py - Customer Support Crew (CrewAI + Streamlit)

Three agents run one after another (Process.sequential):
  1. Assistant            - answers the query from its own knowledge (no tools)
  2. Web Search Assistant - searches the web (SerperDevTool) and answers from the results
  3. Entry Agent          - saves the query + both answers to answers.txt (custom tool)
The Streamlit UI shows both answers to the user.

Setup:
    pip install crewai crewai-tools streamlit python-dotenv
    .env file next to this script:
        OPENAI_API_KEY=sk-...
        MODEL=gpt-4o-mini
        SERPER_API_KEY=...

Run:
    streamlit run app.py
"""

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import SerperDevTool

# ---------------------------------------------------------------
# 1. Keys from .env (never hard-coded)
# ---------------------------------------------------------------
load_dotenv(override=True)

ANSWERS_FILE = Path(__file__).parent / "answers.txt"

# Polite message used whenever an agent cannot find a reliable answer
NOT_AVAILABLE = (
    "I'm sorry, the answer to your question is not available at the moment. "
    "Please contact our support team for further help."
)


# ---------------------------------------------------------------
# 2. Tool for the Entry Agent: append one record to answers.txt
# ---------------------------------------------------------------
@tool("Save answers to file")
def save_answers(query: str, assistant_answer: str, web_answer: str) -> str:
    """Save the user's query, the Assistant's answer and the Web Search
    Assistant's answer to answers.txt. Pass each answer exactly as written."""
    record = (
        f"===== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        f"QUERY:\n{query}\n\n"
        f"ANSWER 1 - ASSISTANT:\n{assistant_answer}\n\n"
        f"ANSWER 2 - WEB SEARCH ASSISTANT:\n{web_answer}\n\n"
    )
    with open(ANSWERS_FILE, "a", encoding="utf-8") as f:
        f.write(record)
    return f"Saved to {ANSWERS_FILE.name}"


# ---------------------------------------------------------------
# 3. Build the crew: 3 agents, 3 tasks, sequential
# ---------------------------------------------------------------
def build_crew() -> Crew:
    llm = LLM(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.3)
    search = SerperDevTool(n_results=5)

    assistant = Agent(
        role="Customer Support Assistant",
        goal="Answer the customer's question clearly and correctly from your own knowledge.",
        backstory=(
            "You are a friendly, experienced support agent. You give short, "
            "step-by-step answers that a non-technical customer can follow, "
            "and you never guess: if you do not know, you say so politely."
        ),
        llm=llm,
        verbose=True,
    )  # no tools: answers from its own knowledge

    web_assistant = Agent(
        role="Web Search Support Assistant",
        goal="Search the web for the customer's question and answer using only what you find.",
        backstory=(
            "You are a support researcher. You look up current, reliable sources, "
            "base your answer only on the search results, and mention where the "
            "information came from. You never invent facts or links."
        ),
        tools=[search],  # the only agent that can search
        llm=llm,
        verbose=True,
    )

    entry_agent = Agent(
        role="Support Entry Agent",
        goal="Record every query and both answers in the answers file, then return both answers.",
        backstory=(
            "You are a meticulous record keeper. You copy information exactly, "
            "word for word, without summarising or changing anything."
        ),
        tools=[save_answers],  # the only agent that can write the file
        llm=llm,
        verbose=True,
    )

    answer_task = Task(
        description=(
            "Customer question: {query}\n"
            "Answer it from your own knowledge. If you do not know the answer or are "
            f"not confident it is correct, reply with exactly this sentence: {NOT_AVAILABLE}"
        ),
        expected_output=(
            "A clear answer in under 120 words, using numbered steps if it is a how-to "
            "question, OR the polite not-available sentence if you cannot answer."
        ),
        agent=assistant,
    )

    web_task = Task(
        description=(
            "Customer question: {query}\n"
            "Search the web, then answer using only the search results. "
            "If the search fails or the results do not answer the question, do not guess: "
            f"reply with exactly this sentence: {NOT_AVAILABLE}"
        ),
        expected_output=(
            "A clear answer in under 120 words based on the search results, "
            "followed by a line 'Sources:' listing 1-3 source names or URLs, "
            "OR the polite not-available sentence if no answer was found."
        ),
        agent=web_assistant,
    )

    entry_task = Task(
        description=(
            "Customer question: {query}\n"
            "Use the 'Save answers to file' tool ONCE to save the question, the "
            "Assistant's answer and the Web Search Assistant's answer, copied exactly "
            "from the previous tasks (including a not-available message, if that is "
            "what an agent replied). Then return both answers."
        ),
        expected_output=(
            "A confirmation line 'Saved to answers.txt', then 'ANSWER 1 - ASSISTANT:' "
            "with the first answer and 'ANSWER 2 - WEB SEARCH ASSISTANT:' with the second."
        ),
        agent=entry_agent,
        context=[answer_task, web_task],  # hand BOTH earlier answers to the Entry Agent
    )

    return Crew(
        agents=[assistant, web_assistant, entry_agent],
        tasks=[answer_task, web_task, entry_task],
        process=Process.sequential,
        verbose=True,
    )


# ---------------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Customer Support Crew", page_icon="🤝", layout="wide")
st.title("🤝 Customer Support Crew")
st.caption("Assistant → Web Search Assistant → Entry Agent (CrewAI, sequential)")

missing = [k for k in ("OPENAI_API_KEY", "SERPER_API_KEY") if not os.getenv(k)]
if missing:
    st.error(f"Missing in .env: {', '.join(missing)}. Add them and restart the app.")
    st.stop()

query = st.text_area("Enter your query or task", placeholder="e.g. How do I reset my password?")

if st.button("Ask the crew", type="primary"):
    if not query.strip():
        st.warning("Please type a query first.")
        st.stop()

    with st.status("The crew is working... (about 20-60 seconds)", expanded=True) as status:
        st.write("1️⃣ Assistant answers from its own knowledge")
        st.write("2️⃣ Web Search Assistant searches the web and answers")
        st.write("3️⃣ Entry Agent saves everything to answers.txt")
        try:
            result = build_crew().kickoff(inputs={"query": query.strip()})
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.info(NOT_AVAILABLE)
            st.caption(f"Technical details: {e}")
            st.stop()
        status.update(label="Done!", state="complete", expanded=False)

    # Each task's own output, so the UI shows the answers exactly as the agents wrote them
    assistant_answer = result.tasks_output[0].raw
    web_answer = result.tasks_output[1].raw

    def show_answer(answer: str):
        # Show the polite not-available message in a blue info box
        if "not available" in answer.lower():
            st.info(answer)
        else:
            st.markdown(answer)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 Assistant answer")
        show_answer(assistant_answer)
    with col2:
        st.subheader("🌐 Web Search answer")
        show_answer(web_answer)

    if ANSWERS_FILE.exists():
        st.success(f"Entry Agent saved this query and both answers to {ANSWERS_FILE.name}")
        st.download_button(
            "Download answers.txt",
            data=ANSWERS_FILE.read_text(encoding="utf-8"),
            file_name="answers.txt",
        )
    else:
        st.warning("The Entry Agent did not write answers.txt this time. Check the terminal log.")

    with st.expander("Entry Agent's final output"):
        st.text(result.raw)