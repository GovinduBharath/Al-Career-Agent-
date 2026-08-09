# ============================================================
# PLACEMENT-READY AI CAREER AGENT
# LangChain + Gemini + FAISS + FastAPI + LangServe
# ============================================================

import os
import requests
import faiss

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

import uvicorn

from langserve import add_routes

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain.tools import tool
from langchain.agents import create_agent


# ============================================================
# 1. LOAD API KEY
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not configured.")


# ============================================================
# 2. INITIALIZE GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="models/gemini-3-flash-preview",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

print("Gemini LLM initialized.")


# ============================================================
# 3. CAREER KNOWLEDGE BASE
# ============================================================

career_knowledge = """

DATA SCIENTIST:
Important skills:
Python, NumPy, Pandas, SQL, Statistics,
Machine Learning, Scikit-learn, Matplotlib,
Seaborn, Power BI and basic Deep Learning.

Typical projects:
Customer churn prediction, sales forecasting,
fraud detection, recommendation systems,
sentiment analysis and predictive maintenance.

PYTHON DEVELOPER:
Important skills:
Python, OOP, SQL, Git, REST APIs,
FastAPI, Flask, Django and basic cloud deployment.

Typical projects:
REST API, web application, automation system,
chatbot and backend application.

DATA ANALYST:
Important skills:
Python, SQL, Excel, Pandas,
Statistics, Power BI, Tableau and data visualization.

Typical projects:
Sales dashboard, customer analysis,
business intelligence dashboard and data analysis.

MACHINE LEARNING ENGINEER:
Important skills:
Python, NumPy, Pandas, Scikit-learn,
Machine Learning, Deep Learning, TensorFlow,
PyTorch, APIs, Docker and Git.

Typical projects:
Image classification, NLP application,
recommendation system and predictive maintenance.

SOFTWARE DEVELOPER:
Important skills:
Programming, Data Structures,
Algorithms, OOP, Git, SQL,
REST APIs and problem solving.

PLACEMENT PREPARATION:
Students should focus on:
Programming fundamentals,
Data Structures and Algorithms,
SQL,
Aptitude,
Logical Reasoning,
Communication,
Projects,
GitHub and interview preparation.

GOOD PROJECT:
A good placement project should solve a real-world problem,
use appropriate technologies,
contain clear documentation,
have a GitHub repository,
and preferably be deployed online.

GITHUB BEST PRACTICES:
Use meaningful repository names.
Write a good README.
Include project screenshots.
Add installation instructions.
Explain technologies used.
Keep repositories organized.
Commit code regularly.
Add deployed project links when possible.
"""


documents = [
    Document(page_content=career_knowledge)
]

print("Career knowledge base created.")


# ============================================================
# 4. SPLIT DOCUMENT
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Knowledge base split into chunks.")


# ============================================================
# 5. GOOGLE EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# 6. CREATE FAISS VECTOR STORE
# ============================================================

embedding_dimension = len(
    embeddings.embed_query("career skills")
)

index = faiss.IndexFlatL2(embedding_dimension)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(chunks)

print("FAISS vector store created.")


# ============================================================
# 7. RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 8. FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        f"Content:\n{doc.page_content}"
        for doc in docs
    )


# ============================================================
# 9. PLACEMENT RAG PROMPT
# ============================================================

career_prompt = ChatPromptTemplate.from_template(

"""
You are a Placement-Ready AI Career Assistant.

Use the provided career knowledge to answer the student's
career-related question.

Do not invent information.

If the knowledge base does not contain enough information,
clearly say that more information is required.

Career Knowledge:
{context}

Student Question:
{question}

Give a practical and concise answer.
"""
)


# ============================================================
# 10. BASIC RAG CHAIN
# ============================================================

career_rag_chain = (

    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }

    | career_prompt
    | llm
    | StrOutputParser()
)

print("Career RAG chain created.")


# ============================================================
# 11. TOOL 1 - CAREER KNOWLEDGE SEARCH
# ============================================================

@tool(response_format="content_and_artifact")
def career_search(query: str):
    """
    Search the career knowledge base for job requirements,
    skills, projects and placement preparation information.
    """

    retrieved_docs = vector_store.similarity_search(
        query,
        k=3
    )

    serialized = "\n\n".join(
        f"Source:\n{doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


# ============================================================
# 12. TOOL 2 - SKILL GAP ANALYZER
# ============================================================

@tool
def skill_gap_analysis(
    student_skills: str,
    target_role: str
):
    """
    Compare student skills with the expected skills
    for the target job role.
    """

    role_skills = {

        "data scientist": [
            "python",
            "numpy",
            "pandas",
            "sql",
            "statistics",
            "machine learning",
            "scikit-learn"
        ],

        "data analyst": [
            "python",
            "sql",
            "pandas",
            "excel",
            "statistics",
            "power bi",
            "data visualization"
        ],

        "python developer": [
            "python",
            "oop",
            "sql",
            "git",
            "fastapi",
            "flask",
            "rest api"
        ],

        "machine learning engineer": [
            "python",
            "numpy",
            "pandas",
            "machine learning",
            "scikit-learn",
            "deep learning",
            "tensorflow",
            "pytorch"
        ],

        "software developer": [
            "programming",
            "data structures",
            "algorithms",
            "oop",
            "sql",
            "git"
        ]
    }

    role = target_role.lower().strip()

    required = role_skills.get(
        role,
        [
            "python",
            "sql",
            "git",
            "problem solving"
        ]
    )

    student = [
        skill.strip().lower()
        for skill in student_skills.split(",")
    ]

    matched = [
        skill for skill in required
        if skill in student
    ]

    missing = [
        skill for skill in required
        if skill not in student
    ]

    if required:

        score = round(
            len(matched) / len(required) * 100
        )

    else:
        score = 0

    return f"""
Target Role: {target_role}

Skill Match: {score}%

Skills You Have:
{", ".join(matched) if matched else "None"}

Skills You Should Learn:
{", ".join(missing) if missing else "No major skill gaps"}
"""


# ============================================================
# 13. TOOL 3 - PROJECT ANALYZER
# ============================================================

@tool
def project_analysis(
    projects: str,
    target_role: str
):
    """
    Analyze student's projects and provide recommendations
    for the target placement role.
    """

    return f"""
Target Role:
{target_role}

Student Projects:
{projects}

Project Evaluation:

1. Does the project solve a real-world problem?
2. Does it use technologies related to the target role?
3. Is machine learning/data analysis/software development
   demonstrated where appropriate?
4. Is the project available on GitHub?
5. Does the README explain the project clearly?
6. Is the project deployed?

Recommendation:

Improve projects by adding:
- Clear README
- Architecture diagram
- Screenshots
- Technologies used
- Results
- GitHub repository
- Deployment link
"""


# ============================================================
# 14. TOOL 4 - GITHUB CHECKER
# ============================================================

@tool
def github_check(username: str):
    """
    Check a public GitHub profile and provide
    basic profile recommendations.
    """

    url = f"https://api.github.com/users/{username}"

    try:

        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code != 200:

            return (
                f"GitHub profile '{username}' "
                "could not be found."
            )

        data = response.json()

        return f"""
GitHub Profile Analysis

Username:
{data.get('login')}

Public Repositories:
{data.get('public_repos')}

Followers:
{data.get('followers')}

Following:
{data.get('following')}

Profile:
{data.get('html_url')}

Recommendations:
- Maintain meaningful repositories.
- Add README files.
- Add screenshots.
- Explain technologies used.
- Add project deployment links.
- Keep projects updated.
"""

    except Exception as e:

        return f"GitHub API error: {str(e)}"


# ============================================================
# 15. TOOL 5 - JOB ROLE ANALYZER
# ============================================================

@tool
def job_role_analysis(target_role: str):
    """
    Provide important skills and preparation areas
    for a target placement role.
    """

    roles = {

        "data scientist": """
Required:
Python, NumPy, Pandas, SQL,
Statistics, Machine Learning,
Scikit-learn and Data Visualization.

Interview:
Python, SQL, statistics,
machine learning and projects.
""",

        "data analyst": """
Required:
SQL, Excel, Python, Pandas,
Statistics, Power BI and Tableau.

Interview:
SQL, Excel, statistics,
data visualization and case studies.
""",

        "python developer": """
Required:
Python, OOP, SQL, Git,
FastAPI/Flask/Django and REST APIs.

Interview:
Python, OOP, SQL,
DSA and project explanation.
""",

        "machine learning engineer": """
Required:
Python, Machine Learning,
Deep Learning, Scikit-learn,
TensorFlow/PyTorch, APIs and Docker.

Interview:
ML algorithms, Python,
deep learning and system deployment.
"""
    }

    return roles.get(
        target_role.lower(),
        """
General placement preparation:

Python
SQL
Git
Data Structures
Algorithms
Aptitude
Logical Reasoning
Projects
Communication
Interview preparation
"""
    )


# ============================================================
# 16. CREATE CAREER AGENT
# ============================================================

tools = [
    career_search,
    skill_gap_analysis,
    project_analysis,
    github_check,
    job_role_analysis
]


system_prompt = """

You are a Placement-Ready AI Career Agent.

Your goal is to help students become ready for
software and technology placements.

The student may provide:

- Resume information
- Technical skills
- Target job role
- Projects
- GitHub username
- Career questions

You have access to several tools.

Use career_search when you need career knowledge.

Use job_role_analysis when the student asks about
a particular job role.

Use skill_gap_analysis to identify missing skills.

Use project_analysis to evaluate student projects.

Use github_check to analyze a public GitHub profile.

IMPORTANT:

1. Use tools when appropriate.
2. Do not invent student information.
3. Give practical recommendations.
4. Identify skill gaps clearly.
5. Recommend projects based on the target role.
6. Give a learning roadmap.
7. Help with interview preparation.
8. Keep responses easy to understand.

At the end, provide:

PLACEMENT READINESS
-------------------
Target Role:
Strong Skills:
Skill Gaps:
Project Recommendations:
GitHub Recommendations:
Learning Roadmap:
Interview Preparation:
Overall Advice:

"""


career_agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt
)

print("Placement Career Agent created.")


# ============================================================
# 17. TEST THE AGENT
# ============================================================

test_query = """

Student Skills:
Python, C, Java, SQL, Pandas

Target Role:
Data Scientist

Projects:
1. Sign Language to Text
2. AI Vehicle Maintenance Prediction

GitHub:
bharath123

Analyze my placement readiness.
"""

print("\n========== AGENT RESPONSE ==========\n")

for event in career_agent.stream(

    {
        "messages": [
            {
                "role": "user",
                "content": test_query
            }
        ]
    },

    stream_mode="values"

):

    message = event["messages"][-1]

    if isinstance(message.content, list):

        text_parts = []

        for item in message.content:

            if isinstance(item, dict):

                if item.get("type") != "thinking":

                    if "text" in item:
                        text_parts.append(item["text"])

            else:

                text_parts.append(str(item))

        if text_parts:
            print("\n".join(text_parts))

    else:

        print(message.content)


# ============================================================
# 18. FASTAPI INPUT
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Student career information or question"
    )


# ============================================================
# 19. FORMAT INPUT FOR AGENT
# ============================================================

def format_for_agent(x) -> dict:

    if isinstance(x, dict):

        user_input = x["input"]

    else:

        user_input = x.input

    return {
        "messages": [
            (
                "user",
                user_input
            )
        ]
    }


# ============================================================
# 20. EXTRACT FINAL RESPONSE
# ============================================================

def extract_text_response(agent_output) -> str:

    if not isinstance(agent_output, dict):

        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:

        for value in agent_output.values():

            if (
                isinstance(value, dict)
                and "messages" in value
            ):

                messages = value["messages"]
                break

    if messages:

        last = messages[-1]

        content = getattr(
            last,
            "content",
            str(last)
        )

        if isinstance(content, list):

            text = []

            for item in content:

                if isinstance(item, dict):

                    if (
                        item.get("type")
                        != "thinking"
                    ):

                        if "text" in item:
                            text.append(
                                item["text"]
                            )

                else:

                    text.append(str(item))

            return "\n".join(text)

        return str(content)

    return str(agent_output)


# ============================================================
# 21. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Placement-Ready AI Career Agent",
    description=(
        "AI agent that analyzes student skills, "
        "projects, GitHub and career requirements."
    ),
    version="1.0"
)


# ============================================================
# 22. CONNECT LANGCHAIN AGENT TO FASTAPI
# ============================================================

formatted_agent_chain = (

    RunnableLambda(format_for_agent)

    | career_agent

    | RunnableLambda(extract_text_response)

).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 23. LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/agent",

    playground_type="default"
)


# ============================================================
# 24. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
