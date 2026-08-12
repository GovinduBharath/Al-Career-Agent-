# ============================================================
# PLACEMENT-READY AI CAREER AGENT
# LangChain + Gemini + FAISS + FastAPI + LangServe
# Render Deployment Ready
# ============================================================

import os
import requests
import faiss

from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. "
        "Add GOOGLE_API_KEY in Render Environment Variables."
    )


# ============================================================
# 2. GEMINI MODEL
# ============================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3-flash-preview"
)

print("Using Gemini model:", GEMINI_MODEL)


# ============================================================
# 3. INITIALIZE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

print("Gemini LLM initialized.")


# ============================================================
# 4. CAREER KNOWLEDGE BASE
# ============================================================

career_knowledge = """

PLACEMENT-READY AI CAREER KNOWLEDGE BASE

DATA SCIENTIST

Important skills:
Python
NumPy
Pandas
SQL
Statistics
Machine Learning
Scikit-learn
Matplotlib
Seaborn
Power BI
Basic Deep Learning

Typical projects:
Customer churn prediction
Sales forecasting
Fraud detection
Recommendation systems
Sentiment analysis
Predictive maintenance
Customer segmentation


DATA ANALYST

Important skills:
SQL
Excel
Python
Pandas
Statistics
Power BI
Tableau
Data Visualization

Typical projects:
Sales dashboard
Customer analysis
Business intelligence dashboard
Data cleaning project
Sales forecasting
Customer segmentation


PYTHON DEVELOPER

Important skills:
Python
Object Oriented Programming
SQL
Git
REST APIs
FastAPI
Flask
Django

Typical projects:
REST API
Web application
Automation system
Chatbot
Backend application


MACHINE LEARNING ENGINEER

Important skills:
Python
NumPy
Pandas
Scikit-learn
Machine Learning
Deep Learning
TensorFlow
PyTorch
REST APIs
Docker
Git

Typical projects:
Image classification
NLP application
Recommendation system
Predictive maintenance
Fraud detection


SOFTWARE DEVELOPER

Important skills:
Programming
Data Structures
Algorithms
Object Oriented Programming
SQL
Git
Problem Solving

Placement preparation should include:
Programming
Data Structures
Algorithms
SQL
Aptitude
Logical Reasoning
Communication
Projects
GitHub
Interview preparation


GOOD PLACEMENT PROJECT

A strong project should:

1. Solve a real-world problem.
2. Use appropriate technologies.
3. Have clear documentation.
4. Have a GitHub repository.
5. Include screenshots.
6. Explain the architecture.
7. Include results.
8. Preferably be deployed online.


GITHUB BEST PRACTICES

Use meaningful repository names.

Write a detailed README.

Include:
Project description
Features
Technologies
Installation steps
Usage
Screenshots
Architecture
Results

Keep repositories organized.
Commit code regularly.
Add deployment links when possible.


INTERVIEW PREPARATION

Students should prepare:

Python
Java/C/C++
SQL
Data Structures
Algorithms
Machine Learning
Statistics
Projects
Git/GitHub
Aptitude
Logical Reasoning
Communication

Students should be able to explain:

1. Problem statement
2. Technology used
3. Architecture
4. Their contribution
5. Challenges
6. Results
7. Future improvements
"""


documents = [
    Document(
        page_content=career_knowledge,
        metadata={
            "source": "Career Knowledge Base"
        }
    )
]

print("Career knowledge base created.")


# ============================================================
# 5. SPLIT DOCUMENT
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Created", len(chunks), "knowledge chunks.")


# ============================================================
# 6. GOOGLE EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

print("Embedding model:", EMBEDDING_MODEL)


# ============================================================
# 7. CREATE FAISS VECTOR STORE
# ============================================================

sample_embedding = embeddings.embed_query(
    "placement career skills"
)

embedding_dimension = len(sample_embedding)

index = faiss.IndexFlatL2(
    embedding_dimension
)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(
    documents=chunks
)

print("FAISS vector store created successfully.")


# ============================================================
# 8. CAREER SEARCH TOOL
# ============================================================

@tool
def career_search(query: str) -> str:
    """
    Search the career knowledge base for placement,
    career skills, job roles, projects and preparation.
    """

    retrieved_docs = vector_store.similarity_search(
        query,
        k=3
    )

    if not retrieved_docs:
        return "No relevant career information found."

    result = []

    for doc in retrieved_docs:
        result.append(
            f"Source: {doc.metadata}\n"
            f"Content: {doc.page_content}"
        )

    return "\n\n".join(result)


# ============================================================
# 9. JOB ROLE ANALYSIS TOOL
# ============================================================

@tool
def job_role_analysis(target_role: str) -> str:
    """
    Analyze skills and interview topics required
    for a target technology job role.
    """

    role = target_role.lower().strip()

    roles = {

        "data scientist": """
TARGET ROLE: Data Scientist

Important skills:
Python
NumPy
Pandas
SQL
Statistics
Machine Learning
Scikit-learn
Data Visualization

Interview topics:
Python
SQL
Statistics
Machine Learning
Projects
Problem Solving
""",

        "data analyst": """
TARGET ROLE: Data Analyst

Important skills:
SQL
Excel
Python
Pandas
Statistics
Power BI
Tableau
Data Visualization

Interview topics:
SQL
Excel
Statistics
Power BI
Data Analysis
Case Studies
""",

        "python developer": """
TARGET ROLE: Python Developer

Important skills:
Python
OOP
SQL
Git
FastAPI
Flask
Django
REST APIs

Interview topics:
Python
OOP
SQL
REST APIs
Data Structures
Projects
""",

        "machine learning engineer": """
TARGET ROLE: Machine Learning Engineer

Important skills:
Python
NumPy
Pandas
Machine Learning
Scikit-learn
Deep Learning
TensorFlow
PyTorch
Docker
APIs

Interview topics:
Machine Learning
Deep Learning
Python
Model Deployment
Projects
""",

        "software developer": """
TARGET ROLE: Software Developer

Important skills:
Programming
Data Structures
Algorithms
OOP
SQL
Git
Problem Solving

Interview topics:
DSA
OOP
SQL
Programming
Algorithms
Projects
"""
    }

    return roles.get(
        role,
        """
No specific role profile found.

General placement skills:

Python
SQL
Git
Data Structures
Algorithms
Problem Solving
Aptitude
Logical Reasoning
Projects
Communication
Interview Preparation
"""
    )


# ============================================================
# 10. SKILL GAP ANALYSIS TOOL
# ============================================================

@tool
def skill_gap_analysis(
    student_skills: str,
    target_role: str
) -> str:
    """
    Compare student skills with skills required
    for the selected job role.
    """

    role = target_role.lower().strip()

    role_skills = {

        "data scientist": [
            "python",
            "numpy",
            "pandas",
            "sql",
            "statistics",
            "machine learning",
            "scikit-learn",
            "data visualization"
        ],

        "data analyst": [
            "python",
            "sql",
            "pandas",
            "excel",
            "statistics",
            "power bi",
            "tableau",
            "data visualization"
        ],

        "python developer": [
            "python",
            "oop",
            "sql",
            "git",
            "fastapi",
            "flask",
            "django",
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
            "pytorch",
            "docker"
        ],

        "software developer": [
            "programming",
            "data structures",
            "algorithms",
            "oop",
            "sql",
            "git",
            "problem solving"
        ]
    }

    required_skills = role_skills.get(
        role,
        [
            "python",
            "sql",
            "git",
            "problem solving"
        ]
    )

    student_skill_list = [
        skill.strip().lower()
        for skill in student_skills.split(",")
        if skill.strip()
    ]

    matched = []
    missing = []

    for required in required_skills:

        if required in student_skill_list:
            matched.append(required)
        else:
            missing.append(required)

    score = round(
        len(matched) / len(required_skills) * 100
    )

    return f"""
TARGET ROLE:
{target_role}

SKILL MATCH:
{score}%

MATCHING SKILLS:
{", ".join(matched) if matched else "None"}

SKILL GAPS:
{", ".join(missing) if missing else "No major gaps found"}

RECOMMENDATION:

Focus on the missing skills.
Build practical projects using the most
important missing skills.
"""


# ============================================================
# 11. PROJECT ANALYSIS TOOL
# ============================================================

@tool
def project_analysis(
    projects: str,
    target_role: str
) -> str:
    """
    Analyze student projects according to the
    target placement role.
    """

    return f"""
PROJECT ANALYSIS

TARGET ROLE:
{target_role}

STUDENT PROJECTS:
{projects}

Evaluate projects using:

1. Real-world problem
2. Technical complexity
3. Relevance to target role
4. Technologies used
5. Data/AI/Software concepts
6. GitHub documentation
7. Deployment
8. Results

IMPROVEMENTS:

- Add a clear README.
- Explain the problem.
- Explain the architecture.
- Add screenshots.
- Mention technologies.
- Add results.
- Add GitHub link.
- Deploy the project if possible.
- Explain future improvements.
"""


# ============================================================
# 12. GITHUB CHECK TOOL
# ============================================================

@tool
def github_check(username: str) -> str:
    """
    Analyze a public GitHub profile.
    """

    username = username.strip()

    if not username:
        return "GitHub username was not provided."

    # FIXED GitHub API URL
    url = f"https://api.github.com/users/{username}"

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "Accept": "application/vnd.github+json"
            }
        )

        if response.status_code == 404:
            return f"GitHub user '{username}' was not found."

        if response.status_code != 200:
            return (
                "Unable to access GitHub profile. "
                f"Status code: {response.status_code}"
            )

        data = response.json()

        return f"""
GITHUB PROFILE ANALYSIS

Username:
{data.get("login", "N/A")}

Public Repositories:
{data.get("public_repos", 0)}

Followers:
{data.get("followers", 0)}

Following:
{data.get("following", 0)}

Profile:
{data.get("html_url", "N/A")}

RECOMMENDATIONS:

1. Create meaningful repositories.
2. Add detailed README files.
3. Add screenshots.
4. Explain technologies.
5. Add installation instructions.
6. Add project results.
7. Add deployment links.
8. Keep projects updated.
"""

    except requests.RequestException as error:

        return f"GitHub request failed: {error}"


# ============================================================
# 13. TOOLS
# ============================================================

tools = [
    career_search,
    job_role_analysis,
    skill_gap_analysis,
    project_analysis,
    github_check
]


# ============================================================
# 14. AGENT SYSTEM PROMPT
# ============================================================

system_prompt = """
You are a Placement-Ready AI Career Agent.

Your purpose is to help college students prepare
for technology placements.

You can help with:

- Career guidance
- Resume information
- Technical skills
- Target job roles
- Projects
- GitHub
- Skill gap analysis
- Interview preparation
- Learning roadmaps

AVAILABLE TOOLS:

career_search:
Search the career knowledge base.

job_role_analysis:
Identify skills and interview topics for a role.

skill_gap_analysis:
Compare student skills with job requirements.

project_analysis:
Evaluate student projects.

github_check:
Analyze a public GitHub profile.

IMPORTANT RULES:

1. Use tools when useful.
2. Never invent student information.
3. Do not claim the student has a skill unless provided.
4. Give practical recommendations.
5. Clearly identify skill gaps.
6. Recommend relevant projects.
7. Recommend a learning roadmap.
8. Help with interview preparation.
9. Keep answers clear and easy to understand.
10. Be honest when information is unavailable.

When appropriate, structure the response as:

PLACEMENT READINESS REPORT

Target Role:
Current Skills:
Strong Skills:
Skill Gaps:
Project Analysis:
GitHub Analysis:
Learning Roadmap:
Interview Preparation:
Overall Recommendation:
"""


# ============================================================
# 15. CREATE LANGCHAIN AGENT
# ============================================================

career_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

print("Placement AI Career Agent created.")


# ============================================================
# 16. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Placement-Ready AI Career Agent",
    description=(
        "AI Career Agent using LangChain, Gemini, "
        "FAISS, FastAPI and LangServe."
    ),
    version="1.0.0"
)


# ============================================================
# 17. HOME PAGE
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "application": "Placement-Ready AI Career Agent",
        "message": "AI Career Agent is running.",
        "agent_endpoint": "/agent/",
        "playground": "/agent/playground/",
        "health": "/health"
    }


# ============================================================
# 18. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 19. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Student career information or question"
    )


# ============================================================
# 20. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):
        user_input = x.get("input", "")
    else:
        user_input = x.input

    return {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    }


# ============================================================
# 21. EXTRACT AGENT RESPONSE
# ============================================================

def extract_text_response(agent_output):

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

    if not messages:
        return str(agent_output)

    last_message = messages[-1]

    content = getattr(
        last_message,
        "content",
        str(last_message)
    )

    # Gemini may return content as a list
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "thinking":
                    continue

                if "text" in item:
                    text_parts.append(
                        str(item["text"])
                    )

            else:
                text_parts.append(str(item))

        return "\n".join(text_parts)

    return str(content)


# ============================================================
# 22. LANGSERVE CHAIN
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
# 24. LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
