from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def initialize_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Initialize Google Gemini Pro model"""
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.2)

def create_chains(llm: ChatGoogleGenerativeAI) -> dict:
    """Create LLM chains for different resume sections"""
    chains = {}

    # Personal Information Chain
    chains['personal_info'] = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["text"],
            template="""Extract the following details from the resume:
                - Full Name
                - Phone Number
                - Email Address
                Text: {text}
                Return only the extracted information in key: value format"""
        )
    )

    # Professional Summary Chain
    chains['professional_summary'] = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["text"],
            template="Extract the professional summary or career objective from the resume: {text}"
        )
    )

    # Work Experience Chain
    chains['work_experience'] = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["text"],
            template="""Extract work experience details. For each job include:
                - Job Title
                - Company Name
                - Employment Period
                - Responsibilities/Accomplishments
                - Technologies Used
                Text: {text}
                Format as: Job [number]: [details]"""
        )
    )

    # Skills Chain
    chains['skills'] = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["text"],
            template="Extract technical skills including programming languages, tools, and frameworks: {text}"
        )
    )

    # Education Chain
    chains['education'] = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["text"],
            template="""Extract education details including:
                - Degree
                - Institution
                - Graduation Year
                - GPA
                Text: {text}"""
        )
    )

    # Projects Chain
    chains['projects'] = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["text"],
            template="""Extract top 3 projects including:
                - Project Title
                - Description
                - Technologies Used
                Text: {text}"""
        )
    )

    return chains