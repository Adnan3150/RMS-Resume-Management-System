from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
import json
from typing import Union
from datetime import datetime

def extract_fields(resume_text, llm):
    class ResumeFields(BaseModel):
        candidate_name: str = Field(description="Full name of the candidate")
        email: str = Field(description="Email ID of the candidate")
        phone_number: str = Field(description="Phone number or mobile contact")
        linkedin: str = Field(description="LinkedIn profile URL if available")
        github: str = Field(description="GitHub or portfolio link if available")
        total_experience: str = Field(description=f"Total experience explicitly mentioned up to now {datetime.now().date().isoformat()}")
        employment_details: Union[list[dict],dict]=Field(description="current and previous employment including individual experience")   
        current_company: str = Field(description="Current company or most recent employer")
        current_designation: str = Field(description="Current or most recent job title")
        skills: Union[list[str], str] = Field(description="All listed technical and soft skills from overall resume irrespective only skill section")
        education: Union[list[str], str] = Field(description="Education details (degrees, universities, years) as string")
        certifications: Union[list[str], str] = Field(description="Relevant certifications or courses")
        projects: Union[list[dict], dict] = Field(description="Key projects names and descriptions with key skills")
        location: str = Field(description="Current location or city")
        preferred_location: str = Field(description="Preferred job location if mentioned")
        notice_period: str = Field(description="Notice period or availability for joining")
        expected_salary: str = Field(description="Expected salary or compensation details if provided")
        languages: Union[list[str], str] = Field(description="Languages known by the candidate")
        summary: str = Field(description="Brief professional summary or objective from the resume")

    parser = PydanticOutputParser(pydantic_object=ResumeFields)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert recruiter assistant. Extract structured resume data in detail for comparison with a job description."),
        ("user", "{resume_text}\n\nExtract all possible fields about the candidate. Return output strictly in JSON format following these fields: candidate_name, email, phone_number, linkedin, github, total_experience, current_company, current_designation, skills, education, certifications, projects, location, preferred_location, notice_period, expected_salary, languages, summary.\n{format_instructions}\nIf any field is missing, fill with 'N/A'. For skills, education, certifications, and projects, prefer list format if multiple are present.")
    ])

    chain = prompt | llm | parser
    raw_output = chain.invoke({
        "resume_text": resume_text,
        "format_instructions": parser.get_format_instructions()
    })

    data = json.loads(raw_output.json())

    # Convert string to list for list-like fields if LLM returned plain text
    for field in ["skills", "education", "certifications", "projects", "languages","employment_details"]:
        if isinstance(data.get(field), str|dict):
            data[field] = [data[field]]
    # print(data)
    result = ResumeFields(**data)
    print(result.dict())
    return result
