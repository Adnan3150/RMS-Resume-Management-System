from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
import json
from typing import Union

def extract_feilds(jd_text, llm):
    class JDFields(BaseModel):
            job_title: str = Field(description="Title of job opening")
            company_name:str = Field(description="Name of the organization")
            must_have_skills: Union[list[str], str] = Field(description="Required skills, key skills")
            nice_to_have_skills: Union[list[str], str] = Field(description="Only Optional skills or good to have")
            experience: str = Field(description="Years of experience required")
            location: str = Field(description="Job location (city or remote or hybrid)")
            education:str= Field(description="required education qualification")
            employment_type:str=Field(description="type of employment (full-time, part-time, contract etc..)")

        
    parser = PydanticOutputParser(pydantic_object=JDFields)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert HR assistant. Extract structured fields from the job description."),
        ("user", "{jd_text}\n\nReturn output in JSON format including 'job_title','must_have_skills', 'nice_to_have_skills', 'experience', and 'location'.\n{format_instructions}, Note: ADD N/A if any field missing")
    ])

    chain = prompt | llm | parser
    raw_output = chain.invoke({
        "jd_text": jd_text,
        "format_instructions": parser.get_format_instructions()
    })  
    data = json.loads(raw_output.json())
    for field in ["must_have_skills", "nice_to_have_skills"]:
        if isinstance(data.get(field), str):  
            data[field] = [data[field]]
    result = JDFields(**data)
    return result