# resume_llm_scorer.py
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
# --------------------------
# Step 1 — define schema
# --------------------------
class ResumeScore(BaseModel):
    must_have_match_score: float = Field(..., description="Explicit keyword match score (0–100)")
    semantic_match_score: float = Field(..., description="Semantic relevance score (0–100)")
    experience_alignment: float = Field(..., description="Experience alignment (0–100)")
    education_alignment: float = Field(..., description="Education alignment (0–100)")
    overall_score: float = Field(..., description="Weighted overall fit score (0–100)")
    missing_must_have_skills: List[str] = Field(..., description="List of missing must-have skills in resume compared to JD")
    summary: str = Field(..., description="Recruiter-style explanation of match quality")

# --------------------------
# Step 2 — prompt template
# --------------------------
def create_agent(llm):
    prompt = ChatPromptTemplate.from_template("""
    You are an expert technical recruiter.

    Compare the following **Job Description (JD)** and **Candidate Resume**.
    You must evaluate both *explicit keyword matches* and *semantic relevance*.

    ### JOB DESCRIPTION
    {jd_text}

    ### RESUME
    {resume_text}

    Return your analysis as a JSON object with these fields:
    {{
    "must_have_match_score": 0-100,    // explicit skills overlap
    "semantic_match_score": 0-100,     // contextual fit even if wording differs
    "experience_alignment": 0-100,     // experience years and domain relevance
    "education_alignment": 0-100,      // degree/qualification match
    "overall_score": 0-100,            // weighted holistic fit
    "missing_must_have_skills": ["list of missing must-have skills. NOTE: Do not add skills if candidate has semantic relevance"],
    "summary": "short recruiter-style justification (2–3 sentences)"
    }}
    . 
    Base the overall_score primarily (60%) on must_have_match_score and semantic_match_score combined,
    and the rest (40%) on experience alignment.
    """)

    # --------------------------
    # Step 3 — build structured LLM
    # --------------------------

    # Create structured version
    structured_llm = llm.with_structured_output(ResumeScore)

    # --------------------------
    # Step 4 — compose the chain
    # --------------------------


    chain = RunnableSequence(prompt | structured_llm)
    return chain

# --------------------------
# Step 5 — run it
# --------------------------
def score_resume_with_llm(chain, jd_text: str, resume_text: str) -> ResumeScore:
    result = chain.invoke({"jd_text": jd_text, "resume_text": resume_text})
    return result


# if __name__ == "__main__":

#     result = score_resume_with_llm(jd_text, resume_text)
#     print(result.model_dump_json(indent=2))
