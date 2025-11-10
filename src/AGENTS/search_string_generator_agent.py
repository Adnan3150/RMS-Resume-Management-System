from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import os


# -----------------------------
# ✅ Structured Output Schema
# -----------------------------
class RecruiterSearchStrings(BaseModel):
    naukri_search_strings: list[str] = Field(..., description="List of Naukri-style Boolean search strings combining all key fields intelligently")
    recruiter_tags: list[str] = Field(..., description="List of recruiter-friendly tags/keywords derived from the JD")
    variations: list[str] = Field(..., description="3 recruiter-friendly variations of combined search strings using all fields")



parser = PydanticOutputParser(pydantic_object=RecruiterSearchStrings)


# -----------------------------
# ✅ Prompt Template
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are an expert AI Recruiter Assistant Agent.
Your task is to generate Boolean search strings, Naukri tags, and recruiter variations from the job description below.

### Job Description Data:
{jd_data}

### Instructions:
- Use **all available fields** from the JD such as:
  - job_title
  - must_have_skills
  - nice_to_have_skills
  - total_experience or min_experience/max_experience
  - notice_period
  - location / preferred_location
  - availability / joining time
  - domain or industry (if available)
- Do NOT limit yourself to skills — use all relevant data that helps recruiters find matching candidates.
- Use recruiter-style syntax compatible with **Naukri, LinkedIn, or Google Boolean searches**.
- Combine terms with `AND`, `OR`, and parentheses logically.
- Make 3–5 unique **Naukri-style search string variations**.
- Create a **tag list** containing core terms, tech stacks, experience levels, and availability markers.
- Return output strictly in this format:
{format_instructions}

### Example Output Format (for reference only)
{{
  "naukri_search_strings": [
    "('Python' OR 'Machine Learning' OR AI) AND ('Data Scientist' OR 'Analyst') AND (Hyderabad) AND (5-8 years)",
    "(('ML Engineer' OR 'AI Specialist') AND ('Deep Learning' OR 'TensorFlow') AND ('Immediate Joiner' OR '15 days notice'))"
  ],
  "recruiter_tags": ["Python", "ML", "Data Science", "5-8 Years", "Immediate Joiner"],
  "variations": [
    "Data Scientist (Python, ML, 5+ yrs, Immediate Joiner)",
    "ML Engineer (Deep Learning, AI, 6 yrs exp, 15-day notice)",
    "Senior Data Science Specialist - Python, TensorFlow, Immediate availability"
  ]
}}
""")


# -----------------------------
# ✅ Function: Generate Recruiter Strings Dynamically
# -----------------------------
def generate_recruiter_strings(jd_structured_data: dict, llm):
    """
    Input: structured JD dict (already extracted)
    Output: RecruiterSearchStrings object (parsed JSON)
    """
    input_prompt = prompt.format_messages(
        jd_data=json.dumps(jd_structured_data, indent=2),
        format_instructions=parser.get_format_instructions()
    )
    response = llm(input_prompt)
    try:
        return parser.parse(response.content)
    except Exception as e:
        # fallback if JSON parsing fails
        return RecruiterSearchStrings(
            naukri_search_strings=["Error generating string"],
            recruiter_tags=["Error parsing output"],
            variations=[response.content]
        )


# # -----------------------------
# # ✅ Example Usage
# # -----------------------------
# if __name__ == "__main__":
#     jd_data = {
#             "job_title": "AI Developer",
#              "must_have_skills": [      "Strong programming skills in Python",      "Basic to intermediate knowledge of Machine Learning algorithms (supervised, unsupervised, reinforcement learning)",      "Understanding of Generative AI concepts (LLMs, transformers, GANs, diffusion models, etc.)",      "Familiarity with libraries/frameworks such as TensorFlow, PyTorch, Scikit-learn, Hugging Face, OpenAI APIs",      "Good knowledge of data preprocessing, model training, and evaluation techniques",      "Problem-solving ability and eagerness to learn new technologies",      "Strong communication and teamwork skills"     ],
#               "nice_to_have_skills": [      "Hands-on project or internship experience in AI/ML or Generative AI",      "Exposure to cloud platforms (AWS, Azure, GCP) for AI/ML services",      "Knowledge of Natural Language Processing (NLP), Computer Vision, or Speech AI",      "Understanding of MLOps concepts (CI/CD, model deployment pipelines)"     ],
#                "experience": "Freshers or candidates with up to 2 years of experience",
#               "location": "Not specified"   }

#     result = generate_recruiter_strings(jd_data)
#     print("\n🧩 Naukri Search Strings:\n", result.naukri_search_strings)
#     print("\n🏷️ Recruiter Tags:\n", result.recruiter_tags)
#     print("\n🌀 Variations:\n", result.variations)
