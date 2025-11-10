import os
import json
import docx2txt
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import StrOutputParser
from typing import TypedDict, List, Dict, Any
from src.AGENTS import resume_field_agent, resume_scorer_agent, resume_fields_extractor


def create_executer(llm):
    class JDState(TypedDict):
        resume_text: str
        resume_path:str
        jd_text:str
        resume_json: Dict[str, Any]
        resume_score: Dict[str, Any]
        messages: List[str]

    def resume_parsing_agent(state:JDState) -> JDState:
        ext = os.path.splitext(state["resume_path"])[1].lower()
        if ext in [".doc", ".docx"]:
            clean_text= docx2txt.process(state["resume_path"])
        else:
            layout_text= resume_field_agent.extract_layout_text(state["resume_path"])
            clean_text = " ".join(layout_text.split()) 
        state["resume_text"]=clean_text
        state["messages"] = add_messages(state["messages"], f"Parsed resume text: {clean_text}")
        return state

    def resume_fields_agent(state:JDState) -> JDState:
        json_data=resume_fields_extractor.extract_fields(state["resume_text"],llm)
        state["resume_json"]=json_data
        return state

    def resume_scoring_agent(state:JDState) -> JDState:
        chain=resume_scorer_agent.create_agent(llm)
        resume_score=resume_scorer_agent.score_resume_with_llm(chain,state["jd_text"],state["resume_text"])
        state["resume_score"]=resume_score
        return state

    workflow=StateGraph(JDState)

    workflow.add_node("resume_parsing_agent",resume_parsing_agent)
    workflow.add_node("resume_fields_agent",resume_fields_agent)
    workflow.add_node("resume_scoring_agent", resume_scoring_agent)

    workflow.set_entry_point("resume_parsing_agent")

    workflow.add_edge("resume_parsing_agent","resume_fields_agent")
    workflow.add_edge("resume_fields_agent","resume_scoring_agent")
    workflow.add_edge("resume_scoring_agent",END)

    graph=workflow.compile()
    return graph





    
