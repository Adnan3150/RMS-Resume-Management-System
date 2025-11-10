import json
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import StrOutputParser
from typing import TypedDict, List, Dict, Any
# import search_string_generator_agent, jd_feilds_extractor
from src.AGENTS import search_string_generator_agent, jd_feilds_extractor
# ========== 1️⃣ Define the Shared Graph State ==========
class JDState(TypedDict):
    jd_text: str
    jd_json: Dict[str, Any]
    search_string: Dict[str, Any]
    messages: List[str]


def string_to_json(jd_json_str, search_json_str):
        """
        Convert two JSON strings (JD fields and search strings) into a single JSON object (Python dict)
        with keys 'jd_fields' and 'search_strings'.
        """
        # Parse JSON strings into Python dictionaries
        jd_data = json.loads(jd_json_str)
        search_data = json.loads(search_json_str)

        # Merge under separate keys
        merged_data = {
            "jd_fields": jd_data,
            "search_strings": search_data
        }

        # Return the JSON object (Python dict) directly
        return merged_data
def build_graph(llm):
    # ========== 3️⃣ JD Parser Agent ==========
    def jd_parser_agent(state: JDState) -> JDState:
        # jd_json = chain.invoke({"jd_text": state["jd_text"]})
        jd_json=jd_feilds_extractor.extract_feilds(state["jd_text"],llm).json()
        state["jd_json"] = jd_json
        state["messages"] = add_messages(state["messages"], f"Parsed JD: {jd_json}")
        return state

    # ========== 4️⃣ Search String Agent ==========
    def search_string_agent(state: JDState) -> JDState:
        search_str=search_string_generator_agent.generate_recruiter_strings(state["jd_json"],llm)
        as_json_str = json.dumps(search_str.model_dump(), ensure_ascii=False)
        state["search_string"] = as_json_str
        state["messages"] = add_messages(state["messages"], f"Generated search string: {as_json_str}")
        return state

    # ========== 5️⃣ Supervisor ==========
    def supervisor(state: JDState) -> str:
        """
        Determines the next node.
        """
        if "jd_json" not in state:
            return "jd_parser_agent"
        elif "search_string" not in state:
            return "search_string_agent"
        else:
            return END

    # ========== 6️⃣ Build the Graph ==========
    workflow = StateGraph(JDState)
    workflow.add_node("jd_parser_agent", jd_parser_agent)
    workflow.add_node("search_string_agent", search_string_agent)
    workflow.add_conditional_edges("jd_parser_agent", supervisor)
    workflow.add_conditional_edges("search_string_agent", supervisor)
    workflow.set_entry_point("jd_parser_agent")
    graph = workflow.compile()
    # print("graph",graph)
    return graph

def execute(graph, jd_text):
    inputs = {"jd_text": jd_text, "messages": []}
    final_state = graph.invoke(inputs)
    if final_state:
        # print("\n🧾 Parsed JD JSON:\n", final_state["jd_json"])
        # print("\n🔍 Naukri Boolean Query:\n", final_state["search_string"])
        final_json=string_to_json(final_state["jd_json"],final_state["search_string"])
        # print(final_json)
        # print("final json",final_json)
        return final_json

