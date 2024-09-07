from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import operator
from langchain_core.tools import tool
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser

from agente_farmacia.utils.nodes import generate_csv_vector_store, generate_pdf_vector_store
#from agente_farmacia.utils.tools import vademecum_retriever_tool, forbidden_meds_retriever_tool
from agente_farmacia.utils.prompts import vademecum_prompt_template, reviewer_prompt_template, farmacia_prompt_template, fonasa_prompt_template
from agente_farmacia.utils.distance import coordenadas_direccion, get_pharmacy_data, farmacia_mas_cercana
from agente_farmacia.utils.state import AgentState, Medication

from typing import TypedDict, Annotated, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

from dotenv import load_dotenv
import os
from langchain_core.tools import tool

import warnings

# Suprimir todas las advertencias
warnings.filterwarnings("ignore")

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

import time

from geopy.geocoders import Nominatim
import os 
import ssl
import certifi
import pandas as pd
import geopy.distance

# Deshabilitar la verificación SSL
ctx = ssl.create_default_context(cafile=certifi.where())
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

import pandas as pd

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from pprint import pprint

from langchain.schema import AIMessage
import requests

from IPython.display import Image, display

from agente_farmacia.utils.distance import coordenadas_direccion, get_pharmacy_data, farmacia_mas_cercana


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEOPY_API_KEY = os.getenv("GEOPY_API_KEY")

langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "test_farmacias_fonasa"
os.environ["LANGCHAIN_SESSION"] = "1"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configuración inicial de los vector stores
vademecum_vectorstore = generate_csv_vector_store('data/DrugData.csv')
restricted_med_vectorstore = generate_pdf_vector_store('data/Medicamentos_Registro_Sanitario.pdf')

@tool
def vademecum_retriever_tool(question, k=1):
    """
    Retrieve documents from the vector store based on a given question.
    It will return documents closest to the query but may not be exact
    K must be 1 (k=1)

    Args:
        question (str): The question to retrieve documents for.
        k (int, optional): The number of documents to retrieve. Defaults to 1.

    Returns:
        list: A list of retrieved documents, each medication includes the following information
        Drug ID,Drug Name,Generic Name,Drug Class,Indications,Dosage Form,Strength,Route of Administration,
        Mechanism of Action,Side Effects,Contraindications,Interactions,Warnings and Precautions,Pregnancy Category,
        Storage Conditions,Manufacturer,Approval Date,Availability,NDC,Price
        """
    # Generate Retrieval
    retriever = vademecum_vectorstore.as_retriever(search_kwargs={"k": k})

    # Retrieve documents
    docs_response = retriever.invoke(question)

    return docs_response

tools_vademecum = [vademecum_retriever_tool]
llm_vademecum = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools_vademecum)

def vademecum_agent(state: AgentState)->AgentState:
    messages = state['messages']
    chain = vademecum_prompt_template | llm_vademecum 
    response = chain.invoke(messages)
    return {"messages": [response], "sender": "vademecum_agent"}

@tool
def forbiden_meds_retriever_tool(medications: List[str], k=3):
    """
    Retrieve a list of forbidden medications for the given list of medications.
    It will return documents closest to the query using each medication as a key.
    If a medication is not in the list, then the medication is allowed.

    Args:
        medications (List[str]): A list of medications to check.
        k (int, optional): The number of documents to retrieve for each medication. Defaults to 3.

    Returns:
        Dict[str, Union[List[str], str]]: A dictionary with medication names as keys and either a list of 
        "NOT ALOWED" or "ALLOWED" as values.
    """
    results = {}
    retriever = restricted_med_vectorstore.as_retriever(search_kwargs={"k": k})

    for medication in medications:
        docs_response = retriever.invoke(medication)
        if docs_response and any(medication.lower() in doc.page_content.lower() for doc in docs_response):
            results[medication] = "NOT ALLOWED"
        else:
            results[medication] = "ALLOWED"

    return results

tools_reviewer = [forbiden_meds_retriever_tool]  # Add any necessary tools here
llm_reviewer = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools_reviewer)

def reviewer_agent(state: AgentState)->AgentState:
    messages = state['messages']
    chain = reviewer_prompt_template | llm_reviewer
    response = chain.invoke(messages)
    return {"messages": [response], "sender": "reviewer_agent"}


# Define the workflow
workflow = StateGraph(AgentState)
workflow.add_node("vademecum_agent", vademecum_agent)
workflow.add_node("reviewer_agent", reviewer_agent)
tools = tools_vademecum + tools_reviewer

workflow.add_node("tool_node", ToolNode(tools))

# Conditional edges
def should_continue(state: dict) -> Literal["tools", "__end__","vademecum_agent"]:
    messages = state['messages']
    last_message = messages[-1]
    print("last message router ")
    print(last_message.content)
    print(last_message.tool_calls)

# Instructions:
    if "No puedo prescribir medicamentos, por favor consulta a un médico" in last_message.content:
        return "__end__"
    if "FINAL RESPONSE" in last_message.content:
        return "__end__"
    if "RE EVALUAR" in last_message.content:
        return "vademecum_agent"
    if last_message.tool_calls:
        return "tools"
    return "__end__"

workflow.add_conditional_edges("vademecum_agent", should_continue, {"tools": "tool_node", "__end__": "reviewer_agent"})
workflow.add_conditional_edges("reviewer_agent", should_continue, {"tools": "tool_node", "__end__": END})
workflow.add_conditional_edges(
    "tool_node",

    lambda x: x["sender"],
    {
        "vademecum_agent": "vademecum_agent",
        "reviewer_agent": "reviewer_agent",
    },
)

workflow.set_entry_point("vademecum_agent")

# Checkpointer
checkpointer = MemorySaver()

# Compile the workflow
app = workflow.compile(checkpointer=checkpointer)

# Draw the graph
#from IPython.display import Image, display
#display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# Pregunta a Responder
#question = "paracetamol"
question = "¿que es la aspirina, tafirol, amoxicilina, Alprazolam?"
#question = "¿que tomo para la fiebre?"
#question = input("Ingrese la pregunta: ")

# Define la consulta inicial y el mensaje del sistema
system_message = """
"""

start_time = time.time()
app.invoke(
    {"messages": [SystemMessage(content=system_message), HumanMessage(content=question)]},
    config={"configurable": {"thread_id": 7}}
)