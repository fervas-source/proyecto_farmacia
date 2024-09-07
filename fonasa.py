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

@tool
def buscar_farmacia_mas_cercana(direccion, GEOPY_API_KEY):

    """
    Encuentra la farmacia más cercana y la farmacia de turno más cercana a la dirección proporcionada.

    Args:
        direccion (str): La dirección para la cual se desea encontrar la farmacia más cercana.
        GEOPY_API_KEY (str): Clave de API de Geopy para acceder a los servicios de geocodificación.

    Returns:
        tuple: Una tupla que contiene dos diccionarios:
            - El primer diccionario contiene la información de la farmacia más cercana con las claves:
                "pharmacy_name" (str): El nombre de la farmacia.
                "address" (str): La dirección de la farmacia.
                "opening_hours" (str): El horario de apertura de la farmacia.
                "closing_hours" (str): El horario de cierre de la farmacia.
                "distance" (str): La distancia desde la dirección proporcionada a la farmacia.
            - El segundo diccionario contiene la información de la farmacia de turno más cercana con las mismas claves.
              Retorna None si no se puede encontrar la farmacia más cercana.
    """
    
    referencia = coordenadas_direccion(direccion, GEOPY_API_KEY)
    if referencia is None:
        return None

    df_farmacias = get_pharmacy_data('https://midas.minsal.cl/farmacia_v2/WS/getLocales.php')
    local_mas_cercano = farmacia_mas_cercana(referencia, df_farmacias)

    df_farmacias_turno = get_pharmacy_data('https://midas.minsal.cl/farmacia_v2/WS/getLocalesTurnos.php')
    local_mas_cercano_turno = farmacia_mas_cercana(referencia, df_farmacias_turno)

    return {
        "pharmacy_name": local_mas_cercano["local_nombre"],
        "address": local_mas_cercano["local_direccion"],
        "city":  local_mas_cercano["comuna_nombre"],
        "opening_hours": local_mas_cercano["funcionamiento_hora_apertura"],
        "closing_hours": local_mas_cercano["funcionamiento_hora_cierre"],
        "distance": local_mas_cercano["distancia"]
    }, {
        "pharmacy_name": local_mas_cercano_turno["local_nombre"],
        "address": local_mas_cercano_turno["local_direccion"],
        "city":  local_mas_cercano["comuna_nombre"],
        "opening_hours": local_mas_cercano_turno["funcionamiento_hora_apertura"],
        "closing_hours": local_mas_cercano_turno["funcionamiento_hora_cierre"],
        "distance": local_mas_cercano_turno["distancia"]
    }
    
# Vincular la herramienta con el modelo LLM
tools_farmacia = [buscar_farmacia_mas_cercana]
llm_farmacia = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools_farmacia)

# Definir el agente 'farmacia_agent'
def farmacia_agent(state: AgentState) -> AgentState:
    messages = state['messages']
    chain = farmacia_prompt_template | llm_farmacia
    response = chain.invoke(messages)
    return {"messages": [response], "sender": "farmacia_agent"}

@tool
def farmacias_fonasa():
    """
    Carga el dataset de farmacias que tienen convenio con FONASA por región.

    Esta función no toma argumentos. Simplemente carga y devuelve un DataFrame
    que contiene la información sobre las farmacias con cobertura FONASA 
    en distintas regiones.

    Returns:
        pd.DataFrame: Un DataFrame con la información de las farmacias que tienen 
        convenio con FONASA. El DataFrame contiene las siguientes columnas:
            - "local_nombre" (str): El nombre de la farmacia.
            - "region" (str): La región donde se encuentra la farmacia.
            - "direccion" (str): La dirección de la farmacia.
            - "coordenadas" (tuple): Las coordenadas geográficas (latitud, longitud) de la farmacia.
    """
    # Cargar el dataset de farmacias con convenio FONASA
    dataset_fonasa = pd.read_csv('data/farmacias_fonasa_region.csv', encoding='iso-8859-1', sep=';')

    return dataset_fonasa

tools_fonasa = [farmacias_fonasa]  # Add any necessary tools here
llm_fonasa = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools_fonasa)

def fonasa_agent(state: AgentState)->AgentState:
    messages = state['messages']
    chain = fonasa_prompt_template | llm_fonasa
    response = chain.invoke(messages)
    return {"messages": [response], "sender": "fonasa_agent"}


# Define the workflow
workflow = StateGraph(AgentState)
workflow.add_node("farmacia_agent", farmacia_agent)
workflow.add_node("fonasa_agent", fonasa_agent)
tools = tools_farmacia + tools_fonasa

workflow.add_node("tool_node", ToolNode(tools))

# Conditional edges
def should_continue(state: dict) -> Literal["tools", "__end__","farmacia_agent"]:
    messages = state['messages']
    last_message = messages[-1]
    print("last message router ")
    print(last_message.content)
    print(last_message.tool_calls)

    content_lower = last_message.content.lower()

# Instructions:
    if "Error" in content_lower:
        return "__end__"
    if "FINAL RESPONSE" in last_message.content:
        return "__end__"
    if "RE EVALUAR" in content_lower:
        return "farmacia_agent"
    if last_message.tool_calls:
        return "tools"
    return "__end__"

workflow.add_conditional_edges("farmacia_agent", should_continue, {"tools": "tool_node", "__end__": "fonasa_agent"})
workflow.add_conditional_edges("fonasa_agent", should_continue, {"tools": "tool_node", "__end__": END})
workflow.add_conditional_edges(
    "tool_node",

    lambda x: x["sender"],
    {
        "farmacia_agent": "farmacia_agent",
        "fonasa_agent": "fonasa_agent",
    },
)

workflow.set_entry_point("farmacia_agent")

# Checkpointer
checkpointer = MemorySaver()

# Compile the workflow
app = workflow.compile(checkpointer=checkpointer)

question = "¿Qué farmacia esta cerca del parque arauco?"

# Define la consulta inicial y el mensaje del sistema
system_message = """
"""

app.invoke(
    {"messages": [SystemMessage(content=system_message), HumanMessage(content=question)]},
    config={"configurable": {"thread_id": 7}})