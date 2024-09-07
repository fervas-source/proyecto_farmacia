import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agente_farmacia.utils.state import AgentState
from agente_farmacia.utils.distance import coordenadas_direccion, get_pharmacy_data, farmacia_mas_cercana
from agente_farmacia.utils.prompts import vademecum_prompt_template, reviewer_prompt_template, farmacia_prompt_template, fonasa_prompt_template
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from typing import TypedDict, Literal


# Cargar variables de entorno
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEOPY_API_KEY = os.getenv("GEOPY_API_KEY")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "fonasa_streamlit"
os.environ["LANGCHAIN_SESSION"] = "1"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Definir herramientas para farmacia y FONASA
@tool
def buscar_farmacia_mas_cercana(direccion, GEOPY_API_KEY):
    """
    Encuentra la farmacia más cercana y la farmacia de turno más cercana a la dirección proporcionada.
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
        "city":  local_mas_cercano_turno["comuna_nombre"],
        "opening_hours": local_mas_cercano_turno["funcionamiento_hora_apertura"],
        "closing_hours": local_mas_cercano_turno["funcionamiento_hora_cierre"],
        "distance": local_mas_cercano_turno["distancia"]
    }

# Definir el agente 'farmacia_agent'
def farmacia_agent(state: AgentState) -> AgentState:
    """
    Agente que maneja consultas sobre farmacias cercanas.
    """
    messages = state['messages']
    
    # Usar el prompt para crear la respuesta
    prompt = farmacia_prompt_template.format(messages=messages)
    
    # Enlazar la herramienta con el modelo de lenguaje
    tools_farmacia = [buscar_farmacia_mas_cercana]
    llm_farmacia = ChatOpenAI(model="gpt-4", temperature=0).bind_tools(tools_farmacia)
    
    response = llm_farmacia.invoke([HumanMessage(content=prompt)])

    # Actualizar el estado con el último agente
    state['messages'].append(response)
    state['last_agent'] = "farmacia_agent"  # Guardamos el último agente
    
    return state  # Retornamos el estado actualizado

@tool
def farmacias_fonasa():
    """
    Carga el dataset de farmacias que tienen convenio con FONASA por región.
    """
    dataset_fonasa = pd.read_csv('data/farmacias_fonasa_region.csv', encoding='iso-8859-1', sep=';')
    
    return dataset_fonasa

# Definir el agente 'fonasa_agent'
def fonasa_agent(state: AgentState) -> AgentState:
    """
    Agente que verifica si una farmacia tiene convenio con FONASA.
    """
    # Recuperar la respuesta del agente de farmacia
    farmacia_responses = state['messages'][-1].content  # Esto obtiene la última respuesta del farmacia_agent

    # Usar el prompt para crear la respuesta de FONASA
    prompt = fonasa_prompt_template.format(responses=farmacia_responses)
    
    tools_fonasa = [farmacias_fonasa]
    llm_fonasa = ChatOpenAI(model="gpt-4", temperature=0).bind_tools(tools_fonasa)
    
    response = llm_fonasa.invoke([HumanMessage(content=prompt)])

    # Actualizar el estado con el último agente
    state['messages'].append(response)
    state['last_agent'] = "fonasa_agent"  # Guardamos el último agente
    
    return state  # Retornamos el estado actualizado

# Define the workflow
workflow = StateGraph(AgentState)
workflow.add_node("farmacia_agent", farmacia_agent)
workflow.add_node("fonasa_agent", fonasa_agent)
tools = [buscar_farmacia_mas_cercana, farmacias_fonasa]

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

    lambda state: state.get("last_agent", "farmacia_agent"),
    {
        "farmacia_agent": "farmacia_agent",
        "fonasa_agent": "fonasa_agent",
    },
)

workflow.set_entry_point("farmacia_agent")

compiled_workflow = workflow.compile()  # Guardamos el flujo compilado