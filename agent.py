from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agente_farmacia.utils.state import AgentState
from agente_farmacia.utils.nodes import generate_csv_vector_store, generate_pdf_vector_store
from agente_farmacia.utils.tools import vademecum_retriever_tool, forbidden_meds_retriever_tool
from agente_farmacia.utils.prompts import vademecum_prompt_template, reviewer_prompt_template

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuración inicial de los vector stores
vademecum_vectorstore = generate_csv_vector_store('data/DrugData.csv')
restricted_med_vectorstore = generate_pdf_vector_store('data/Medicamentos_Registro_Sanitario.pdf')

# Definir la configuración del gráfico
class GraphConfig(TypedDict):
    model_name: Literal["openai"]

# Definir una nueva gráfica (StateGraph) utilizando la configuración y el estado del agente
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Definir el nodo que llama al modelo (agente)
def call_model(state: AgentState) -> AgentState:
    messages = state['messages']
    
    # Manejo del vectorstore y llamada a la función directamente
    response = vademecum_retriever_tool(vademecum_vectorstore, messages[-1])
    return {"messages": [response], "sender": "vademecum_agent"}

# Definir el nodo que maneja las herramientas (acciones adicionales)
def tool_node(state: AgentState) -> AgentState:
    messages = state['messages']
    
    # Manejo del vectorstore y llamada a la función directamente
    response = forbidden_meds_retriever_tool(restricted_med_vectorstore, messages[-1])
    return {"messages": [response], "sender": "reviewer_agent"}

# Definir la función de transición condicional
def should_continue(state: dict) -> str:
    messages = state['messages']
    last_message = messages[-1]
    if "No puedo prescribir medicamentos, por favor consulta a un médico" in last_message.content:
        return "end"
    if "FINAL RESPONSE" in last_message.content:
        return "end"
    if last_message.tool_calls:
        return "continue"
    return "end"

# Añadir los nodos "agent" y "action" al gráfico
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Establecer el punto de entrada como `agent`
workflow.set_entry_point("agent")

# Añadir un borde condicional para decidir la siguiente acción
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# Añadir un borde normal para volver de "action" a "agent"
workflow.add_edge("action", "agent")

# Compilar el gráfico en un LangChain Runnable
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
