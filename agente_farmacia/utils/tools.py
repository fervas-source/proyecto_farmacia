from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from agente_farmacia.utils.state import Medication
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.agents import AgentAction  
from typing import List

# No utilizamos Pydantic para el manejo de FAISS
@tool
def vademecum_retriever_tool(vectorstore: FAISS, question: str, k=1):
    """
    Retrieve documents from the vector store based on a given question.
    It will return documents closest to the query but may not be exact.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs_response = retriever.invoke(question)
    return docs_response

@tool
def forbidden_meds_retriever_tool(vectorstore: FAISS, medications: List[str], k=3):
    """
    Retrieve a list of forbidden medications for the given list of medications.
    If a medication is not in the list, then the medication is allowed.
    """
    results = {}
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    for medication in medications:
        docs_response = retriever.invoke(medication)
        if docs_response and any(medication.lower() in doc.page_content.lower() for doc in docs_response):
            results[medication] = "NOT ALLOWED"
        else:
            results[medication] = "ALLOWED"
    return results

# Inicializar el parser para las respuestas que usan el modelo Medication
parser_medicationlist = PydanticOutputParser(pydantic_object=Medication)

# Función auxiliar para analizar los pasos intermedios del agente
def create_scratchpad(intermediate_steps: List[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)
