from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

class VademecumTool:
    """
    Wrapper para interactuar con un vectorstore basado en FAISS para realizar búsquedas de documentos.

    Esta clase encapsula un vectorstore basado en FAISS y proporciona un método para recuperar
    documentos relevantes en función de una pregunta específica. Es útil para manejar consultas 
    dentro de un flujo de trabajo donde se necesita acceder rápidamente a información de medicamentos 
    almacenada en un vectorstore.

    Attributes:
        vectorstore (FAISS): El objeto vectorstore utilizado para realizar las búsquedas de documentos.

    Methods:
        retrieve(question: str, k=1) -> list:
            Recupera documentos del vectorstore basados en la pregunta proporcionada.
            Se devolverán los documentos más cercanos a la consulta.
    """

    def __init__(self, vectorstore: FAISS):
        """
        Inicializa la instancia de VademecumTool con un vectorstore.

        Args:
            vectorstore (FAISS): Un objeto vectorstore basado en FAISS que contiene 
                                 los documentos vectorizados para las búsquedas.
        """
        self.vectorstore = vectorstore

    def retrieve(self, question: str, k=1):
        """
        Recupera documentos del vectorstore basados en la pregunta proporcionada.

        Este método utiliza el vectorstore para buscar los documentos más cercanos a la 
        consulta proporcionada. El número de documentos recuperados está determinado 
        por el parámetro `k`.

        Args:
            question (str): La pregunta o consulta para la cual se deben recuperar los documentos.
            k (int, optional): El número de documentos a recuperar. Por defecto es 1.

        Returns:
            list: Una lista de documentos recuperados que incluyen información relevante 
                  como ID del medicamento, nombre, clase, indicaciones, efectos secundarios, etc.
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs_response = retriever.invoke(question)
        return docs_response


# Instancia global del wrapper, que será inicializada en main.py
global_vademecum_tool_instance = None

@tool
def vademecum_retriever_tool(question: str, k=1):
    """
    Función decorada para recuperar documentos del vectorstore utilizando VademecumTool.

    Esta función actúa como un punto de acceso simple para recuperar documentos relevantes 
    desde el vectorstore basado en FAISS. Está decorada para integrarse en flujos de trabajo 
    que requieren herramientas encadenadas.

    Args:
        question (str): La pregunta o consulta para la cual se deben recuperar los documentos.
        k (int, optional): El número de documentos a recuperar. Por defecto es 1.

    Returns:
        list: Una lista de documentos recuperados relacionados con la consulta.
    """
    return global_vademecum_tool_instance.retrieve(question, k)

