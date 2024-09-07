import streamlit as st
from langchain.schema import HumanMessage, SystemMessage
from fonasa_2 import compiled_workflow  # Importar el flujo compilado
import os

# Configurar claves API
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEOPY_API_KEY = os.getenv("GEOPY_API_KEY")

# Configuración de la API de OpenAI
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Configurar Streamlit
st.title("Farmacias Cercanas y Convenio FONASA")
st.write("Ingrese su dirección para encontrar la farmacia más cercana y verificar si tiene convenio FONASA.")

# Solicitar dirección al usuario
direccion = st.text_input("Dirección", "Ingrese su dirección aquí")

# Si el usuario ha ingresado una dirección
if st.button("Buscar Farmacia"):
    if direccion:
        # Mostrar un spinner mientras se procesa la búsqueda
        with st.spinner('Buscando farmacias cercanas...'):
            try:
                # Crear un estado inicial del agente
                state = {"messages": [SystemMessage(content="Sistema inicializando...")]}
                human_message = HumanMessage(content=f"Quiero información sobre la farmacia más cercana a {direccion}")
                
                # Añadir el mensaje del usuario al estado
                state['messages'].append(human_message)

                # Invocar el flujo compilado
                result = compiled_workflow.invoke(state)  # Ejecutar el flujo

                # Verificar si result tiene el atributo 'messages'
                if 'messages' in result:
                    st.subheader("Resultado:")
                    st.write(result['messages'][-1].content)
                else:
                    st.error("No se pudieron obtener resultados del flujo.")

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
    else:
        st.error("Por favor, ingrese una dirección válida.")
