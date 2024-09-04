# nodes.py

import pandas as pd
import pdfplumber
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

def generate_csv_vector_store(file_path: str) -> FAISS:
    """
    Generate a FAISS vector store from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        api_key (str): OpenAI API key.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    # Leer el archivo CSV
    df = pd.read_csv(file_path)

    # Inicializar una lista vacía para almacenar los textos
    texts = [
        ' '.join([f"{col}: {str(value)}" for col, value in zip(df.columns, row.values) if pd.notnull(value)])
        for _, row in df.iterrows()
    ]

    # Dividir los textos en chunks usando un text splitter
    text_splitter = CharacterTextSplitter()
    docs = [Document(page_content=chunk) for text in texts for chunk in text_splitter.split_text(text)]

    # Generar embeddings usando OpenAI
    embeddings = OpenAIEmbeddings()

    # Crear el vector store con FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def generate_pdf_vector_store(pdf_path: str) -> FAISS:
    """
    Generate a FAISS vector store from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        api_key (str): OpenAI API key.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    # Configurar parámetros
    chunk_size = 1000
    chunk_overlap = 200
    embedding_model = "text-embedding-ada-002"

    # Extraer texto del PDF usando pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join([page.extract_text() + "\n" for page in pdf.pages])

    # Crear un único objeto Document
    doc = Document(page_content=text, metadata={"source": pdf_path})

    # Dividir el documento en chunks con superposición
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents([doc])

    # Generar embeddings
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Crear el vector store con FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def generate_pdf_vector_store2(pdf_path: str) -> FAISS:
    """
    Generate a FAISS vector store specifically handling tables from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        api_key (str): OpenAI API key.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    tables_texts = []
    chunks = []

    # Abrir el PDF usando pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extraer el contenido de texto de la página
            text_content = page.extract_text()

            # Extraer las tablas de la página
            tables = page.extract_tables()
            field_names = ["Principio activo", "Registro sanitario", "Titular Reg. Sanitario", "Especialidad farmacéutica", "Forma farmacéutica", "Dosis", "Presentación x envase", "Estupefaciente o Psicotrópico"]
            table_metadata = {"field_names": field_names}

            for table in tables:
                # Convertir la tabla en texto manejando valores None
                table_text = "\n".join(["\t".join([str(cell) if cell is not None else "" for cell in row]) for row in table])
                tables_texts.append(table_text)

            # Dividir el contenido de las tablas en chunks
            for table in tables_texts:
                chunks.extend([Document(page_content=table, metadata=table_metadata)])

    # Generar embeddings
    embeddings = OpenAIEmbeddings()

    # Crear el vector store con FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
