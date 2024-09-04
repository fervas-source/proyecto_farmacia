import requests
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

def get_pharmacy_data(api_url: str):
    
    """
    Realiza una solicitud GET a la URL de la API proporcionada y devuelve los datos en formato JSON.

    :param api_url: URL de la API a la que se realizará la solicitud.
    :return: Los datos de la respuesta en formato JSON si la solicitud es exitosa, de lo contrario, None.
    """
    try:
        response = requests.get(api_url)

        # Comprobar si la solicitud fue exitosa (código de estado 200)
        if response.status_code == 200:
            # Devolver la respuesta en formato JSON
            return pd.DataFrame(response.json())
        else:
            print(f"Error en la solicitud: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error al realizar la solicitud: {e}")
        return None

def clean_coordinates(dataset, lat_col='local_lat', lng_col='local_lng'):

    df = dataset.copy()

    # Definir la condición para eliminar filas no deseadas
    condition = (
        (df[lat_col] == '') | (df[lng_col] == '') |  # Valores vacíos en local_lat o local_lng
        (df[lat_col] == '1') | (df[lng_col] == '1') |  # Valor 1 en local_lat o local_lng
        (df[lat_col] == 'Itinerante') | (df[lng_col] == 'Itinerante')  # 'Itinerante' en local_lat o local_lng
    )

    # Eliminar las filas que cumplen con la condición
    df_cleaned = df[~condition]

    # Limpiar las columnas de coordenadas
    df_cleaned[lat_col] = df_cleaned[lat_col].astype(str).str.rstrip(',').str.rstrip('°')
    df_cleaned[lng_col] = df_cleaned[lng_col].astype(str).str.rstrip(',').str.rstrip('°')

    # Función para reemplazar el cuarto carácter si es un guion
    def replace_fourth_char_if_hyphen(value):
        if len(value) > 3 and value[3] == '-':
            return value[:3] + '.' + value[4:]
        return value

    # Aplicar la transformación solo a los valores que cumplen la condición
    df_cleaned[lng_col] = df_cleaned[lng_col].apply(replace_fourth_char_if_hyphen)
    df_cleaned[lng_col] = df_cleaned[lng_col].apply(replace_fourth_char_if_hyphen)

    # Función para transformar solo los valores que comienzan con una coma
    def transform_if_needed(value):
        if isinstance(value, str) and value.startswith(','):
            return value.lstrip(',')
        return value

    # Aplicar la transformación solo a los valores que cumplen la condición
    df_cleaned[lat_col] = df_cleaned[lat_col].apply(transform_if_needed)
    df_cleaned[lng_col] = df_cleaned[lng_col].apply(transform_if_needed)

    # Reemplazar la coma por un punto
    df_cleaned[lat_col] = df_cleaned[lat_col].str.replace(',', '.', regex=False)
    df_cleaned[lng_col] = df_cleaned[lng_col].str.replace(',', '.', regex=False)

    # Convertir a float después de limpieza
    df_cleaned[lat_col] = pd.to_numeric(df_cleaned[lat_col], errors='coerce')
    df_cleaned[lng_col] = pd.to_numeric(df_cleaned[lng_col], errors='coerce')

    # Filtrar las filas con latitud y longitud fuera del rango permitido
    valid_lat_condition = df_cleaned[lat_col].between(-90, 90)
    valid_lng_condition = df_cleaned[lng_col].between(-180, 180)
    df_cleaned = df_cleaned[valid_lat_condition & valid_lng_condition]

    return df_cleaned

def coordenadas_direccion(direccion,GEOPY_API_KEY):

    # Inicializa el geocodificador
    geolocator = Nominatim(user_agent=GEOPY_API_KEY,ssl_context=ctx)

    # Geocodifica la dirección
    ubicacion = geolocator.geocode(direccion,addressdetails=True)

    # Validación 1: Verificar si la dirección se pudo geocodificar
    if not ubicacion:
        print("Error: No se pudo encontrar la dirección. Verifica si es correcta.")
        return None

    # Obtener la dirección devuelta y las coordenadas
    direccion_devuelta = ubicacion.address
    latitud = ubicacion.latitude
    longitud = ubicacion.longitude

    # Extraer la comuna, ciudad y país desde la información cruda
    detalle = ubicacion.raw['address']
    ciudad = detalle.get('city', detalle.get('town', detalle.get('village', 'No disponible')))
    pais = detalle.get('country', 'No disponible')

    # Validación 3: Verificar componentes clave como comuna, ciudad y país
    if ciudad == 'No disponible' or pais == 'No disponible':
        print("Error: La dirección no contiene información clave como la ciudad o el país.")
        return None

    # Si pasa todas las validaciones, la dirección es válida
    print("La dirección es válida.")
    print(f"Latitud: {latitud}, Longitud: {longitud}")
    print(f"Ciudad: {ciudad}, País: {pais}")

    # Devolver las coordenadas y la información relevante
    return {
        "direccion": direccion_devuelta,
        "Latitud": latitud,
        "Longitud": longitud,
        "ciudad": ciudad,
        "pais": pais
    }

def farmacia_mas_cercana(referencia,dataset):
    
    # Leer el dataset y el archivo de referencia
    
    df = clean_coordinates(dataset)
    punto_referencia = (referencia['Latitud'], referencia['Longitud'])

    # Calcular la distancia de cada local respecto al punto de referencia
    df['distancia'] = df.apply(
        lambda row: geopy.distance.geodesic((float(row['local_lat']), float(row['local_lng'])), punto_referencia).kilometers, axis=1
    )

    # Filtrar el dataset para el local con la menor distancia
    local_mas_cercano = df.loc[df['distancia'].idxmin()]

    return local_mas_cercano