# python
import streamlit as st
import pandas as pd
import plotly.express as px
import io
from gensim.models import KeyedVectors
from pathlib import Path


st.set_page_config(page_title="Bio-Psych Analyzer", layout="wide")

# --- GENERACION DEL EXCEL MULTISOLAPA ---
def create_master_xlsx(df_full, df_vec, df_troyer, group_col, col_name):
    output = io.BytesIO()
    
    # Determinar columnas para la hoja de Resumen
    cols_res = [col_name, col_sim, col_cat] # En lugar de nombres fijos entre comillas
    if group_col != "(Ninguno)":
        cols_res = [group_col] + cols_res
        
    # Seleccionamos solo las que existen para evitar errores
    df_resumen = df_full[cols_res]
    
    # Cambiamos engine='xlsxwriter' por engine='openpyxl'
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_full.to_excel(writer, sheet_name='1_Resultados_Completos', index=False)
        df_resumen.to_excel(writer, sheet_name='2_Resumen_Semantico', index=False)
        df_vec.to_excel(writer, sheet_name='3_Lista_Vectores', index=False)
        df_troyer.to_excel(writer, sheet_name='4_Diccionario_Troyer', index=False)
    
    return output.getvalue()

# --- DEFINIR RUTA BASE ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "data" / "SBW-vectors-300-min5.bin.gz"

# --- CARGA DEL MODELO (CACHE) ---
@st.cache_resource
def load_model():
    return KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# --- DICCIONARIO TROYER POR DEFECTO ---
DEFAULT_TROYER = {
    'perro': 'Mascotas',
    'gato': 'Mascotas',
    'hámster': 'Mascotas',
    'vaca': 'Granja',
    'cerdo': 'Granja',
    'oveja': 'Granja',
    'león': 'Selva',
    'tigre': 'Selva',
    'jirafa': 'Selva',
    'águila': 'Aves',
    'pájaro': 'Aves',
    'loro': 'Aves'
}

if 'troyer' not in st.session_state:
    # normalizar claves a minusculas y sin espacios al costado
    st.session_state.troyer = {k.lower().strip(): v for k, v in DEFAULT_TROYER.items()}

st.title("🧠 Analizador de Fluidez Verbal")

st.sidebar.info("Cargando el cerebro (Word2Vec)...")
modelo = load_model()
st.sidebar.success("Modelo listo.")

def procesar_archivo_troyer(file, nombre_hoja=0):
    """Extrae un diccionario de un archivo subido, soportando hojas de Excel."""
    if file.name.endswith(".csv"):
        df_temp = pd.read_csv(file)
    elif file.name.endswith(".tsv"):
        df_temp = pd.read_csv(file, sep='\t')
    else:
        # Para Excel, usamos la hoja seleccionada por el usuario
        df_temp = pd.read_excel(file, sheet_name=nombre_hoja)
    
    # Normalización de datos (Palabra en col 0, Categoría en col 1)
    return {
        str(row[0]).lower().strip(): str(row[1]).strip() 
        for _, row in df_temp.iterrows() 
        if pd.notnull(row[0])
    }

# --- UI: Mostrar y editar diccionario Troyer ---
st.sidebar.markdown("### 📚 Diccionario Troyer")

# --- NUEVA FEATURE: CARGA DE ARCHIVO ---
archivo_troyer = st.sidebar.file_uploader(
    "Subir diccionario propio (CSV, TSV o Excel)", 
    type=["csv", "tsv", "xlsx"]
)

# Variable para controlar la hoja seleccionada
hoja_elegida = 0

if archivo_troyer:
    # Si es Excel, leemos los nombres de las hojas primero
    if archivo_troyer.name.endswith(".xlsx"):
        try:
            xl = pd.ExcelFile(archivo_troyer)
            hoja_elegida = st.sidebar.selectbox(
                "Selecciona la hoja del diccionario:",
                xl.sheet_names
            )
        except Exception as e:
            st.sidebar.error("No se pudieron leer las hojas del Excel.")

    if st.sidebar.button("Cargar datos del archivo"):
        try:
            nuevo_dict = procesar_archivo_troyer(archivo_troyer, hoja_elegida)
            st.session_state.troyer = nuevo_dict
            st.sidebar.success(f"¡{len(nuevo_dict)} términos cargados de '{hoja_elegida}'!")
        except Exception as e:
            st.sidebar.error(f"Error al procesar el archivo: {e}")

# --- EDITOR DE TABLA ---
troyer_df = pd.DataFrame(
    list(st.session_state.troyer.items()),
    columns=["Palabra", "Categoria"]
)

# Usamos st.data_editor (la versión moderna)
try:
    edited = st.sidebar.data_editor(troyer_df, num_rows="dynamic", key="troyer_editor")
except Exception:
    edited = st.sidebar.experimental_data_editor(troyer_df, num_rows="dynamic", key="troyer_editor")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Actualizar Troyer"):
        # Reconstruir desde el editor con limpieza de datos
        new_dict = {
            str(r["Palabra"]).lower().strip(): str(r["Categoria"]).strip()
            for _, r in edited.iterrows()
            if str(r["Palabra"]).strip() != ""
        }
        st.session_state.troyer = new_dict
        st.sidebar.success("Diccionario actualizado.")
with col2:
    if st.button("Restablecer"):
        st.session_state.troyer = {k.lower().strip(): v for k, v in DEFAULT_TROYER.items()}
        st.rerun() # Refresca para que el editor muestre los valores por defecto

st.sidebar.markdown("---")

# --- UI: CARGA DE ARCHIVO ---
uploaded_file = st.file_uploader("Carga tus datos de sujetos (CSV, XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    # Si es Excel, seleccionamos la hoja
    if uploaded_file.name.endswith(".xlsx"):
        xl_data = pd.ExcelFile(uploaded_file)
        hoja_datos = st.selectbox("Selecciona la hoja donde están las respuestas:", xl_data.sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=hoja_datos)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("Vista previa de tus datos:")
    st.dataframe(df.head())

    col_name = st.selectbox(
        "Selecciona la columna que tiene los animales (palabras):",
        df.columns
    )

    # Elegir columna de sujeto/paciente para resetear análisis
    group_options = ["(Ninguno)"] + list(df.columns)
    group_col = st.selectbox(
        "Selecciona la columna que identifica al sujeto/paciente (opcional):",
        group_options,
        index=0
    )

    if st.button("Procesar Datos"):

        # 1. Obtener y limpiar la lista de palabras
        palabras_originales = df[col_name].astype(str).tolist()
        palabras_clean = [p.lower().strip() for p in palabras_originales]

        # --- FEATURE: CONTROL DE CALIDAD (QC) DEL DICCIONARIO ---
        dict_keys = set(st.session_state.troyer.keys())
        palabras_unicas = set(palabras_clean)
        faltantes = palabras_unicas - dict_keys

        if faltantes:
            with st.expander("⚠️ Alerta: Palabras fuera de Diccionario", expanded=True):
                st.warning(f"Se detectaron **{len(faltantes)}** palabras que no están en tu Troyer.")
                st.info(f"Palabras a revisar: {', '.join(sorted(faltantes))}")
                st.caption("Estas palabras se marcarán como 'Otros / No definido' en el reporte.")
        # -------------------------------------------------------

        # preparar lista de grupos (None si no se usa)
        if group_col != "(Ninguno)":
            grupos = df[group_col].astype(str).tolist()
            use_group = True
        else:
            grupos = [None] * len(palabras_originales)
            use_group = False

        # Creamos copia para mantener todas las columnas originales y añadimos las nuevas
        res_df = df.copy()
        # Definimos los nombres "estándar" que buscamos
        nombre_sim_estandar = "similitud_w2vec_coseno"
        nombre_cat_estandar = "categoria_troyer"

        # Función para buscar columna ignorando mayúsculas
        def encontrar_columna(df, nombre_buscado):
            for c in df.columns:
                if str(c).lower().strip() == nombre_buscado.lower():
                    return c
            return None

        # Buscamos si ya existen en el Excel del cliente
        col_sim = encontrar_columna(res_df, nombre_sim_estandar)
        col_cat = encontrar_columna(res_df, nombre_cat_estandar)

        # Si no existen, las creamos al final
        if col_sim is None:
            col_sim = nombre_sim_estandar
            res_df[col_sim] = None
        
        if col_cat is None:
            col_cat = nombre_cat_estandar
            res_df[col_cat] = None
        vectores_lista = []
        #resultados = []
        

        for i, p_clean in enumerate(palabras_clean):
            # Asignar Categoría usando el nombre detectado o creado
            res_df.at[i, col_cat] = st.session_state.troyer.get(p_clean, "Otros / No definido")

            if p_clean in modelo:
                vec = modelo[p_clean]
                vectores_lista.append([p_clean] + vec.tolist())

                if i == 0 or (use_group and grupos[i] != grupos[i-1]):
                    sim = None
                else:
                    prev = palabras_clean[i-1]
                    sim = float(modelo.similarity(prev, p_clean)) if prev in modelo else None
                
                res_df.at[i, col_sim] = float(sim) if sim is not None else None
            else:
                res_df.at[i, col_sim] = None

        # (Elimina la línea 'res_df = pd.DataFrame(resultados)' que tenías después del bucle)
        #res_df = pd.DataFrame(resultados)

        # --- VISUALIZACIÓN ---
        st.subheader("📊 Análisis Comparativo de Trayectorias")

        # 1. Crear una columna de posición relativa (para que todos empiecen en 0 en el eje X)
        if group_col != "(Ninguno)":
            # Esto hace que cada sujeto tenga su propio conteo 0, 1, 2, 3...
            res_df["Posicion_Sujeto"] = res_df.groupby(group_col).cumcount()
            eje_x = "Posicion_Sujeto"
            color_param = group_col  # El color cambiará según el ID del sujeto
        else:
            res_df["Posicion_Sujeto"] = res_df.index
            eje_x = "Posicion_Sujeto"
            color_param = None

        # 2. Crear el gráfico interactivo
        fig = px.line(
            res_df,
            x=eje_x,
            y=col_sim,
            color=color_param,  # Esto crea la leyenda y permite "apagar" sujetos
            hover_name=col_name,
            markers=True,
            title="Comparación de Similitud Semántica entre Sujetos",
            labels={
                eje_x: "Orden de la palabra producida",
                col_sim: "Similitud Coseno (W2Vec)",
                group_col: "ID Sujeto"
            },
            template="plotly_white" # Un estilo más limpio/académico
        )

        # 3. Ajustes finos de interactividad
        fig.update_layout(
            legend_title_text='Haga clic para ocultar:',
            hovermode="x unified" # Muestra todas las similitudes de los sujetos al mismo tiempo en esa posición
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- DESCARGAS ---
        st.subheader("📥 Descargar Reportes")
        
        # Preparar tablas adicionales
        vec_df = pd.DataFrame(vectores_lista, columns=["Palabra"] + [f"Dim_{i}" for i in range(300)])
        troyer_ref = pd.DataFrame(list(st.session_state.troyer.items()), columns=["Palabra", "Categoria"])

        # Botón para Excel Maestro
        excel_data = create_master_xlsx(res_df, vec_df, troyer_ref, group_col, col_name)
        st.download_button(
            "🌟 Descargar Reporte Excel Completo (4 solapas)",
            data=excel_data,
            file_name="SemanticFlow_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Botones individuales (CSV/TSV)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Resultados (CSV)", res_df.to_csv(index=False).encode("utf-8"), "resultados.csv", "text/csv")
        with col2:
            st.download_button("Vectores (TSV)", vec_df.to_csv(sep='\t', index=False).encode("utf-8"), "vectores.tsv", "text/tab-separated-values")