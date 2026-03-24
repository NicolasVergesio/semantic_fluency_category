# python
import streamlit as st
import pandas as pd
import plotly.express as px
from gensim.models import KeyedVectors
from pathlib import Path

st.set_page_config(page_title="Bio-Psych Analyzer", layout="wide")

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
uploaded_file = st.file_uploader("Carga tu archivo CSV o Excel", type=["csv", "xlsx"])

if uploaded_file:

    # Leer archivo según tipo
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
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

        resultados = []
        vectores_lista = []

        for i, p in enumerate(palabras_clean):

            p_clean = p.lower().strip()

            row = {
                "Orden": i + 1,
                "Palabra": p_clean
            }

            # --- Categoria Troyer (usar diccionario en session_state) ---
            row["Categoria_Troyer"] = st.session_state.troyer.get(
                p_clean,
                "Otros / No definido"
            )

            # --- Word2Vec ---
            if p_clean in modelo:

                vec = modelo[p_clean]

                # Determinar similitud: NULL si es primera fila del dataset
                # o si cambiamos de sujeto (cuando se eligió columna de grupo)
                if i == 0:
                    sim = None
                elif use_group and (grupos[i] != grupos[i - 1]):
                    sim = None
                else:
                    prev = palabras_clean[i - 1].lower().strip()
                    if prev in modelo:
                        sim = float(modelo.similarity(prev, p_clean))
                    else:
                        sim = None

                row["Similitud_Coseno"] = sim

                vectores_lista.append(
                    [p_clean] + vec.tolist()
                )

            else:
                row["Similitud_Coseno"] = None

            resultados.append(row)

        res_df = pd.DataFrame(resultados)

        # --- VISUALIZACIÓN ---
        st.subheader("Gráfico de Similitud Semántica")

        fig = px.line(
            res_df,
            x="Orden",
            y="Similitud_Coseno",
            hover_name="Palabra",
            markers=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Mostrar tabla de resultados
        st.subheader("Resultados")
        st.dataframe(res_df)

        # --- DESCARGAS ---
        col1, col2 = st.columns(2)

        with col1:
            csv_sim = res_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Descargar Resultados (CSV)",
                csv_sim,
                "resultados.csv",
                "text/csv"
            )

        with col2:

            if vectores_lista:

                vec_df = pd.DataFrame(
                    vectores_lista,
                    columns=["Palabra"] + [f"Dim_{i}" for i in range(300)]
                )

                csv_vec = vec_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Descargar Lista de Vectores (CSV)",
                    csv_vec,
                    "vectores.csv",
                    "text/csv"
                )
