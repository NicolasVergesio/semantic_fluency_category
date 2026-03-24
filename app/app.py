# python
import streamlit as st
import pandas as pd
import plotly.express as px
from gensim.models import KeyedVectors
from pathlib import Path

st.set_page_config(page_title="Bio-Psych Analyzer", layout="wide")

# --- DEFINIR RUTA BASE ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "SBW-vectors-300-min5.bin.gz"

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

# --- UI: Mostrar y editar diccionario Troyer ---
st.sidebar.markdown("### Diccionario Troyer (editable)")
troyer_df = pd.DataFrame(
    list(st.session_state.troyer.items()),
    columns=["Palabra", "Categoria"]
)
# Editable table (compatibilidad con diferentes versiones de Streamlit)
try:
    edited = st.sidebar.data_editor(troyer_df, num_rows="dynamic", key="troyer_editor")
except Exception:
    edited = st.sidebar.experimental_data_editor(troyer_df, num_rows="dynamic", key="troyer_editor")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Actualizar Troyer"):
        # reconstruir diccionario desde el editor, normalizando claves
        new_dict = {
            str(r["Palabra"]).lower().strip(): r["Categoria"]
            for _, r in edited.iterrows()
            if str(r["Palabra"]).strip() != ""
        }
        st.session_state.troyer = new_dict
        st.sidebar.success("Diccionario actualizado.")
with col2:
    if st.button("Restablecer Troyer"):
        st.session_state.troyer = {k.lower().strip(): v for k, v in DEFAULT_TROYER.items()}
        st.sidebar.success("Diccionario restablecido al valor por defecto.")

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

        palabras = df[col_name].astype(str).tolist()

        # preparar lista de grupos (None si no se usa)
        if group_col != "(Ninguno)":
            grupos = df[group_col].astype(str).tolist()
            use_group = True
        else:
            grupos = [None] * len(palabras)
            use_group = False

        resultados = []
        vectores_lista = []

        for i, p in enumerate(palabras):

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
                    prev = palabras[i - 1].lower().strip()
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