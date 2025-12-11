import streamlit as st
import onnxruntime as ort
import numpy as np
import sqlite3
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# --- Configuração Repositório ---
REPO_ID = "viniciuslima47/pneumonia-vgg-model" 
MODEL_FILENAME = "modelo_pneumonia.onnx"

# --- Configuração BD ---
conn = sqlite3.connect("usos.db")
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resultado TEXT,
                prob REAL,
                filename TEXT
              )""")
conn.commit()

# --- Função de Carregamento do Modelo (ONNX) ---
@st.cache_resource
def load_onnx_model():
    """Baixa e carrega o modelo ONNX usando onnxruntime."""
    try:
        st.info(f"Baixando modelo {MODEL_FILENAME}...")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)

        session = ort.InferenceSession(model_path)
        st.success("Modelo ONNX carregado com sucesso!")
        return session
    except Exception as e:
        st.error(f"Erro ao carregar o modelo ONNX: {e}")
        st.caption(f"Verifique o REPO_ID ({REPO_ID}), se o arquivo ONNX está no repo, e se o setup.sh foi executado.")
        return None

# Carrega o modelo na inicialização
session = load_onnx_model()

# --- Interface do Streamlit ---
st.title("Classificador de Pneumonia (Versão ONNX)")
st.markdown("Diagnóstico de Raio-X com alto desempenho e estabilidade.")

if session is None:
     st.error("O aplicativo não pode rodar. O modelo falhou ao carregar.")
else:
    uploaded = st.file_uploader("Envie uma imagem de Raio-X (peito)", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption='Imagem Enviada', width=350)
        
        if st.button("Analisar Raio-X"):
            with st.spinner('Processando imagem...'):
                try:
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized).astype(np.float32)
                    img_array /= 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    input_name = session.get_inputs()[0].name
                    outputs = session.run(None, {input_name: img_array})

                    prob = float(outputs[0][0][0])
                    
                    # 3. Definição do Resultado
                    pred = "Pneumonia" if prob > 0.5 else "Normal"
                    
                    st.success("Análise concluída!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"Diagnóstico:")
                        if pred == "Pneumonia":
                            st.error(f"**{pred}**")
                        else:
                            st.success(f"**{pred}**")
                    with col2:
                        st.metric(label="Confiança do Modelo", value=f"{prob:.4f}")

                    # 4. Salvar no Banco de Dados SQLite
                    cur.execute("INSERT INTO historico (resultado, prob, filename) VALUES (?, ?, ?)",
                                (pred, prob, uploaded.name))
                    conn.commit()

                except Exception as e:
                    st.error(f"Erro durante o processamento (Inferência): {e}")

# --- Seção de Histórico ---
if st.checkbox("Mostrar Histórico de Uso"):
    st.markdown("### Registros do Banco de Dados")
    rows = cur.execute("SELECT * FROM historico ORDER BY id DESC").fetchall()
    df = pd.DataFrame(rows, columns=["ID", "Resultado", "Probabilidade", "Nome do Arquivo"])
    st.dataframe(
        df,
        use_container_width=True
    )
