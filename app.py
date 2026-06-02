import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# 1. Configuração da Página do Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Scanner com Yolo",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("📷 Scanner em Tempo Real com YOLO")
st.write("Protótipo funcional para detecção de objetos via webcam.")

# -----------------------------------------------------------------------------
# 2. Inicialização e Cache do Modelo YOLO
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    """
    Carrega o modelo YOLOv8 Nano. 
    O decorator cache_resource garante que o modelo seja carregado apenas uma vez.
    """
    # Utiliza o modelo nano (yolov8n.pt) por ser leve e ideal para deploys em nuvem (Render)
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# -----------------------------------------------------------------------------
# 4. Controle da Câmera via Interface do Usuário
# -----------------------------------------------------------------------------
# Botão para ativar/desativar o fluxo da câmera
click = st.checkbox("🔄 Ligar Câmera para Detecção")

# Espaço reservado na interface para atualizar os frames do vídeo continuamente
FRAME_WINDOW = st.image([])

# -----------------------------------------------------------------------------
# 3. Acesso à Câmera e Processamento de Detecção
# -----------------------------------------------------------------------------
if click:
    # Inicializa a captura de vídeo da webcam local (índice 0)
    camera = cv2.VideoCapture(0)
    
    # Loop de renderização contínua enquanto o checkbox estiver marcado
    while click:
        success, frame = camera.read()
        if not success:
            st.error("Não foi possível acessar a câmera.")
            break
            
        # O OpenCV captura em BGR, o Streamlit e o YOLO operam melhor em RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Executa a inferência do YOLO no frame atual
        results = model(frame_rgb, verbose=False)
        
        # Desenha as caixas delimitadoras e rótulos diretamente no frame processado
        annotated_frame = results[0].plot()
        
        # Atualiza a imagem na interface do Streamlit em tempo real
        FRAME_WINDOW.image(annotated_frame, channels="RGB")
        
    # Libera os recursos do hardware ao desmarcar o botão
    camera.release()
else:
    st.info("Clique no botão acima para iniciar o scanner.")