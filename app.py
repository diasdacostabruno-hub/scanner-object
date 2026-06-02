import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import torch

# -----------------------------------------------------------------------------
# CORREÇÃO DE SEGURANÇA (Compatibilidade PyTorch / Ultralytics)
# -----------------------------------------------------------------------------
# Libera as classes do YOLO na lista de permissões seguras de serialização do PyTorch
try:
    import ultralytics.nn.tasks
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel, YOLO])
except ImportError:
    pass

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
    Carrega o modelo YOLOv8 Nano de forma otimizada.
    O cache do Streamlit evita recargas custosas de memória a cada interação.
    """
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# -----------------------------------------------------------------------------
# 4. Controle da Câmera via Interface do Usuário
# -----------------------------------------------------------------------------
# Botão de ativação do fluxo de vídeo local
click = st.checkbox("🔄 Ligar Câmera para Detecção")

# Container dinâmico onde os frames atualizados serão renderizados
FRAME_WINDOW = st.image([])

# -----------------------------------------------------------------------------
# 3. Acesso à Câmera e Processamento de Detecção
# -----------------------------------------------------------------------------
if click:
    # Inicializa o dispositivo de captura padrão (Webcam local)
    camera = cv2.VideoCapture(0)
    
    # Loop de execução contínua enquanto o componente estiver marcado como True
    while click:
        success, frame = camera.read()
        if not success:
            st.error("Não foi possível acessar a câmera local ou o dispositivo está ocupado.")
            break
            
        # Converte o padrão de cor nativo do OpenCV (BGR) para o padrão do Streamlit/YOLO (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Executa o algoritmo de inferência no frame atual obtido
        results = model(frame_rgb, verbose=False)
        
        # Plota as caixas delimitadoras e os rótulos de identificação sobre a imagem
        annotated_frame = results[0].plot()
        
        # Renderiza a imagem atualizada no painel do Streamlit
        FRAME_WINDOW.image(annotated_frame, channels="RGB")
        
    # Liberação imediata do hardware assim que o ciclo for encerrado pelo usuário
    camera.release()
else:
    st.info("Clique no botão acima para iniciar o scanner.")
