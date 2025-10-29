"""
Sistema de Processamento e Análise de Imagens
Versão Streamlit v1.0

Instalação:
pip install streamlit opencv-python pillow numpy matplotlib scikit-image scikit-learn reportlab

Execução:
streamlit run sistema_processamento_imagens.py

Autor: Sistema de Processamento de Imagens
Data: 2025
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters, exposure, metrics
from skimage.filters import gaussian, median
from scipy import ndimage
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import tempfile
import os

# Configuração da página
st.set_page_config(
    page_title="Sistema de Processamento de Imagens",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS customizado
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CLASSE PRINCIPAL DO SISTEMA
# ============================================================================

class ImageProcessingSystem:
    """Sistema completo de processamento e análise de imagens"""
    
    def __init__(self):
        if 'initialized' not in st.session_state:
            st.session_state.original_image = None
            st.session_state.processed_image = None
            st.session_state.normalized_image = None
            st.session_state.history = []
            st.session_state.metrics = {}
            st.session_state.user = "Operador"
            st.session_state.initialized = True
    
    @staticmethod
    def log_action(action):
        """Registra ação no histórico"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {st.session_state.user}: {action}"
        st.session_state.history.insert(0, entry)
    
    @staticmethod
    def load_image(uploaded_file):
        """Carrega e normaliza imagem para 512x512px"""
        try:
            # Ler imagem
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("❌ Não foi possível carregar a imagem. Verifique a integridade do arquivo.")
                return False
            
            # Converter BGR para RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Salvar original
            st.session_state.original_image = img.copy()
            
            # Normalizar para 512x512
            st.session_state.normalized_image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            st.session_state.processed_image = st.session_state.normalized_image.copy()
            
            ImageProcessingSystem.log_action(f"Imagem '{uploaded_file.name}' carregada e normalizada para 512x512px")
            st.success("✅ Imagem carregada com sucesso!")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar imagem: {str(e)}")
            return False
    
    @staticmethod
    def apply_preprocessing(filter_type, kernel_radius, sigma):
        """Aplica filtros de pré-processamento"""
        try:
            if st.session_state.normalized_image is None:
                st.warning("⚠️ Carregue uma imagem primeiro!")
                return False
            
            # Validar parâmetros
            if sigma < 0.5 or sigma > 2.0:
                st.error("❌ Sigma deve estar entre 0.5 e 2.0")
                return False
            
            if kernel_radius % 2 == 0:
                kernel_radius += 1  # Garantir que seja ímpar
            
            img = st.session_state.processed_image.copy()
            
            with st.spinner('Aplicando pré-processamento...'):
                if filter_type == 'Gaussiano':
                    # Aplicar filtro gaussiano
                    filtered = np.zeros_like(img, dtype=np.float64)
                    for i in range(3):
                        filtered[:,:,i] = gaussian(img[:,:,i], sigma=sigma, preserve_range=True)
                    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
                    
                elif filter_type == 'Mediana':
                    # Aplicar filtro de mediana
                    filtered = cv2.medianBlur(img, kernel_radius)
                
                st.session_state.processed_image = filtered
                ImageProcessingSystem.log_action(f"Pré-processamento aplicado: {filter_type}, raio={kernel_radius}, sigma={sigma}")
                st.success(f"✅ Pré-processamento aplicado: {filter_type}")
                return True
                
        except Exception as e:
            st.error(f"❌ Erro no pré-processamento: {str(e)}")
            return False
    
    @staticmethod
    def apply_sharpening(method, weight, threshold, intensity):
        """Aplica métodos de realce de nitidez"""
        try:
            if st.session_state.processed_image is None:
                st.warning("⚠️ Carregue uma imagem primeiro!")
                return False
            
            img = st.session_state.processed_image.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            with st.spinner('Aplicando realce de nitidez...'):
                if method == 'Laplaciano':
                    # Filtro Laplaciano
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    laplacian = np.uint8(np.absolute(laplacian))
                    
                    # Aplicar em cada canal
                    sharpened = np.zeros_like(img)
                    for i in range(3):
                        sharpened[:,:,i] = cv2.addWeighted(img[:,:,i], 1.0, laplacian, weight, 0)
                    
                elif method == 'Sobel':
                    # Filtro Sobel
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    sobel = np.sqrt(sobelx**2 + sobely**2)
                    sobel = np.uint8(sobel)
                    
                    # Aplicar limiar
                    _, sobel = cv2.threshold(sobel, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Aplicar em cada canal
                    sharpened = np.zeros_like(img)
                    for i in range(3):
                        sharpened[:,:,i] = cv2.addWeighted(img[:,:,i], 1.0, sobel, weight, 0)
                    
                elif method == 'Alta Frequência':
                    # Filtro de alta frequência
                    blurred = cv2.GaussianBlur(img, (0, 0), 3)
                    sharpened = cv2.addWeighted(img, intensity, blurred, -(intensity-1), 0)
                
                st.session_state.processed_image = np.clip(sharpened, 0, 255).astype(np.uint8)
                ImageProcessingSystem.log_action(f"Nitidez aplicada: {method}, peso={weight}")
                st.success(f"✅ Nitidez aplicada: {method}")
                return True
                
        except Exception as e:
            st.error(f"❌ Erro ao aplicar nitidez: {str(e)}")
            return False
    
    @staticmethod
    def apply_contrast_enhancement(method, clip_limit, tile_size):
        """Aplica realce de contraste"""
        try:
            if st.session_state.processed_image is None:
                st.warning("⚠️ Carregue uma imagem primeiro!")
                return False
            
            # Validar parâmetros
            if clip_limit < 2.0 or clip_limit > 3.0:
                st.error("❌ Clip limit deve estar entre 2.0 e 3.0")
                return False
            
            img = st.session_state.processed_image.copy()
            
            with st.spinner('Aplicando realce de contraste...'):
                if method == 'CLAHE (Local)':
                    # CLAHE
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                    l = clahe.apply(l)
                    
                    lab = cv2.merge([l, a, b])
                    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    
                elif method == 'Equalização Global':
                    # Equalização de histograma global
                    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    
                    y = cv2.equalizeHist(y)
                    
                    ycrcb = cv2.merge([y, cr, cb])
                    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
                
                st.session_state.processed_image = enhanced
                ImageProcessingSystem.log_action(f"Contraste aplicado: {method}, clip_limit={clip_limit}")
                st.success(f"✅ Contraste aplicado: {method}")
                return True
                
        except Exception as e:
            st.error(f"❌ Erro ao aplicar contraste: {str(e)}")
            return False
    
    @staticmethod
    def calculate_metrics():
        """Calcula métricas quantitativas"""
        try:
            if st.session_state.normalized_image is None or st.session_state.processed_image is None:
                st.warning("⚠️ Carregue e processe uma imagem primeiro!")
                return False
            
            with st.spinner('Calculando métricas...'):
                original = st.session_state.normalized_image.astype(np.float64)
                processed = st.session_state.processed_image.astype(np.float64)
                
                # PSNR
                mse = np.mean((original - processed) ** 2)
                if mse == 0:
                    psnr = 100
                else:
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                
                # SSIM
                ssim = metrics.structural_similarity(
                    original, processed,
                    multichannel=True,
                    channel_axis=2,
                    data_range=255
                )
                
                # LC (Local Contrast)
                gray_processed = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_RGB2GRAY)
                lc = np.std(gray_processed) / (np.mean(gray_processed) + 1e-10)
                
                # Edge Sharpness
                edges = cv2.Canny(gray_processed, 100, 200)
                edge_sharpness = np.sum(edges > 0) / edges.size
                
                st.session_state.metrics = {
                    'PSNR': psnr,
                    'SSIM': ssim,
                    'LC': lc,
                    'Edge_Sharpness': edge_sharpness
                }
                
                ImageProcessingSystem.log_action("Métricas calculadas")
                st.success("✅ Métricas calculadas com sucesso!")
                return True
                
        except Exception as e:
            st.error(f"❌ Erro ao calcular métricas: {str(e)}")
            return False
    
    @staticmethod
    def generate_pdf_report():
        """Gera relatório em PDF"""
        try:
            if st.session_state.processed_image is None:
                st.warning("⚠️ Processe uma imagem primeiro!")
                return None
            
            with st.spinner('Gerando relatório PDF...'):
                # Criar arquivo temporário
                pdf_buffer = io.BytesIO()
                c = canvas.Canvas(pdf_buffer, pagesize=A4)
                width, height = A4
                
                # Título
                c.setFont("Helvetica-Bold", 20)
                c.drawString(50, height - 50, "Relatório de Processamento de Imagens")
                
                # Data
                c.setFont("Helvetica", 10)
                c.drawString(50, height - 70, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
                c.drawString(50, height - 85, f"Usuário: {st.session_state.user}")
                
                # Linha separadora
                c.line(50, height - 95, width - 50, height - 95)
                
                # Salvar imagens temporariamente
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_orig:
                    cv2.imwrite(tmp_orig.name, cv2.cvtColor(st.session_state.normalized_image, cv2.COLOR_RGB2BGR))
                    temp_orig_path = tmp_orig.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_proc:
                    cv2.imwrite(tmp_proc.name, cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_RGB2BGR))
                    temp_proc_path = tmp_proc.name
                
                # Adicionar imagens
                y_position = height - 300
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_position + 20, "Imagem Original:")
                c.drawImage(temp_orig_path, 50, y_position - 180, width=200, height=200)
                
                c.drawString(300, y_position + 20, "Imagem Processada:")
                c.drawImage(temp_proc_path, 300, y_position - 180, width=200, height=200)
                
                # Métricas
                y_position -= 220
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, y_position, "Métricas Quantitativas:")
                
                c.setFont("Helvetica", 11)
                y_position -= 25
                
                if st.session_state.metrics:
                    metrics_lines = [
                        f"PSNR: {st.session_state.metrics['PSNR']:.2f} dB (Alvo: >= 30 dB)",
                        f"SSIM: {st.session_state.metrics['SSIM']:.3f} (Alvo: >= 0.85)",
                        f"LC (Contraste Local): {st.session_state.metrics['LC']:.3f}",
                        f"Edge Sharpness: {st.session_state.metrics['Edge_Sharpness']:.3f}"
                    ]
                    
                    for line in metrics_lines:
                        c.drawString(70, y_position, line)
                        y_position -= 20
                
                # Histórico
                y_position -= 30
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, y_position, "Histórico de Processamento:")
                
                c.setFont("Helvetica", 8)
                y_position -= 20
                
                for entry in st.session_state.history[:15]:
                    if y_position < 50:
                        c.showPage()
                        y_position = height - 50
                    c.drawString(60, y_position, entry[:100])
                    y_position -= 12
                
                # Finalizar PDF
                c.save()
                
                # Limpar arquivos temporários
                os.unlink(temp_orig_path)
                os.unlink(temp_proc_path)
                
                ImageProcessingSystem.log_action("Relatório PDF gerado")
                
                pdf_buffer.seek(0)
                return pdf_buffer.getvalue()
                
        except Exception as e:
            st.error(f"❌ Erro ao gerar PDF: {str(e)}")
            return None

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

def main():
    """Função principal da aplicação"""
    
    # Inicializar sistema
    sistema = ImageProcessingSystem()
    
    # Cabeçalho
    st.title("🖼️ Sistema de Processamento e Análise de Imagens")
    st.markdown("### Análise e realce avançado com métricas quantitativas")
    
    # Sidebar - Informações do usuário
    with st.sidebar:
        st.header("👤 Informações do Usuário")
        st.session_state.user = st.text_input("Nome do Usuário", value=st.session_state.user)
        user_role = st.selectbox("Nível de Acesso", ["Operador", "Administrador"])
        
        st.divider()
        
        st.header("📊 Informações do Sistema")
        st.info("""
        **Formatos Suportados:** PNG, JPEG
        
        **Tamanho Normalizado:** 512x512px
        
        **Tamanho Máximo:** 10MB
        
        **Status:** 🟢 Online
        """)
        
        st.divider()
        
        st.header("📚 Fluxo de Trabalho")
        st.markdown("""
        1. Importar imagem
        2. Pré-processar
        3. Aplicar nitidez
        4. Aplicar contraste
        5. Calcular métricas
        6. Gerar relatório
        """)
    
    # Abas principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload", 
        "🔧 Processamento", 
        "📊 Análise", 
        "📈 Métricas", 
        "📄 Relatório"
    ])
    
    # ========================================================================
    # TAB 1: UPLOAD DE IMAGEM
    # ========================================================================
    with tab1:
        st.header("📤 Importação de Imagem")
        
        uploaded_file = st.file_uploader(
            "Escolha uma imagem (PNG ou JPEG)",
            type=['png', 'jpg', 'jpeg'],
            help="Tamanho máximo: 10MB. A imagem será normalizada para 512x512px"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Carregar Imagem", type="primary"):
                if ImageProcessingSystem.load_image(uploaded_file):
                    st.balloons()
        
        # Exibir imagem se carregada
        if st.session_state.normalized_image is not None:
            st.subheader("✅ Imagem Carregada")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(st.session_state.original_image, caption="Imagem Original", use_container_width=True)
                st.caption(f"Tamanho: {st.session_state.original_image.shape[1]}x{st.session_state.original_image.shape[0]}")
            
            with col2:
                st.image(st.session_state.normalized_image, caption="Imagem Normalizada (512x512)", use_container_width=True)
                st.caption("Pronta para processamento")
    
    # ========================================================================
    # TAB 2: PROCESSAMENTO
    # ========================================================================
    with tab2:
        if st.session_state.normalized_image is None:
            st.warning("⚠️ Por favor, carregue uma imagem primeiro na aba 'Upload'")
        else:
            st.header("🔧 Processamento de Imagem")
            
            # Pré-processamento
            with st.expander("🔹 1. Pré-processamento", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    filter_type = st.selectbox(
                        "Tipo de Filtro",
                        ["Gaussiano", "Mediana"],
                        help="Filtro de suavização"
                    )
                
                with col2:
                    kernel_radius = st.slider(
                        "Raio do Kernel",
                        min_value=1,
                        max_value=9,
                        value=3,
                        step=2,
                        help="Tamanho do kernel (deve ser ímpar)"
                    )
                
                with col3:
                    sigma = st.slider(
                        "Sigma",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        help="Desvio padrão para filtro gaussiano"
                    )
                
                if st.button("▶️ Aplicar Pré-processamento", key="btn_preproc"):
                    ImageProcessingSystem.apply_preprocessing(filter_type, kernel_radius, sigma)
            
            # Nitidez
            with st.expander("🔹 2. Realce de Nitidez"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sharp_method = st.selectbox(
                        "Método",
                        ["Laplaciano", "Sobel", "Alta Frequência"],
                        help="Método de realce de nitidez"
                    )
                
                with col2:
                    weight = st.slider(
                        "Peso",
                        min_value=0.1,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        help="Peso do filtro"
                    )
                
                with col3:
                    threshold = st.slider(
                        "Limiar (Sobel)",
                        min_value=10,
                        max_value=200,
                        value=50,
                        help="Limiar para detecção de bordas"
                    )
                
                with col4:
                    intensity = st.slider(
                        "Intensidade",
                        min_value=1.0,
                        max_value=1.5,
                        value=1.2,
                        step=0.1,
                        help="Intensidade do realce"
                    )
                
                if st.button("▶️ Aplicar Nitidez", key="btn_sharp"):
                    ImageProcessingSystem.apply_sharpening(sharp_method, weight, threshold, intensity)
            
            # Contraste
            with st.expander("🔹 3. Realce de Contraste"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    contrast_method = st.selectbox(
                        "Método",
                        ["CLAHE (Local)", "Equalização Global"],
                        help="Método de realce de contraste"
                    )
                
                with col2:
                    clip_limit = st.slider(
                        "Limite de Clipagem",
                        min_value=2.0,
                        max_value=3.0,
                        value=2.5,
                        step=0.1,
                        help="Limite para CLAHE"
                    )
                
                with col3:
                    tile_size = st.select_slider(
                        "Tamanho do Bloco",
                        options=[4, 8, 16],
                        value=8,
                        help="Tamanho do bloco para CLAHE"
                    )
                
                if st.button("▶️ Aplicar Contraste", key="btn_contrast"):
                    ImageProcessingSystem.apply_contrast_enhancement(contrast_method, clip_limit, tile_size)
            
            # Botão para calcular métricas
            st.divider()
            if st.button("📊 Calcular Métricas", type="primary", use_container_width=True):
                ImageProcessingSystem.calculate_metrics()
    
    # ========================================================================
    # TAB 3: ANÁLISE VISUAL
    # ========================================================================
    with tab3:
        if st.session_state.processed_image is None:
            st.warning("⚠️ Processe uma imagem primeiro na aba 'Processamento'")
        else:
            st.header("📊 Análise Visual Comparativa")
            
            # Comparação lado a lado
            st.subheader("🔍 Comparação Original vs Processada")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(st.session_state.normalized_image, caption="Original", use_container_width=True)
            
            with col2:
                st.image(st.session_state.processed_image, caption="Processada", use_container_width=True)
            
            st.divider()
            
            # Análise detalhada
            st.subheader("📈 Análise Detalhada")
            
            # Mapa de diferença
            diff = np.abs(
                st.session_state.normalized_image.astype(np.float32) - 
                st.session_state.processed_image.astype(np.float32)
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(diff.astype(np.uint8), caption="Mapa de Diferença", use_container_width=True)
            
            with col2:
                # Detecção de bordas - Original
                gray_orig = cv2.cvtColor(st.session_state.normalized_image, cv2.COLOR_RGB2GRAY)
                edges_orig = cv2.Canny(gray_orig, 100, 200)
                st.image(edges_orig, caption="Bordas - Original", use_container_width=True)
            
            with col3:
                # Detecção de bordas - Processada
                gray_proc = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_RGB2GRAY)
                edges_proc = cv2.Canny(gray_proc, 100, 200)
                st.image(edges_proc, caption="Bordas - Processada", use_container_width=True)
            
            st.divider()
            
            # Histogramas
            st.subheader("📊 Histogramas RGB")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histograma Original
            for i, color in enumerate(['red', 'green', 'blue']):
                hist = cv2.calcHist([st.session_state.normalized_image], [i], None, [256], [0, 256])
                ax1.plot(hist, color=color, alpha=0.7, label=color.upper())
            ax1.set_title('Histograma RGB - Original', fontweight='bold')
            ax1.set_xlim([0, 256])
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Histograma Processada
            for i, color in enumerate(['red', 'green', 'blue']):
                hist = cv2.calcHist([st.session_state.processed_image], [i], None, [256], [0, 256])
                ax2.plot(hist, color=color, alpha=0.7, label=color.upper())
            ax2.set_title('Histograma RGB - Processada', fontweight='bold')
            ax2.set_xlim([0, 256])
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # ========================================================================
    # TAB 4: MÉTRICAS
    # ========================================================================
    with tab4:
        st.header("📈 Métricas Quantitativas")
        
        if not st.session_state.metrics:
            st.info("ℹ️ Calcule as métricas na aba 'Processamento'")
        else:
            # Exibir métricas em cards
            col1, col2, col3, col4 = st.columns(4)
            
            psnr = st.session_state.metrics['PSNR']
            ssim = st.session_state.metrics['SSIM']
            lc = st.session_state.metrics['LC']
            edge = st.session_state.metrics['Edge_Sharpness']
            
            with col1:
                psnr_status = "✅" if psnr >= 30 else "⚠️"
                st.metric(
                    label=f"{psnr_status} PSNR",
                    value=f"{psnr:.2f} dB",
                    help="Peak Signal-to-Noise Ratio (Alvo: ≥ 30 dB)"
                )
                st.caption("Alvo: ≥ 30 dB")
            
            with col2:
                ssim_status = "✅" if ssim >= 0.85 else "⚠️"
                st.metric(
                    label=f"{ssim_status} SSIM",
                    value=f"{ssim:.3f}",
                    help="Structural Similarity Index (Alvo: ≥ 0.85)"
                )
                st.caption("Alvo: ≥ 0.85")
            
            with col3:
                st.metric(
                    label="📈 LC",
                    value=f"{lc:.3f}",
                    help="Local Contrast"
                )
                st.caption("Contraste Local")
            
            with col4:
                st.metric(
                    label="🔍 Edge Sharpness",
                    value=f"{edge:.3f}",
                    help="Nitidez das Bordas"
                )
                st.caption("Nitidez de Bordas")
            
            st.divider()
            
            # Validação
            if psnr >= 30 and ssim >= 0.85:
                st.success("✅ Todas as métricas estão dentro dos parâmetros especificados!")
            else:
                st.warning("⚠️ Algumas métricas estão abaixo do esperado. Considere ajustar os parâmetros de processamento.")
            
            st.divider()
            
            # Gráfico de métricas
            st.subheader("📊 Visualização das Métricas")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # PSNR
            ax1 = axes[0, 0]
            colors = ['green' if psnr >= 30 else 'orange']
            ax1.bar(['PSNR'], [psnr], color=colors, alpha=0.7)
            ax1.axhline(y=30, color='red', linestyle='--', label='Alvo: 30 dB')
            ax1.set_ylabel('dB')
            ax1.set_title('PSNR (Peak Signal-to-Noise Ratio)', fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # SSIM
            ax2 = axes[0, 1]
            colors = ['green' if ssim >= 0.85 else 'orange']
            ax2.bar(['SSIM'], [ssim], color=colors, alpha=0.7)
            ax2.axhline(y=0.85, color='red', linestyle='--', label='Alvo: 0.85')
            ax2.set_ylim([0, 1])
            ax2.set_title('SSIM (Structural Similarity Index)', fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # LC
            ax3 = axes[1, 0]
            ax3.bar(['LC'], [lc], color='blue', alpha=0.7)
            ax3.set_title('LC (Local Contrast)', fontweight='bold')
            ax3.grid(alpha=0.3)
            
            # Edge Sharpness
            ax4 = axes[1, 1]
            ax4.bar(['Edge Sharpness'], [edge], color='purple', alpha=0.7)
            ax4.set_ylim([0, 1])
            ax4.set_title('Edge Sharpness', fontweight='bold')
            ax4.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.divider()
            
            # Detalhes técnicos
            with st.expander("📋 Detalhes Técnicos das Métricas"):
                st.markdown("""
                ### PSNR (Peak Signal-to-Noise Ratio)
                Mede a razão entre o sinal máximo possível e a potência do ruído. 
                Valores mais altos indicam melhor qualidade.
                - **Excelente:** > 40 dB
                - **Bom:** 30-40 dB
                - **Aceitável:** 20-30 dB
                
                ### SSIM (Structural Similarity Index)
                Mede a similaridade estrutural entre duas imagens. 
                Valores mais próximos de 1 indicam maior similaridade.
                - **Excelente:** > 0.95
                - **Bom:** 0.85-0.95
                - **Aceitável:** 0.70-0.85
                
                ### LC (Local Contrast)
                Mede a variação local de intensidade na imagem.
                Valores mais altos indicam maior contraste local.
                
                ### Edge Sharpness
                Mede a nitidez das bordas na imagem.
                Valores mais altos indicam bordas mais definidas.
                """)
    
    # ========================================================================
    # TAB 5: RELATÓRIO
    # ========================================================================
    with tab5:
        st.header("📄 Geração de Relatório")
        
        if st.session_state.processed_image is None:
            st.warning("⚠️ Processe uma imagem primeiro")
        else:
            st.subheader("📋 Histórico de Operações")
            
            # Exibir histórico
            if st.session_state.history:
                for entry in st.session_state.history[:20]:
                    st.text(entry)
            else:
                st.info("Nenhuma operação registrada ainda")
            
            st.divider()
            
            # Botões de ação
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Gerar Relatório PDF", type="primary", use_container_width=True):
                    pdf_data = ImageProcessingSystem.generate_pdf_report()
                    if pdf_data:
                        st.download_button(
                            label="⬇️ Download Relatório PDF",
                            data=pdf_data,
                            file_name=f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
            
            with col2:
                if st.button("💾 Baixar Imagem Processada", use_container_width=True):
                    # Converter para bytes
                    img_pil = Image.fromarray(st.session_state.processed_image)
                    buf = io.BytesIO()
                    img_pil.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Imagem PNG",
                        data=buf,
                        file_name=f"imagem_processada_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            st.divider()
            
            # Resumo do processamento
            st.subheader("📊 Resumo do Processamento")
            
            if st.session_state.metrics:
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown("**Informações da Imagem:**")
                    st.write(f"- Tamanho: 512x512 pixels")
                    st.write(f"- Canais: RGB")
                    st.write(f"- Operações realizadas: {len(st.session_state.history)}")
                
                with summary_col2:
                    st.markdown("**Métricas de Qualidade:**")
                    st.write(f"- PSNR: {st.session_state.metrics['PSNR']:.2f} dB")
                    st.write(f"- SSIM: {st.session_state.metrics['SSIM']:.3f}")
                    st.write(f"- LC: {st.session_state.metrics['LC']:.3f}")
            
            st.divider()
            
            # Opções adicionais
            st.subheader("⚙️ Opções Adicionais")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Resetar Sistema", use_container_width=True):
                    st.session_state.original_image = None
                    st.session_state.processed_image = None
                    st.session_state.normalized_image = None
                    st.session_state.history = []
                    st.session_state.metrics = {}
                    st.success("✅ Sistema resetado!")
                    st.rerun()
            
            with col2:
                if st.button("📋 Limpar Histórico", use_container_width=True):
                    st.session_state.history = []
                    st.success("✅ Histórico limpo!")
                    st.rerun()
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p><strong>Sistema de Processamento de Imagens v1.0</strong></p>
        <p>Desenvolvido com Python, OpenCV, scikit-image e Streamlit</p>
        <p>Suporte a processamento em lote disponível para administradores</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# EXECUTAR APLICAÇÃO
# ============================================================================

if __name__ == "__main__":
    main()