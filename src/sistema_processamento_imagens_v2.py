"""
Sistema de Processamento e Análise de Imagens
Versão Streamlit v2.0

Instalação:
pip install streamlit opencv-python pillow numpy matplotlib scikit-image scikit-learn reportlab graphviz

Execução:
streamlit run sistema_processamento_imagens_v2.py

Autor: Sistema de Processamento de Imagens
Data: 2025
Versão: 2.0
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
    page_title="Sistema de Processamento de Imagens v2.0",
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
    
    # Constantes e limiares
    MAX_FILE_SIZE_MB = 10
    PSNR_THRESHOLD = 30.0
    SSIM_THRESHOLD = 0.85
    LC_MIN_THRESHOLD = 0.12
    EDGE_MIN_THRESHOLD = 0.03
    EDGE_MAX_THRESHOLD = 0.25
    
    def __init__(self):
        if 'initialized' not in st.session_state:
            st.session_state.original_image = None
            st.session_state.processed_image = None
            st.session_state.normalized_image = None
            st.session_state.preview_image = None
            st.session_state.versions = {}  # Para comparação global vs local
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
            # Verificar tamanho do arquivo
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > ImageProcessingSystem.MAX_FILE_SIZE_MB:
                st.error(f"❌ Arquivo muito grande ({file_size_mb:.2f} MB). Máximo: {ImageProcessingSystem.MAX_FILE_SIZE_MB} MB")
                return False
            
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
            st.session_state.preview_image = st.session_state.normalized_image.copy()
            
            ImageProcessingSystem.log_action(f"Imagem '{uploaded_file.name}' carregada ({file_size_mb:.2f} MB) e normalizada para 512x512px")
            st.success(f"✅ Imagem carregada com sucesso! ({file_size_mb:.2f} MB)")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar imagem: {str(e)}")
            return False
    
    @staticmethod
    def apply_preprocessing(filter_type, kernel_radius, sigma):
        """Aplica filtros de pré-processamento"""
        try:
            if st.session_state.processed_image is None:
                st.warning("⚠️ Carregue uma imagem primeiro!")
                return False
            
            # Validar parâmetros
            if sigma < 0.5 or sigma > 2.0:
                st.error("❌ Sigma deve estar entre 0.5 e 2.0")
                return False
            
            if kernel_radius % 2 == 0:
                kernel_radius += 1  # Garantir que seja ímpar
            
            img = st.session_state.processed_image.copy()
            
            if filter_type == 'Gaussiano':
                # Aplicar filtro gaussiano
                filtered = np.zeros_like(img, dtype=np.float64)
                for i in range(3):
                    filtered[:,:,i] = gaussian(img[:,:,i], sigma=sigma, preserve_range=True)
                filtered = np.clip(filtered, 0, 255).astype(np.uint8)
                
            elif filter_type == 'Mediana':
                # Aplicar filtro de mediana
                filtered = cv2.medianBlur(img, kernel_radius)
            
            st.session_state.preview_image = filtered
            ImageProcessingSystem.log_action(f"Pré-processamento aplicado: {filter_type}, raio={kernel_radius}, sigma={sigma}")
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
            
            if method == 'Laplaciano':
                # Filtro Laplaciano com máscara 3x3 (CORRIGIDO)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
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
            
            st.session_state.preview_image = np.clip(sharpened, 0, 255).astype(np.uint8)
            ImageProcessingSystem.log_action(f"Nitidez aplicada: {method}, peso={weight}")
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
            
            if method == 'CLAHE (Local)':
                # CLAHE
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                l = clahe.apply(l)
                
                lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                
                # Salvar versão local
                st.session_state.versions['local'] = enhanced.copy()
                
            elif method == 'Equalização Global':
                # Equalização de histograma global
                ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                
                y = cv2.equalizeHist(y)
                
                ycrcb = cv2.merge([y, cr, cb])
                enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
                
                # Salvar versão global
                st.session_state.versions['global'] = enhanced.copy()
            
            st.session_state.preview_image = enhanced
            ImageProcessingSystem.log_action(f"Contraste aplicado: {method}, clip_limit={clip_limit}")
            return True
                
        except Exception as e:
            st.error(f"❌ Erro ao aplicar contraste: {str(e)}")
            return False
    
    @staticmethod
    def apply_hybrid_processing(sigma, clip_limit, tile_size, sharp_method, weight, intensity):
        """Função híbrida: suavização + CLAHE + nitidez com guard-rails"""
        try:
            if st.session_state.normalized_image is None:
                st.warning("⚠️ Carregue uma imagem primeiro!")
                return False
            
            img = st.session_state.normalized_image.copy()
            
            # Etapa 1: Suavização Gaussiana
            smoothed = np.zeros_like(img, dtype=np.float64)
            for i in range(3):
                smoothed[:,:,i] = gaussian(img[:,:,i], sigma=sigma, preserve_range=True)
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
            
            # Etapa 2: CLAHE
            lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            contrasted = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Etapa 3: Verificar risco de oversharpening
            gray_contrasted = cv2.cvtColor(contrasted, cv2.COLOR_RGB2GRAY)
            edges_before = cv2.Canny(gray_contrasted, 100, 200)
            edge_density_before = np.sum(edges_before > 0) / edges_before.size
            
            # Ajustar peso se necessário
            adjusted_weight = weight
            adjusted_intensity = intensity
            oversharpening_risk = False
            
            if edge_density_before > 0.20:  # Alto conteúdo de bordas
                adjusted_weight = min(weight, 1.0)
                adjusted_intensity = min(intensity, 1.2)
                oversharpening_risk = True
            
            # Etapa 4: Aplicar nitidez
            gray = cv2.cvtColor(contrasted, cv2.COLOR_RGB2GRAY)
            
            if sharp_method == 'Laplaciano':
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
                laplacian = np.uint8(np.absolute(laplacian))
                sharpened = np.zeros_like(contrasted)
                for i in range(3):
                    sharpened[:,:,i] = cv2.addWeighted(contrasted[:,:,i], 1.0, laplacian, adjusted_weight, 0)
            
            elif sharp_method == 'Alta Frequência':
                blurred = cv2.GaussianBlur(contrasted, (0, 0), 3)
                sharpened = cv2.addWeighted(contrasted, adjusted_intensity, blurred, -(adjusted_intensity-1), 0)
            
            result = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            st.session_state.processed_image = result
            st.session_state.preview_image = result
            
            log_msg = f"Processamento híbrido aplicado: σ={sigma}, clip={clip_limit}, método={sharp_method}"
            if oversharpening_risk:
                log_msg += f" [AJUSTADO: peso {weight}→{adjusted_weight}, risco de oversharpening detectado]"
                st.warning(f"⚠️ Risco de oversharpening detectado! Parâmetros ajustados automaticamente.")
            
            ImageProcessingSystem.log_action(log_msg)
            st.success("✅ Processamento híbrido aplicado com sucesso!")
            return True
            
        except Exception as e:
            st.error(f"❌ Erro no processamento híbrido: {str(e)}")
            return False
    
    @staticmethod
    def confirm_preview():
        """Confirma a visualização e aplica ao processamento"""
        if st.session_state.preview_image is not None:
            st.session_state.processed_image = st.session_state.preview_image.copy()
            ImageProcessingSystem.log_action("Preview confirmado e aplicado")
            st.success("✅ Processamento aplicado!")
            return True
        return False
    
    @staticmethod
    def calculate_metrics():
        """Calcula métricas quantitativas com validação completa"""
        try:
            if st.session_state.normalized_image is None or st.session_state.processed_image is None:
                st.warning("⚠️ Carregue e processe uma imagem primeiro!")
                return False
            
            original = st.session_state.normalized_image.astype(np.float64)
            processed = st.session_state.processed_image.astype(np.float64)
            
            # PSNR
            mse = np.mean((original - processed) ** 2)
            if mse == 0:
                psnr = 100
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # SSIM (CORRIGIDO - removido multichannel)
            ssim = metrics.structural_similarity(
                original, processed,
                channel_axis=2,
                data_range=255.0
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
                'Edge_Sharpness': edge_sharpness,
                'psnr_ok': psnr >= ImageProcessingSystem.PSNR_THRESHOLD,
                'ssim_ok': ssim >= ImageProcessingSystem.SSIM_THRESHOLD,
                'lc_ok': lc >= ImageProcessingSystem.LC_MIN_THRESHOLD,
                'edge_ok': ImageProcessingSystem.EDGE_MIN_THRESHOLD <= edge_sharpness <= ImageProcessingSystem.EDGE_MAX_THRESHOLD
            }
            
            ImageProcessingSystem.log_action("Métricas calculadas com validação completa")
            st.success("✅ Métricas calculadas com sucesso!")
            return True
                
        except Exception as e:
            st.error(f"❌ Erro ao calcular métricas: {str(e)}")
            return False
    
    @staticmethod
    def generate_flowchart():
        """Gera diagrama de fluxo do processamento"""
        flowchart_text = """
        FLUXO DE PROCESSAMENTO:
        
        1. AQUISIÇÃO
           └─> Upload de imagem (PNG/JPEG, ≤10MB)
           └─> Normalização para 512x512px
        
        2. PRÉ-PROCESSAMENTO
           ├─> Filtro Gaussiano (σ: 0.5-2.0)
           └─> Filtro de Mediana (raio: 1-9)
        
        3. PROCESSAMENTO
           ├─> NITIDEZ
           │   ├─> Laplaciano 3x3
           │   ├─> Sobel
           │   └─> Alta Frequência
           │
           └─> CONTRASTE
               ├─> CLAHE Local (clip: 2.0-3.0)
               └─> Equalização Global
        
        4. AVALIAÇÃO
           ├─> PSNR ≥ 30 dB
           ├─> SSIM ≥ 0.85
           ├─> LC ≥ 0.12
           └─> Edge: 0.03-0.25
        
        5. DOCUMENTAÇÃO
           └─> Relatório PDF + Histórico
        """
        return flowchart_text
    
    @staticmethod
    def generate_pdf_report():
        """Gera relatório em PDF com conclusões"""
        try:
            if st.session_state.processed_image is None:
                st.warning("⚠️ Processe uma imagem primeiro!")
                return None
            
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=A4)
            width, height = A4
            
            # Título
            c.setFont("Helvetica-Bold", 20)
            c.drawString(50, height - 50, "Relatório de Processamento de Imagens v2.0")
            
            # Data
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 70, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            c.drawString(50, height - 85, f"Usuário: {st.session_state.user}")
            
            # Linha separadora
            c.line(50, height - 95, width - 50, height - 95)
            
            # Diagrama de fluxo
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, height - 115, "Fluxo de Processamento:")
            c.setFont("Courier", 7)
            y_pos = height - 135
            for line in ImageProcessingSystem.generate_flowchart().split('\n'):
                if y_pos < 400:
                    break
                c.drawString(50, y_pos, line[:100])
                y_pos -= 10
            
            # Salvar imagens temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_orig:
                cv2.imwrite(tmp_orig.name, cv2.cvtColor(st.session_state.normalized_image, cv2.COLOR_RGB2BGR))
                temp_orig_path = tmp_orig.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_proc:
                cv2.imwrite(tmp_proc.name, cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_RGB2BGR))
                temp_proc_path = tmp_proc.name
            
            # Adicionar imagens
            y_position = height - 500
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
                m = st.session_state.metrics
                metrics_lines = [
                    f"PSNR: {m['PSNR']:.2f} dB {'✓ OK' if m['psnr_ok'] else '✗ ABAIXO'} (Alvo: >= 30 dB)",
                    f"SSIM: {m['SSIM']:.3f} {'✓ OK' if m['ssim_ok'] else '✗ ABAIXO'} (Alvo: >= 0.85)",
                    f"LC: {m['LC']:.3f} {'✓ OK' if m['lc_ok'] else '✗ ABAIXO'} (Alvo: >= 0.12)",
                    f"Edge: {m['Edge_Sharpness']:.3f} {'✓ OK' if m['edge_ok'] else '✗ FORA'} (Alvo: 0.03-0.25)"
                ]
                
                for line in metrics_lines:
                    c.drawString(70, y_position, line)
                    y_position -= 20
                
                # CONCLUSÕES (NOVO)
                y_position -= 20
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, y_position, "Conclusões:")
                c.setFont("Helvetica", 10)
                y_position -= 20
                
                all_ok = m['psnr_ok'] and m['ssim_ok'] and m['lc_ok'] and m['edge_ok']
                
                if all_ok:
                    c.drawString(70, y_position, "✓ APROVADO - Todas as métricas dentro dos parâmetros.")
                    y_position -= 15
                    c.drawString(70, y_position, "A imagem processada atende aos critérios de qualidade.")
                else:
                    c.drawString(70, y_position, "✗ REPROVADO - Ajustes necessários:")
                    y_position -= 15
                    if not m['psnr_ok']:
                        c.drawString(85, y_position, "• PSNR baixo: reduzir intensidade de processamento")
                        y_position -= 12
                    if not m['ssim_ok']:
                        c.drawString(85, y_position, "• SSIM baixo: preservar melhor a estrutura original")
                        y_position -= 12
                    if not m['lc_ok']:
                        c.drawString(85, y_position, "• LC baixo: aumentar contraste local (CLAHE)")
                        y_position -= 12
                    if not m['edge_ok']:
                        c.drawString(85, y_position, "• Edge fora da faixa: ajustar nitidez")
                        y_position -= 12
            
            # Nova página para histórico
            c.showPage()
            y_position = height - 50
            
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Histórico de Processamento:")
            
            c.setFont("Helvetica", 8)
            y_position -= 20
            
            for entry in st.session_state.history[:30]:
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
            
            ImageProcessingSystem.log_action("Relatório PDF gerado com conclusões")
            
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
    st.title("🖼️ Sistema de Processamento de Imagens v2.0")
    st.markdown("### Análise e realce avançado com métricas quantitativas")
    
    # Sidebar
    with st.sidebar:
        st.header("👤 Informações do Usuário")
        st.session_state.user = st.text_input("Nome do Usuário", value=st.session_state.user)
        user_role = st.selectbox("Nível de Acesso", ["Operador", "Administrador"])
        
        st.divider()
        
        st.header("📊 Informações do Sistema")
        st.info(f"""
        **Versão:** 2.0
        
        **Formatos:** PNG, JPEG
        
        **Tamanho:** 512x512px
        
        **Limite:** {ImageProcessingSystem.MAX_FILE_SIZE_MB} MB
        
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
        
        st.divider()
        
        st.header("🎯 Critérios de Qualidade")
        st.markdown(f"""
        - **PSNR:** ≥ {ImageProcessingSystem.PSNR_THRESHOLD} dB
        - **SSIM:** ≥ {ImageProcessingSystem.SSIM_THRESHOLD}
        - **LC:** ≥ {ImageProcessingSystem.LC_MIN_THRESHOLD}
        - **Edge:** {ImageProcessingSystem.EDGE_MIN_THRESHOLD}-{ImageProcessingSystem.EDGE_MAX_THRESHOLD}
        """)
    
    # Abas principais
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Upload", 
        "🔧 Processamento", 
        "📊 Análise", 
        "📈 Métricas", 
        "📄 Relatório",
        "⚡ Híbrido"
    ])
    
    # ========================================================================
    # TAB 1: UPLOAD
    # ========================================================================
    with tab1:
        st.header("📤 Importação de Imagem")
        
        uploaded_file = st.file_uploader(
            f"Escolha uma imagem (PNG ou JPEG, máx {ImageProcessingSystem.MAX_FILE_SIZE_MB} MB)",
            type=['png', 'jpg', 'jpeg'],
            help=f"Tamanho máximo: {ImageProcessingSystem.MAX_FILE_SIZE_MB} MB. A imagem será normalizada para 512x512px"
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
    # TAB 2: PROCESSAMENTO COM VISUALIZAÇÃO EM TEMPO REAL
    # ========================================================================
    with tab2:
        if st.session_state.normalized_image is None:
            st.warning("⚠️ Por favor, carregue uma imagem primeiro na aba 'Upload'")
        else:
            st.header("🔧 Processamento de Imagem com Preview em Tempo Real")
            
            # Layout: Controles à esquerda, visualização à direita
            col_controls, col_preview = st.columns([1, 1])
            
            with col_controls:
                # Pré-processamento
                with st.expander("🔹 1. Pré-processamento", expanded=True):
                    filter_type = st.selectbox(
                        "Tipo de Filtro",
                        ["Gaussiano", "Mediana"],
                        help="Filtro de suavização"
                    )
                    
                    kernel_radius = st.slider(
                        "Raio do Kernel",
                        min_value=1,
                        max_value=9,
                        value=3,
                        step=2,
                        help="Tamanho do kernel (deve ser ímpar)"
                    )
                    
                    sigma = st.slider(
                        "Sigma (0.5-2.0)",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        help="Desvio padrão para filtro gaussiano"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👁️ Preview", key="preview_preproc", use_container_width=True):
                            ImageProcessingSystem.apply_preprocessing(filter_type, kernel_radius, sigma)
                    
                    with col2:
                        if st.button("✅ Aplicar", key="apply_preproc", use_container_width=True):
                            if ImageProcessingSystem.apply_preprocessing(filter_type, kernel_radius, sigma):
                                ImageProcessingSystem.confirm_preview()
                
                # Nitidez
                with st.expander("🔹 2. Realce de Nitidez"):
                    sharp_method = st.selectbox(
                        "Método",
                        ["Laplaciano", "Sobel", "Alta Frequência"],
                        help="Método de realce de nitidez"
                    )
                    
                    weight = st.slider(
                        "Peso",
                        min_value=0.1,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        help="Peso do filtro (valores altos podem causar oversharpening)"
                    )
                    
                    threshold = st.slider(
                        "Limiar (Sobel)",
                        min_value=10,
                        max_value=200,
                        value=50,
                        help="Limiar para detecção de bordas"
                    )
                    
                    intensity = st.slider(
                        "Intensidade (≤1.5)",
                        min_value=1.0,
                        max_value=1.5,
                        value=1.2,
                        step=0.1,
                        help="Intensidade do realce"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👁️ Preview", key="preview_sharp", use_container_width=True):
                            ImageProcessingSystem.apply_sharpening(sharp_method, weight, threshold, intensity)
                    
                    with col2:
                        if st.button("✅ Aplicar", key="apply_sharp", use_container_width=True):
                            if ImageProcessingSystem.apply_sharpening(sharp_method, weight, threshold, intensity):
                                ImageProcessingSystem.confirm_preview()
                
                # Contraste
                with st.expander("🔹 3. Realce de Contraste"):
                    contrast_method = st.selectbox(
                        "Método",
                        ["CLAHE (Local)", "Equalização Global"],
                        help="Método de realce de contraste"
                    )
                    
                    clip_limit = st.slider(
                        "Limite de Clipagem (2.0-3.0)",
                        min_value=2.0,
                        max_value=3.0,
                        value=2.5,
                        step=0.1,
                        help="Limite para CLAHE"
                    )
                    
                    tile_size = st.select_slider(
                        "Tamanho do Bloco",
                        options=[4, 8, 16],
                        value=8,
                        help="Tamanho do bloco para CLAHE"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👁️ Preview", key="preview_contrast", use_container_width=True):
                            ImageProcessingSystem.apply_contrast_enhancement(contrast_method, clip_limit, tile_size)
                    
                    with col2:
                        if st.button("✅ Aplicar", key="apply_contrast", use_container_width=True):
                            if ImageProcessingSystem.apply_contrast_enhancement(contrast_method, clip_limit, tile_size):
                                ImageProcessingSystem.confirm_preview()
                
                st.divider()
                
                # Botão para calcular métricas
                if st.button("📊 Calcular Métricas", type="primary", use_container_width=True):
                    ImageProcessingSystem.calculate_metrics()
            
            with col_preview:
                st.subheader("📺 Visualização em Tempo Real")
                
                # Comparação lado a lado
                if st.session_state.preview_image is not None:
                    # Tabs para diferentes visualizações
                    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Comparação", "Preview", "Diferença"])
                    
                    with preview_tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(st.session_state.processed_image, caption="Atual", use_container_width=True)
                        with col2:
                            st.image(st.session_state.preview_image, caption="Preview", use_container_width=True)
                    
                    with preview_tab2:
                        st.image(st.session_state.preview_image, caption="Preview em Tela Cheia", use_container_width=True)
                    
                    with preview_tab3:
                        diff = np.abs(
                            st.session_state.processed_image.astype(np.float32) - 
                            st.session_state.preview_image.astype(np.float32)
                        )
                        st.image(diff.astype(np.uint8), caption="Mapa de Diferença", use_container_width=True)
                        
                        # Estatísticas da diferença
                        diff_mean = np.mean(diff)
                        diff_max = np.max(diff)
                        st.caption(f"Diferença média: {diff_mean:.2f} | Diferença máxima: {diff_max:.2f}")
                else:
                    st.info("👆 Ajuste os parâmetros acima e clique em 'Preview' para ver o resultado")
                    st.image(st.session_state.processed_image, caption="Imagem Atual", use_container_width=True)
    
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
            
            # Comparação Global vs Local (se disponível)
            if 'global' in st.session_state.versions and 'local' in st.session_state.versions:
                st.divider()
                st.subheader("🌍 Comparação Global vs Local")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(st.session_state.versions['global'], caption="Equalização Global", use_container_width=True)
                
                with col2:
                    st.image(st.session_state.versions['local'], caption="CLAHE Local", use_container_width=True)
            
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
            
            m = st.session_state.metrics
            
            with col1:
                status = "✅" if m['psnr_ok'] else "⚠️"
                st.metric(
                    label=f"{status} PSNR",
                    value=f"{m['PSNR']:.2f} dB",
                    help="Peak Signal-to-Noise Ratio"
                )
                st.caption(f"Alvo: ≥ {ImageProcessingSystem.PSNR_THRESHOLD} dB")
            
            with col2:
                status = "✅" if m['ssim_ok'] else "⚠️"
                st.metric(
                    label=f"{status} SSIM",
                    value=f"{m['SSIM']:.3f}",
                    help="Structural Similarity Index"
                )
                st.caption(f"Alvo: ≥ {ImageProcessingSystem.SSIM_THRESHOLD}")
            
            with col3:
                status = "✅" if m['lc_ok'] else "⚠️"
                st.metric(
                    label=f"{status} LC",
                    value=f"{m['LC']:.3f}",
                    help="Local Contrast"
                )
                st.caption(f"Alvo: ≥ {ImageProcessingSystem.LC_MIN_THRESHOLD}")
            
            with col4:
                status = "✅" if m['edge_ok'] else "⚠️"
                st.metric(
                    label=f"{status} Edge",
                    value=f"{m['Edge_Sharpness']:.3f}",
                    help="Edge Sharpness"
                )
                st.caption(f"Alvo: {ImageProcessingSystem.EDGE_MIN_THRESHOLD}-{ImageProcessingSystem.EDGE_MAX_THRESHOLD}")
            
            st.divider()
            
            # Validação
            all_ok = m['psnr_ok'] and m['ssim_ok'] and m['lc_ok'] and m['edge_ok']
            
            if all_ok:
                st.success("✅ Todas as métricas estão dentro dos parâmetros especificados!")
            else:
                st.warning("⚠️ Algumas métricas estão fora dos parâmetros. Veja as recomendações abaixo:")
                
                if not m['psnr_ok']:
                    st.error(f"❌ PSNR: {m['PSNR']:.2f} dB < {ImageProcessingSystem.PSNR_THRESHOLD} dB - Reduzir intensidade de processamento")
                if not m['ssim_ok']:
                    st.error(f"❌ SSIM: {m['SSIM']:.3f} < {ImageProcessingSystem.SSIM_THRESHOLD} - Preservar melhor a estrutura original")
                if not m['lc_ok']:
                    st.error(f"❌ LC: {m['LC']:.3f} < {ImageProcessingSystem.LC_MIN_THRESHOLD} - Aumentar contraste local (CLAHE)")
                if not m['edge_ok']:
                    st.error(f"❌ Edge: {m['Edge_Sharpness']:.3f} fora de [{ImageProcessingSystem.EDGE_MIN_THRESHOLD}, {ImageProcessingSystem.EDGE_MAX_THRESHOLD}] - Ajustar nitidez")
            
            st.divider()
            
            # Gráfico de métricas
            st.subheader("📊 Visualização das Métricas")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # PSNR
            ax1 = axes[0, 0]
            color = 'green' if m['psnr_ok'] else 'orange'
            ax1.bar(['PSNR'], [m['PSNR']], color=color, alpha=0.7)
            ax1.axhline(y=ImageProcessingSystem.PSNR_THRESHOLD, color='red', linestyle='--', label=f'Alvo: {ImageProcessingSystem.PSNR_THRESHOLD} dB')
            ax1.set_ylabel('dB')
            ax1.set_title('PSNR (Peak Signal-to-Noise Ratio)', fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # SSIM
            ax2 = axes[0, 1]
            color = 'green' if m['ssim_ok'] else 'orange'
            ax2.bar(['SSIM'], [m['SSIM']], color=color, alpha=0.7)
            ax2.axhline(y=ImageProcessingSystem.SSIM_THRESHOLD, color='red', linestyle='--', label=f'Alvo: {ImageProcessingSystem.SSIM_THRESHOLD}')
            ax2.set_ylim([0, 1])
            ax2.set_title('SSIM (Structural Similarity Index)', fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # LC
            ax3 = axes[1, 0]
            color = 'green' if m['lc_ok'] else 'orange'
            ax3.bar(['LC'], [m['LC']], color=color, alpha=0.7)
            ax3.axhline(y=ImageProcessingSystem.LC_MIN_THRESHOLD, color='red', linestyle='--', label=f'Mínimo: {ImageProcessingSystem.LC_MIN_THRESHOLD}')
            ax3.set_title('LC (Local Contrast)', fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            # Edge Sharpness
            ax4 = axes[1, 1]
            color = 'green' if m['edge_ok'] else 'orange'
            ax4.bar(['Edge Sharpness'], [m['Edge_Sharpness']], color=color, alpha=0.7)
            ax4.axhline(y=ImageProcessingSystem.EDGE_MIN_THRESHOLD, color='red', linestyle='--', label=f'Mín: {ImageProcessingSystem.EDGE_MIN_THRESHOLD}')
            ax4.axhline(y=ImageProcessingSystem.EDGE_MAX_THRESHOLD, color='red', linestyle='--', label=f'Máx: {ImageProcessingSystem.EDGE_MAX_THRESHOLD}')
            ax4.set_ylim([0, max(0.3, m['Edge_Sharpness'] * 1.2)])
            ax4.set_title('Edge Sharpness', fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.divider()
            
            # Detalhes técnicos
            with st.expander("📋 Detalhes Técnicos das Métricas"):
                st.markdown(f"""
                ### PSNR (Peak Signal-to-Noise Ratio)
                Mede a razão entre o sinal máximo possível e a potência do ruído.
                - **Excelente:** > 40 dB
                - **Bom:** 30-40 dB
                - **Aceitável:** 20-30 dB
                - **Alvo do sistema:** ≥ {ImageProcessingSystem.PSNR_THRESHOLD} dB
                
                ### SSIM (Structural Similarity Index)
                Mede a similaridade estrutural entre duas imagens.
                - **Excelente:** > 0.95
                - **Bom:** 0.85-0.95
                - **Aceitável:** 0.70-0.85
                - **Alvo do sistema:** ≥ {ImageProcessingSystem.SSIM_THRESHOLD}
                
                ### LC (Local Contrast)
                Mede a variação local de intensidade na imagem.
                Valores mais altos indicam maior contraste local.
                - **Alvo do sistema:** ≥ {ImageProcessingSystem.LC_MIN_THRESHOLD}
                
                ### Edge Sharpness
                Mede a nitidez das bordas na imagem através da densidade de pixels de borda.
                - **Alvo do sistema:** {ImageProcessingSystem.EDGE_MIN_THRESHOLD} - {ImageProcessingSystem.EDGE_MAX_THRESHOLD}
                - Valores muito baixos: bordas pouco definidas
                - Valores muito altos: possível oversharpening
                """)
    
    # ========================================================================
    # TAB 5: RELATÓRIO
    # ========================================================================
    with tab5:
        st.header("📄 Geração de Relatório")
        
        if st.session_state.processed_image is None:
            st.warning("⚠️ Processe uma imagem primeiro")
        else:
            # Diagrama de fluxo
            st.subheader("📊 Diagrama de Fluxo do Processamento")
            st.code(ImageProcessingSystem.generate_flowchart(), language=None)
            
            st.divider()
            
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
                    m = st.session_state.metrics
                    st.write(f"- PSNR: {m['PSNR']:.2f} dB {'✅' if m['psnr_ok'] else '❌'}")
                    st.write(f"- SSIM: {m['SSIM']:.3f} {'✅' if m['ssim_ok'] else '❌'}")
                    st.write(f"- LC: {m['LC']:.3f} {'✅' if m['lc_ok'] else '❌'}")
                    st.write(f"- Edge: {m['Edge_Sharpness']:.3f} {'✅' if m['edge_ok'] else '❌'}")
            
            st.divider()
            
            # Opções adicionais
            st.subheader("⚙️ Opções Adicionais")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Resetar Sistema", use_container_width=True):
                    st.session_state.original_image = None
                    st.session_state.processed_image = None
                    st.session_state.normalized_image = None
                    st.session_state.preview_image = None
                    st.session_state.versions = {}
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
    # TAB 6: PROCESSAMENTO HÍBRIDO
    # ========================================================================
    with tab6:
        if st.session_state.normalized_image is None:
            st.warning("⚠️ Por favor, carregue uma imagem primeiro na aba 'Upload'")
        else:
            st.header("⚡ Processamento Híbrido")
            st.markdown("""
            O processamento híbrido combina suavização, realce de contraste e nitidez em uma única operação otimizada,
            com detecção automática de oversharpening e ajuste de parâmetros.
            """)
            
            col_controls, col_preview = st.columns([1, 1])
            
            with col_controls:
                st.subheader("🎛️ Parâmetros do Pipeline")
                
                # Suavização
                st.markdown("**1. Suavização Gaussiana**")
                hybrid_sigma = st.slider(
                    "Sigma",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key="hybrid_sigma"
                )
                
                st.divider()
                
                # CLAHE
                st.markdown("**2. CLAHE (Contraste Local)**")
                hybrid_clip = st.slider(
                    "Clip Limit",
                    min_value=2.0,
                    max_value=3.0,
                    value=2.5,
                    step=0.1,
                    key="hybrid_clip"
                )
                
                hybrid_tile = st.select_slider(
                    "Tile Size",
                    options=[4, 8, 16],
                    value=8,
                    key="hybrid_tile"
                )
                
                st.divider()
                
                # Nitidez
                st.markdown("**3. Realce de Nitidez**")
                hybrid_sharp_method = st.selectbox(
                    "Método",
                    ["Laplaciano", "Alta Frequência"],
                    key="hybrid_sharp"
                )
                
                hybrid_weight = st.slider(
                    "Peso",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    key="hybrid_weight",
                    help="Será ajustado automaticamente se houver risco de oversharpening"
                )
                
                hybrid_intensity = st.slider(
                    "Intensidade",
                    min_value=1.0,
                    max_value=1.5,
                    value=1.2,
                    step=0.1,
                    key="hybrid_intensity"
                )
                
                st.divider()
                
                if st.button("⚡ Executar Pipeline Híbrido", type="primary", use_container_width=True):
                    ImageProcessingSystem.apply_hybrid_processing(
                        hybrid_sigma,
                        hybrid_clip,
                        hybrid_tile,
                        hybrid_sharp_method,
                        hybrid_weight,
                        hybrid_intensity
                    )
            
            with col_preview:
                st.subheader("📺 Resultado do Processamento Híbrido")
                
                if st.session_state.processed_image is not None:
                    st.image(st.session_state.processed_image, caption="Resultado Final", use_container_width=True)
                    
                    # Mostrar comparação com original
                    with st.expander("🔍 Comparar com Original"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(st.session_state.normalized_image, caption="Original", use_container_width=True)
                        with col2:
                            st.image(st.session_state.processed_image, caption="Híbrido", use_container_width=True)
                else:
                    st.info("👆 Configure os parâmetros e execute o pipeline para ver o resultado")
                
                # Informações sobre o processamento híbrido
                with st.expander("ℹ️ Sobre o Processamento Híbrido"):
                    st.markdown("""
                    ### Como funciona o pipeline híbrido?
                    
                    **Etapa 1: Suavização**
                    - Remove ruído usando filtro Gaussiano
                    - Preserva estruturas importantes
                    
                    **Etapa 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)**
                    - Realça contraste local de forma adaptativa
                    - Evita amplificação excessiva de ruído
                    
                    **Etapa 3: Realce de Nitidez**
                    - Aplica filtros de alta frequência
                    - Sistema detecta automaticamente risco de oversharpening
                    - Ajusta parâmetros se necessário
                    
                    ### Proteção Anti-Oversharpening
                    
                    O sistema analisa a densidade de bordas após o CLAHE:
                    - Se densidade > 0.20: ajusta peso/intensidade automaticamente
                    - Emite aviso ao usuário
                    - Garante qualidade final das métricas
                    
                    ### Vantagens
                    
                    ✅ Pipeline otimizado (3 operações em 1)
                    ✅ Detecção inteligente de oversharpening
                    ✅ Ajuste automático de parâmetros
                    ✅ Resultados consistentes e reproduzíveis
                    """)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p><strong>Sistema de Processamento de Imagens v2.0</strong></p>
        <p>Desenvolvido com Python, OpenCV, scikit-image e Streamlit</p>
        <p>📋 Laplaciano 3x3 | 📊 Métricas validadas | ⚡ Pipeline híbrido | 🛡️ Anti-oversharpening</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# EXECUTAR APLICAÇÃO
# ============================================================================

if __name__ == "__main__":
    main()