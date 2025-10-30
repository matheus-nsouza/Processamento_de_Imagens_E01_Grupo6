"""
Sistema de Processamento e Análise de Imagens
Versão Streamlit v3.0

Instalação:
pip install streamlit opencv-python pillow numpy matplotlib scikit-image scikit-learn reportlab

Execução:
streamlit run sistema_processamento_imagens_v3.py

Autor: Sistema de Processamento de Imagens
Data: 2025
Versão: 4.0
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
    page_title="Sistema de Processamento de Imagens v3.0",
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
    
    # Parâmetros padrão
    DEFAULT_PARAMS = {
        'filter_type': 'Gaussiano',
        'kernel_radius': 3,
        'sigma': 1.0,
        'sharp_method': 'Laplaciano',
        'weight': 1.0,
        'threshold': 50,
        'intensity': 1.2,
        'contrast_method': 'CLAHE (Local)',
        'clip_limit': 2.5,
        'tile_size': 8,
        'hybrid_sigma': 1.0,
        'hybrid_clip': 2.5,
        'hybrid_tile': 8,
        'hybrid_sharp_method': 'Laplaciano',
        'hybrid_weight': 1.0,
        'hybrid_intensity': 1.2
    }
    
    def __init__(self):
        if 'initialized' not in st.session_state:
            st.session_state.original_image = None
            st.session_state.processed_image = None
            st.session_state.normalized_image = None
            st.session_state.preview_image = None
            st.session_state.versions = {}
            st.session_state.history = []
            st.session_state.metrics = {}
            st.session_state.user = "Operador"
            st.session_state.image_history = []
            st.session_state.initialized = True
    
    @staticmethod
    def log_action(action):
        """Registra ação no histórico"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {st.session_state.user}: {action}"
        st.session_state.history.insert(0, entry)
    
    @staticmethod
    def save_state():
        """Salva estado atual para undo"""
        if st.session_state.processed_image is not None:
            st.session_state.image_history.append(st.session_state.processed_image.copy())
            if len(st.session_state.image_history) > 10:
                st.session_state.image_history.pop(0)
    
    @staticmethod
    def undo_last_change():
        """Reverte última alteração"""
        if len(st.session_state.image_history) > 0:
            st.session_state.processed_image = st.session_state.image_history.pop()
            st.session_state.preview_image = st.session_state.processed_image.copy()
            ImageProcessingSystem.log_action("Última alteração revertida")
            return True
        return False
    
    @staticmethod
    def reset_to_defaults():
        """Restaura todos os parâmetros aos valores padrão"""
        # Limpar as keys dos widgets para forçar recriação com valores padrão
        keys_to_reset = [
            'filter_type', 'kernel_radius', 'sigma',
            'sharp_method', 'weight', 'threshold', 'intensity',
            'contrast_method', 'clip_limit', 'tile_size',
            'use_smoothing_hyb', 'hybrid_sigma',
            'use_clahe_hyb', 'hybrid_clip', 'hybrid_tile',
            'use_sharpening_hyb', 'hybrid_sharp_method', 'hybrid_weight', 'hybrid_intensity'
        ]
        
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        
        ImageProcessingSystem.log_action("Parâmetros restaurados aos padrões")
        return True
    
    @staticmethod
    def load_image(uploaded_file):
        """Carrega e normaliza imagem para 512x512px"""
        try:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > ImageProcessingSystem.MAX_FILE_SIZE_MB:
                st.error(f"❌ Arquivo muito grande ({file_size_mb:.2f} MB). Máximo: {ImageProcessingSystem.MAX_FILE_SIZE_MB} MB")
                return False
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("❌ Não foi possível carregar a imagem.")
                return False
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.session_state.original_image = img.copy()
            st.session_state.normalized_image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            st.session_state.processed_image = st.session_state.normalized_image.copy()
            st.session_state.preview_image = st.session_state.normalized_image.copy()
            st.session_state.image_history = []
            
            ImageProcessingSystem.log_action(f"Imagem '{uploaded_file.name}' carregada ({file_size_mb:.2f} MB)")
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
            
            if sigma < 0.5 or sigma > 2.0:
                st.error("❌ Sigma deve estar entre 0.5 e 2.0")
                return False
            
            if kernel_radius % 2 == 0:
                kernel_radius += 1
            
            img = st.session_state.processed_image.copy()
            
            if filter_type == 'Gaussiano':
                filtered = np.zeros_like(img, dtype=np.float64)
                for i in range(3):
                    filtered[:,:,i] = gaussian(img[:,:,i], sigma=sigma, preserve_range=True)
                filtered = np.clip(filtered, 0, 255).astype(np.uint8)
            elif filter_type == 'Mediana':
                filtered = cv2.medianBlur(img, kernel_radius)
            
            st.session_state.preview_image = filtered
            ImageProcessingSystem.log_action(f"Pré-processamento: {filter_type}, raio={kernel_radius}, sigma={sigma}")
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
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
                laplacian = np.uint8(np.absolute(laplacian))
                sharpened = np.zeros_like(img)
                for i in range(3):
                    sharpened[:,:,i] = cv2.addWeighted(img[:,:,i], 1.0, laplacian, weight, 0)
                
            elif method == 'Sobel':
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx**2 + sobely**2)
                sobel = np.uint8(sobel)
                _, sobel = cv2.threshold(sobel, threshold, 255, cv2.THRESH_BINARY)
                sharpened = np.zeros_like(img)
                for i in range(3):
                    sharpened[:,:,i] = cv2.addWeighted(img[:,:,i], 1.0, sobel, weight, 0)
                
            elif method == 'Alta Frequência':
                blurred = cv2.GaussianBlur(img, (0, 0), 3)
                sharpened = cv2.addWeighted(img, intensity, blurred, -(intensity-1), 0)
            
            st.session_state.preview_image = np.clip(sharpened, 0, 255).astype(np.uint8)
            ImageProcessingSystem.log_action(f"Nitidez: {method}, peso={weight}")
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
            
            if clip_limit < 2.0 or clip_limit > 3.0:
                st.error("❌ Clip limit deve estar entre 2.0 e 3.0")
                return False
            
            img = st.session_state.processed_image.copy()
            
            if method == 'CLAHE (Local)':
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                st.session_state.versions['local'] = enhanced.copy()
                
            elif method == 'Equalização Global':
                ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                y = cv2.equalizeHist(y)
                ycrcb = cv2.merge([y, cr, cb])
                enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
                st.session_state.versions['global'] = enhanced.copy()
            
            st.session_state.preview_image = enhanced
            ImageProcessingSystem.log_action(f"Contraste: {method}, clip={clip_limit}")
            return True
                
        except Exception as e:
            st.error(f"❌ Erro ao aplicar contraste: {str(e)}")
            return False
    
    @staticmethod
    def apply_hybrid_processing(use_smoothing, sigma, use_clahe, clip_limit, tile_size, 
                                use_sharpening, sharp_method, weight, intensity):
        """Função híbrida: pipeline opcional de processamento"""
        try:
            if st.session_state.normalized_image is None:
                st.warning("⚠️ Carregue uma imagem primeiro!")
                return False
            
            if not (use_smoothing or use_clahe or use_sharpening):
                st.warning("⚠️ Selecione pelo menos uma técnica!")
                return False
            
            img = st.session_state.normalized_image.copy()
            techniques_used = []
            
            # Suavização
            if use_smoothing:
                smoothed = np.zeros_like(img, dtype=np.float64)
                for i in range(3):
                    smoothed[:,:,i] = gaussian(img[:,:,i], sigma=sigma, preserve_range=True)
                img = np.clip(smoothed, 0, 255).astype(np.uint8)
                techniques_used.append(f"Suavização (σ={sigma})")
            
            # CLAHE
            if use_clahe:
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                techniques_used.append(f"CLAHE (clip={clip_limit})")
            
            # Verificar oversharpening
            adjusted_weight = weight
            adjusted_intensity = intensity
            oversharpening_risk = False
            
            if use_sharpening:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                edges_before = cv2.Canny(gray_img, 100, 200)
                edge_density = np.sum(edges_before > 0) / edges_before.size
                
                if edge_density > 0.20:
                    adjusted_weight = min(weight, 1.0)
                    adjusted_intensity = min(intensity, 1.2)
                    oversharpening_risk = True
                
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                if sharp_method == 'Laplaciano':
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
                    laplacian = np.uint8(np.absolute(laplacian))
                    sharpened = np.zeros_like(img)
                    for i in range(3):
                        sharpened[:,:,i] = cv2.addWeighted(img[:,:,i], 1.0, laplacian, adjusted_weight, 0)
                    img = sharpened
                elif sharp_method == 'Alta Frequência':
                    blurred = cv2.GaussianBlur(img, (0, 0), 3)
                    img = cv2.addWeighted(img, adjusted_intensity, blurred, -(adjusted_intensity-1), 0)
                
                techniques_used.append(f"Nitidez {sharp_method}")
            
            result = np.clip(img, 0, 255).astype(np.uint8)
            st.session_state.processed_image = result
            st.session_state.preview_image = result
            
            log_msg = f"Pipeline híbrido: {' → '.join(techniques_used)}"
            if oversharpening_risk:
                log_msg += f" [Ajustado: {weight}→{adjusted_weight}]"
                st.warning("⚠️ Risco de oversharpening! Parâmetros ajustados.")
            
            ImageProcessingSystem.log_action(log_msg)
            st.success("✅ Pipeline híbrido aplicado!")
            return True
            
        except Exception as e:
            st.error(f"❌ Erro no híbrido: {str(e)}")
            return False
    
    @staticmethod
    def confirm_preview():
        """Confirma preview"""
        if st.session_state.preview_image is not None:
            ImageProcessingSystem.save_state()
            st.session_state.processed_image = st.session_state.preview_image.copy()
            ImageProcessingSystem.log_action("Preview confirmado")
            st.success("✅ Aplicado!")
            return True
        return False
    
    @staticmethod
    def calculate_metrics():
        """Calcula métricas"""
        try:
            if st.session_state.normalized_image is None or st.session_state.processed_image is None:
                st.warning("⚠️ Carregue e processe uma imagem!")
                return False
            
            original = st.session_state.normalized_image.astype(np.float64)
            processed = st.session_state.processed_image.astype(np.float64)
            
            mse = np.mean((original - processed) ** 2)
            psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
            
            ssim = metrics.structural_similarity(original, processed, channel_axis=2, data_range=255.0)
            
            gray_processed = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_RGB2GRAY)
            lc = np.std(gray_processed) / (np.mean(gray_processed) + 1e-10)
            
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
            
            ImageProcessingSystem.log_action("Métricas calculadas")
            st.success("✅ Métricas calculadas!")
            return True
                
        except Exception as e:
            st.error(f"❌ Erro ao calcular métricas: {str(e)}")
            return False
    
    @staticmethod
    def generate_pdf_report():
        """Gera relatório PDF"""
        try:
            if st.session_state.processed_image is None:
                st.warning("⚠️ Processe uma imagem primeiro!")
                return None
            
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=A4)
            width, height = A4
            
            c.setFont("Helvetica-Bold", 20)
            c.drawString(50, height - 50, "Relatório de Processamento de Imagens v3.0")
            
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 70, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            c.drawString(50, height - 85, f"Usuário: {st.session_state.user}")
            c.line(50, height - 95, width - 50, height - 95)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_orig:
                cv2.imwrite(tmp_orig.name, cv2.cvtColor(st.session_state.normalized_image, cv2.COLOR_RGB2BGR))
                temp_orig_path = tmp_orig.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_proc:
                cv2.imwrite(tmp_proc.name, cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_RGB2BGR))
                temp_proc_path = tmp_proc.name
            
            y_position = height - 320
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position + 20, "Análise Visual:")
            
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y_position - 10, "Original:")
            c.drawImage(temp_orig_path, 50, y_position - 190, width=200, height=200)
            
            c.drawString(300, y_position - 10, "Processada:")
            c.drawImage(temp_proc_path, 300, y_position - 190, width=200, height=200)
            
            y_position -= 230
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Métricas Quantitativas:")
            
            c.setFont("Helvetica", 11)
            y_position -= 25
            
            if st.session_state.metrics:
                m = st.session_state.metrics
                metrics_lines = [
                    f"PSNR: {m['PSNR']:.2f} dB {'✓' if m['psnr_ok'] else '✗'} (Alvo: >= 30 dB)",
                    f"SSIM: {m['SSIM']:.3f} {'✓' if m['ssim_ok'] else '✗'} (Alvo: >= 0.85)",
                    f"LC: {m['LC']:.3f} {'✓' if m['lc_ok'] else '✗'} (Alvo: >= 0.12)",
                    f"Edge: {m['Edge_Sharpness']:.3f} {'✓' if m['edge_ok'] else '✗'} (Alvo: 0.03-0.25)"
                ]
                
                for line in metrics_lines:
                    c.drawString(70, y_position, line)
                    y_position -= 20
                
                y_position -= 20
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, y_position, "Conclusões:")
                c.setFont("Helvetica", 10)
                y_position -= 20
                
                all_ok = m['psnr_ok'] and m['ssim_ok'] and m['lc_ok'] and m['edge_ok']
                
                if all_ok:
                    c.drawString(70, y_position, "✓ APROVADO - Métricas dentro dos parâmetros")
                    y_position -= 15
                    c.drawString(70, y_position, "Imagem atende critérios de qualidade")
                else:
                    c.drawString(70, y_position, "✗ REPROVADO - Ajustes necessários:")
                    y_position -= 15
                    if not m['psnr_ok']:
                        c.drawString(85, y_position, "• PSNR baixo: reduzir intensidade")
                        y_position -= 12
                    if not m['ssim_ok']:
                        c.drawString(85, y_position, "• SSIM baixo: preservar estrutura")
                        y_position -= 12
                    if not m['lc_ok']:
                        c.drawString(85, y_position, "• LC baixo: aumentar CLAHE")
                        y_position -= 12
                    if not m['edge_ok']:
                        c.drawString(85, y_position, "• Edge fora: ajustar nitidez")
                        y_position -= 12
            
            c.showPage()
            y_position = height - 50
            
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Histórico:")
            
            c.setFont("Helvetica", 8)
            y_position -= 20
            
            for entry in st.session_state.history[:30]:
                if y_position < 50:
                    c.showPage()
                    y_position = height - 50
                c.drawString(60, y_position, entry[:100])
                y_position -= 12
            
            c.save()
            
            os.unlink(temp_orig_path)
            os.unlink(temp_proc_path)
            
            ImageProcessingSystem.log_action("PDF gerado")
            
            pdf_buffer.seek(0)
            return pdf_buffer.getvalue()
                
        except Exception as e:
            st.error(f"❌ Erro ao gerar PDF: {str(e)}")
            return None

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

def main():
    """Função principal"""
    
    sistema = ImageProcessingSystem()
    
    st.title("🖼️ Sistema de Processamento de Imagens v3.0")
    st.markdown("### Análise e realce avançado com métricas quantitativas")
    
    with st.sidebar:
        st.header("👤 Usuário")
        st.session_state.user = st.text_input("Nome", value=st.session_state.user)
        user_role = st.selectbox("Nível", ["Operador", "Administrador"])
        
        st.divider()
        
        st.header("📊 Sistema")
        st.info(f"""
        **Versão:** 2.0
        **Formatos:** PNG, JPEG
        **Resolução:** 512×512px
        **Limite:** {ImageProcessingSystem.MAX_FILE_SIZE_MB} MB
        **Status:** 🟢 Online
        """)
        
        st.divider()
        
        st.header("🎯 Critérios")
        st.markdown(f"""
        • **PSNR:** ≥ {ImageProcessingSystem.PSNR_THRESHOLD} dB
        • **SSIM:** ≥ {ImageProcessingSystem.SSIM_THRESHOLD}
        • **LC:** ≥ {ImageProcessingSystem.LC_MIN_THRESHOLD}
        • **Edge:** {ImageProcessingSystem.EDGE_MIN_THRESHOLD}-{ImageProcessingSystem.EDGE_MAX_THRESHOLD}
        """)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Upload", "🔧 Processamento", "📊 Análise", 
        "📈 Métricas", "📄 Relatório", "⚡ Híbrido"
    ])
    
    # TAB 1: UPLOAD
    with tab1:
        st.header("📤 Importação")
        
        uploaded_file = st.file_uploader(
            f"Escolha uma imagem (máx {ImageProcessingSystem.MAX_FILE_SIZE_MB} MB)",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Carregar", type="primary"):
                if ImageProcessingSystem.load_image(uploaded_file):
                    st.balloons()
        
        if st.session_state.normalized_image is not None:
            st.subheader("✅ Carregada")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(st.session_state.original_image, caption="Original", use_container_width=True)
                st.caption(f"{st.session_state.original_image.shape[1]}×{st.session_state.original_image.shape[0]}")
            
            with col2:
                st.image(st.session_state.normalized_image, caption="Normalizada (512×512)", use_container_width=True)
                st.caption("Pronta para processamento")
    
    # TAB 2: PROCESSAMENTO
    with tab2:
        if st.session_state.normalized_image is None:
            st.warning("⚠️ Carregue uma imagem na aba 'Upload'")
        else:
            st.header("🔧 Processamento")
            
            col_undo, col_reset, col_calc = st.columns([1, 1, 2])
            with col_undo:
                if st.button("↩️ Reverter", use_container_width=True):
                    if ImageProcessingSystem.undo_last_change():
                        st.success("✅ Revertido!")
                        st.rerun()
                    else:
                        st.info("ℹ️ Nada para reverter")
            
            with col_reset:
                if st.button("🔄 Restaurar Padrões", use_container_width=True):
                    ImageProcessingSystem.reset_to_defaults()
                    st.success("✅ Padrões restaurados!")
                    st.rerun()
            
            with col_calc:
                if st.button("📊 Calcular Métricas", type="primary", use_container_width=True):
                    ImageProcessingSystem.calculate_metrics()
            
            st.divider()
            
            col_controls, col_preview = st.columns([1, 1])
            
            with col_controls:
                with st.expander("🔹 1. Pré-processamento", expanded=True):
                    filter_type = st.selectbox(
                        "Tipo", 
                        ["Gaussiano", "Mediana"],
                        key="filter_type"
                    )
                    
                    kernel_radius = st.slider(
                        "Raio", 1, 9, 
                        st.session_state.get('kernel_radius', ImageProcessingSystem.DEFAULT_PARAMS['kernel_radius']), 
                        2,
                        key="kernel_radius"
                    )
                    
                    sigma = st.slider(
                        "Sigma", 0.5, 2.0, 
                        st.session_state.get('sigma', ImageProcessingSystem.DEFAULT_PARAMS['sigma']), 
                        0.1,
                        key="sigma"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👁️ Preview", key="prev_prep", use_container_width=True):
                            ImageProcessingSystem.apply_preprocessing(filter_type, kernel_radius, sigma)
                    with col2:
                        if st.button("✅ Aplicar", key="app_prep", use_container_width=True):
                            if ImageProcessingSystem.apply_preprocessing(filter_type, kernel_radius, sigma):
                                ImageProcessingSystem.confirm_preview()
                
                with st.expander("🔹 2. Nitidez"):
                    sharp_method = st.selectbox(
                        "Método", 
                        ["Laplaciano", "Sobel", "Alta Frequência"],
                        key="sharp_method"
                    )
                    
                    weight = st.slider(
                        "Peso", 0.1, 3.0, 
                        st.session_state.get('weight', ImageProcessingSystem.DEFAULT_PARAMS['weight']), 
                        0.1,
                        key="weight"
                    )
                    
                    threshold = st.slider(
                        "Limiar", 10, 200, 
                        st.session_state.get('threshold', ImageProcessingSystem.DEFAULT_PARAMS['threshold']),
                        key="threshold"
                    )
                    
                    intensity = st.slider(
                        "Intensidade", 1.0, 1.5, 
                        st.session_state.get('intensity', ImageProcessingSystem.DEFAULT_PARAMS['intensity']), 
                        0.1,
                        key="intensity"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👁️ Preview", key="prev_sharp", use_container_width=True):
                            ImageProcessingSystem.apply_sharpening(sharp_method, weight, threshold, intensity)
                    with col2:
                        if st.button("✅ Aplicar", key="app_sharp", use_container_width=True):
                            if ImageProcessingSystem.apply_sharpening(sharp_method, weight, threshold, intensity):
                                ImageProcessingSystem.confirm_preview()
                
                with st.expander("🔹 3. Contraste"):
                    contrast_method = st.selectbox(
                        "Método", 
                        ["CLAHE (Local)", "Equalização Global"],
                        key="contrast_method"
                    )
                    
                    clip_limit = st.slider(
                        "Clip Limit", 2.0, 3.0, 
                        st.session_state.get('clip_limit', ImageProcessingSystem.DEFAULT_PARAMS['clip_limit']), 
                        0.1,
                        key="clip_limit"
                    )
                    
                    tile_size = st.select_slider(
                        "Tile Size", 
                        options=[4, 8, 16], 
                        value=st.session_state.get('tile_size', ImageProcessingSystem.DEFAULT_PARAMS['tile_size']),
                        key="tile_size"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👁️ Preview", key="prev_cont", use_container_width=True):
                            ImageProcessingSystem.apply_contrast_enhancement(contrast_method, clip_limit, tile_size)
                    with col2:
                        if st.button("✅ Aplicar", key="app_cont", use_container_width=True):
                            if ImageProcessingSystem.apply_contrast_enhancement(contrast_method, clip_limit, tile_size):
                                ImageProcessingSystem.confirm_preview()
            
            with col_preview:
                st.subheader("📺 Visualização")
                
                if st.session_state.preview_image is not None:
                    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Comparação", "Preview", "Diferença"])
                    
                    with preview_tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(st.session_state.processed_image, caption="Atual", use_container_width=True)
                        with col2:
                            st.image(st.session_state.preview_image, caption="Preview", use_container_width=True)
                    
                    with preview_tab2:
                        st.image(st.session_state.preview_image, caption="Preview", use_container_width=True)
                    
                    with preview_tab3:
                        diff = np.abs(st.session_state.processed_image.astype(np.float32) - 
                                     st.session_state.preview_image.astype(np.float32))
                        st.image(diff.astype(np.uint8), caption="Diferença", use_container_width=True)
                        st.caption(f"Média: {np.mean(diff):.2f} | Máxima: {np.max(diff):.2f}")
                else:
                    st.info("👆 Clique em Preview para visualizar")
                    st.image(st.session_state.processed_image, caption="Atual", use_container_width=True)
    
    # TAB 3: ANÁLISE
    with tab3:
        if st.session_state.processed_image is None:
            st.warning("⚠️ Processe uma imagem primeiro")
        else:
            st.header("📊 Análise Visual")
            
            st.subheader("🔍 Comparação")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(st.session_state.normalized_image, caption="Original", use_container_width=True)
            with col2:
                st.image(st.session_state.processed_image, caption="Processada", use_container_width=True)
            
            if 'global' in st.session_state.versions and 'local' in st.session_state.versions:
                st.divider()
                st.subheader("🌍 Global vs Local")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(st.session_state.versions['global'], caption="Global", use_container_width=True)
                with col2:
                    st.image(st.session_state.versions['local'], caption="Local", use_container_width=True)
            
            st.divider()
            st.subheader("📈 Detalhes")
            
            diff = np.abs(st.session_state.normalized_image.astype(np.float32) - 
                         st.session_state.processed_image.astype(np.float32))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(diff.astype(np.uint8), caption="Diferença", use_container_width=True)
            
            with col2:
                gray_orig = cv2.cvtColor(st.session_state.normalized_image, cv2.COLOR_RGB2GRAY)
                edges_orig = cv2.Canny(gray_orig, 100, 200)
                st.image(edges_orig, caption="Bordas Original", use_container_width=True)
            
            with col3:
                gray_proc = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_RGB2GRAY)
                edges_proc = cv2.Canny(gray_proc, 100, 200)
                st.image(edges_proc, caption="Bordas Processada", use_container_width=True)
            
            st.divider()
            st.subheader("📊 Histogramas")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            for i, color in enumerate(['red', 'green', 'blue']):
                hist = cv2.calcHist([st.session_state.normalized_image], [i], None, [256], [0, 256])
                ax1.plot(hist, color=color, alpha=0.7, label=color.upper())
            ax1.set_title('Original', fontweight='bold')
            ax1.set_xlim([0, 256])
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            for i, color in enumerate(['red', 'green', 'blue']):
                hist = cv2.calcHist([st.session_state.processed_image], [i], None, [256], [0, 256])
                ax2.plot(hist, color=color, alpha=0.7, label=color.upper())
            ax2.set_title('Processada', fontweight='bold')
            ax2.set_xlim([0, 256])
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # TAB 4: MÉTRICAS
    with tab4:
        st.header("📈 Métricas")
        
        if not st.session_state.metrics:
            st.info("ℹ️ Calcule as métricas na aba Processamento")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            m = st.session_state.metrics
            
            with col1:
                st.metric(f"{'✅' if m['psnr_ok'] else '⚠️'} PSNR", f"{m['PSNR']:.2f} dB")
                st.caption(f"Alvo: ≥ {ImageProcessingSystem.PSNR_THRESHOLD} dB")
            
            with col2:
                st.metric(f"{'✅' if m['ssim_ok'] else '⚠️'} SSIM", f"{m['SSIM']:.3f}")
                st.caption(f"Alvo: ≥ {ImageProcessingSystem.SSIM_THRESHOLD}")
            
            with col3:
                st.metric(f"{'✅' if m['lc_ok'] else '⚠️'} LC", f"{m['LC']:.3f}")
                st.caption(f"Alvo: ≥ {ImageProcessingSystem.LC_MIN_THRESHOLD}")
            
            with col4:
                st.metric(f"{'✅' if m['edge_ok'] else '⚠️'} Edge", f"{m['Edge_Sharpness']:.3f}")
                st.caption(f"Alvo: {ImageProcessingSystem.EDGE_MIN_THRESHOLD}-{ImageProcessingSystem.EDGE_MAX_THRESHOLD}")
            
            st.divider()
            
            all_ok = m['psnr_ok'] and m['ssim_ok'] and m['lc_ok'] and m['edge_ok']
            
            if all_ok:
                st.success("✅ Todas as métricas dentro dos parâmetros!")
            else:
                st.warning("⚠️ Ajustes necessários:")
                if not m['psnr_ok']:
                    st.error(f"❌ PSNR: {m['PSNR']:.2f} dB - Reduzir intensidade")
                if not m['ssim_ok']:
                    st.error(f"❌ SSIM: {m['SSIM']:.3f} - Preservar estrutura")
                if not m['lc_ok']:
                    st.error(f"❌ LC: {m['LC']:.3f} - Aumentar contraste")
                if not m['edge_ok']:
                    st.error(f"❌ Edge: {m['Edge_Sharpness']:.3f} - Ajustar nitidez")
            
            st.divider()
            st.subheader("📊 Gráficos")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            ax1 = axes[0, 0]
            ax1.bar(['PSNR'], [m['PSNR']], color='green' if m['psnr_ok'] else 'orange', alpha=0.7)
            ax1.axhline(y=ImageProcessingSystem.PSNR_THRESHOLD, color='red', linestyle='--')
            ax1.set_ylabel('dB')
            ax1.set_title('PSNR', fontweight='bold')
            ax1.grid(alpha=0.3)
            
            ax2 = axes[0, 1]
            ax2.bar(['SSIM'], [m['SSIM']], color='green' if m['ssim_ok'] else 'orange', alpha=0.7)
            ax2.axhline(y=ImageProcessingSystem.SSIM_THRESHOLD, color='red', linestyle='--')
            ax2.set_ylim([0, 1])
            ax2.set_title('SSIM', fontweight='bold')
            ax2.grid(alpha=0.3)
            
            ax3 = axes[1, 0]
            ax3.bar(['LC'], [m['LC']], color='green' if m['lc_ok'] else 'orange', alpha=0.7)
            ax3.axhline(y=ImageProcessingSystem.LC_MIN_THRESHOLD, color='red', linestyle='--')
            ax3.set_title('Local Contrast', fontweight='bold')
            ax3.grid(alpha=0.3)
            
            ax4 = axes[1, 1]
            ax4.bar(['Edge'], [m['Edge_Sharpness']], color='green' if m['edge_ok'] else 'orange', alpha=0.7)
            ax4.axhline(y=ImageProcessingSystem.EDGE_MIN_THRESHOLD, color='red', linestyle='--')
            ax4.axhline(y=ImageProcessingSystem.EDGE_MAX_THRESHOLD, color='red', linestyle='--')
            ax4.set_ylim([0, max(0.3, m['Edge_Sharpness'] * 1.2)])
            ax4.set_title('Edge Sharpness', fontweight='bold')
            ax4.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # TAB 5: RELATÓRIO
    with tab5:
        st.header("📄 Relatório")
        
        if st.session_state.processed_image is None:
            st.warning("⚠️ Processe uma imagem primeiro")
        else:
            st.subheader("📋 Histórico")
            
            if st.session_state.history:
                for entry in st.session_state.history[:20]:
                    st.text(entry)
            else:
                st.info("Nenhuma operação registrada")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Gerar PDF", type="primary", use_container_width=True):
                    pdf_data = ImageProcessingSystem.generate_pdf_report()
                    if pdf_data:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_data,
                            file_name=f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
            
            with col2:
                if st.button("💾 Baixar Imagem", use_container_width=True):
                    img_pil = Image.fromarray(st.session_state.processed_image)
                    buf = io.BytesIO()
                    img_pil.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download PNG",
                        data=buf,
                        file_name=f"processada_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            st.divider()
            
            if st.session_state.metrics:
                st.subheader("📊 Resumo")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Informações:**")
                    st.write(f"• Resolução: 512×512 pixels")
                    st.write(f"• Canais: RGB")
                    st.write(f"• Operações: {len(st.session_state.history)}")
                
                with col2:
                    st.markdown("**Métricas:**")
                    m = st.session_state.metrics
                    st.write(f"• PSNR: {m['PSNR']:.2f} dB {'✅' if m['psnr_ok'] else '❌'}")
                    st.write(f"• SSIM: {m['SSIM']:.3f} {'✅' if m['ssim_ok'] else '❌'}")
                    st.write(f"• LC: {m['LC']:.3f} {'✅' if m['lc_ok'] else '❌'}")
                    st.write(f"• Edge: {m['Edge_Sharpness']:.3f} {'✅' if m['edge_ok'] else '❌'}")
            
            st.divider()
            st.subheader("⚙️ Sistema")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Resetar", use_container_width=True):
                    st.session_state.original_image = None
                    st.session_state.processed_image = None
                    st.session_state.normalized_image = None
                    st.session_state.preview_image = None
                    st.session_state.versions = {}
                    st.session_state.history = []
                    st.session_state.metrics = {}
                    st.session_state.image_history = []
                    st.success("✅ Resetado!")
                    st.rerun()
            
            with col2:
                if st.button("📋 Limpar Histórico", use_container_width=True):
                    st.session_state.history = []
                    st.success("✅ Limpo!")
                    st.rerun()
    
    # TAB 6: HÍBRIDO
    with tab6:
        if st.session_state.normalized_image is None:
            st.warning("⚠️ Carregue uma imagem primeiro")
        else:
            st.header("⚡ Pipeline Híbrido")
            st.markdown("Processamento integrado com seleção opcional de técnicas")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("↩️ Reverter", key="undo_hyb", use_container_width=True):
                    if ImageProcessingSystem.undo_last_change():
                        st.success("✅ Revertido!")
                        st.rerun()
            
            with col2:
                if st.button("🔄 Restaurar", key="reset_hyb", use_container_width=True):
                    ImageProcessingSystem.reset_to_defaults()
                    st.success("✅ Padrões restaurados!")
                    st.rerun()
            
            st.divider()
            
            col_controls, col_preview = st.columns([1, 1])
            
            with col_controls:
                st.subheader("🎛️ Configuração")
                
                use_smoothing = st.checkbox("**1. Suavização Gaussiana**", value=True, key="use_smoothing_hyb")
                if use_smoothing:
                    hybrid_sigma = st.slider(
                        "Sigma", 0.5, 2.0, 
                        st.session_state.get('hybrid_sigma', ImageProcessingSystem.DEFAULT_PARAMS['hybrid_sigma']), 
                        0.1, 
                        key="hybrid_sigma"
                    )
                else:
                    hybrid_sigma = 1.0
                
                st.divider()
                
                use_clahe = st.checkbox("**2. CLAHE**", value=True, key="use_clahe_hyb")
                if use_clahe:
                    hybrid_clip = st.slider(
                        "Clip Limit", 2.0, 3.0, 
                        st.session_state.get('hybrid_clip', ImageProcessingSystem.DEFAULT_PARAMS['hybrid_clip']), 
                        0.1, 
                        key="hybrid_clip"
                    )
                    hybrid_tile = st.select_slider(
                        "Tile Size", 
                        options=[4, 8, 16], 
                        value=st.session_state.get('hybrid_tile', ImageProcessingSystem.DEFAULT_PARAMS['hybrid_tile']), 
                        key="hybrid_tile"
                    )
                else:
                    hybrid_clip = 2.5
                    hybrid_tile = 8
                
                st.divider()
                
                use_sharpening = st.checkbox("**3. Nitidez**", value=True, key="use_sharpening_hyb")
                if use_sharpening:
                    hybrid_sharp_method = st.selectbox(
                        "Método", 
                        ["Laplaciano", "Alta Frequência"], 
                        key="hybrid_sharp_method"
                    )
                    hybrid_weight = st.slider(
                        "Peso", 0.1, 3.0, 
                        st.session_state.get('hybrid_weight', ImageProcessingSystem.DEFAULT_PARAMS['hybrid_weight']), 
                        0.1, 
                        key="hybrid_weight"
                    )
                    hybrid_intensity = st.slider(
                        "Intensidade", 1.0, 1.5, 
                        st.session_state.get('hybrid_intensity', ImageProcessingSystem.DEFAULT_PARAMS['hybrid_intensity']), 
                        0.1, 
                        key="hybrid_intensity"
                    )
                else:
                    hybrid_sharp_method = "Laplaciano"
                    hybrid_weight = 1.0
                    hybrid_intensity = 1.2
                
                st.divider()
                
                if st.button("⚡ Executar Pipeline", type="primary", use_container_width=True):
                    ImageProcessingSystem.save_state()
                    ImageProcessingSystem.apply_hybrid_processing(
                        use_smoothing, hybrid_sigma,
                        use_clahe, hybrid_clip, hybrid_tile,
                        use_sharpening, hybrid_sharp_method, hybrid_weight, hybrid_intensity
                    )
            
            with col_preview:
                st.subheader("📺 Resultado")
                
                if st.session_state.processed_image is not None:
                    st.image(st.session_state.processed_image, caption="Resultado Final", use_container_width=True)
                    
                    with st.expander("🔍 Comparar"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(st.session_state.normalized_image, caption="Original", use_container_width=True)
                        with col2:
                            st.image(st.session_state.processed_image, caption="Híbrido", use_container_width=True)
                else:
                    st.info("👆 Configure e execute o pipeline")
                
                with st.expander("ℹ️ Sobre o Pipeline"):
                    st.markdown("""
                    ### Pipeline Integrado
                    
                    **Técnicas Disponíveis:**
                    
                    1️⃣ **Suavização** - Remove ruído (σ)
                    
                    2️⃣ **CLAHE** - Contraste local adaptativo
                    
                    3️⃣ **Nitidez** - Laplaciano 3×3 ou Alta Freq
                    
                    ### Anti-Oversharpening
                    
                    • Análise automática de densidade de bordas
                    • Ajuste inteligente se densidade > 0.20
                    • Aviso ao usuário quando ajustado
                    
                    ### Vantagens
                    
                    ✅ Escolha livre de técnicas
                    ✅ Pipeline otimizado
                    ✅ Proteção automática
                    ✅ Histórico completo
                    """)
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p><strong>Sistema de Processamento de Imagens v3.0</strong></p>
        <p>Python • OpenCV • scikit-image • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()