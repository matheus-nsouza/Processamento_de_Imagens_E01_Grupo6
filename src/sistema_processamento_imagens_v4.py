"""
Sistema de Processamento e Análise de Imagens
Versão Streamlit v4.0 - Contextualizado para Projeto Véridia

Instalação:
pip install streamlit opencv-python pillow numpy matplotlib scikit-image scikit-learn reportlab

Execução:
streamlit run sistema_processamento_imagens_v3_contextualizado.py

Autor: Sistema de Processamento de Imagens
Data: 2025
Versão: v4.0 Contextualizado
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
    page_title="Sistema de Processamento de Imagens v4.0 - Projeto Véridia",
    page_icon="🏙️",
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
    .context-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .context-card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# PRESETS CONTEXTUAIS PARA PROJETO VÉRIDIA
# ============================================================================

CONTEXTOS_VERIDIIA = {
    "Educação": {
        "icone": "🎓",
        "cor": "#3498DB",
        "descricao": "Otimizado para documentos didáticos e material educacional",
        "casos_uso": [
            "Documentos digitalizados",
            "Quadros brancos fotografados",
            "Livros e apostilas antigas",
            "Anotações manuscritas",
            "Cartazes e infográficos"
        ],
        "parametros": {
            "use_smoothing": False,
            "sigma": 1.0,
            "use_clahe": False,
            "clip_limit": 2.5,
            "tile_size": 8,
            "use_sharpening": True,
            "sharp_method": "Alta Frequência",
            "weight": 2.5,
            "intensity": 1.4
        },
        "justificativa": "Alta nitidez para textos e diagramas, sem suavização para preservar detalhes finos"
    },
    "Saúde": {
        "icone": "🏥",
        "cor": "#E74C3C",
        "descricao": "Otimizado para imagens médicas e diagnósticos",
        "casos_uso": [
            "Radiografias (Raio-X)",
            "Tomografias computadorizadas",
            "Ultrassonografias",
            "Ressonâncias magnéticas",
            "Imagens histopatológicas"
        ],
        "parametros": {
            "use_smoothing": True,
            "sigma": 0.8,
            "use_clahe": True,
            "clip_limit": 3.0,
            "tile_size": 8,
            "use_sharpening": True,
            "sharp_method": "Laplaciano",
            "weight": 1.5,
            "intensity": 1.3
        },
        "justificativa": "CLAHE forte para realçar estruturas anatômicas, suavização leve para reduzir ruído, nitidez moderada"
    },
    "Indústria": {
        "icone": "🏭",
        "cor": "#27AE60",
        "descricao": "Otimizado para controle de qualidade e inspeção visual",
        "casos_uso": [
            "Inspeção de superfícies",
            "Detecção de defeitos",
            "Análise de soldas",
            "Controle dimensional",
            "Verificação de montagem"
        ],
        "parametros": {
            "use_smoothing": True,
            "sigma": 0.5,
            "use_clahe": True,
            "clip_limit": 2.5,
            "tile_size": 16,
            "use_sharpening": True,
            "sharp_method": "Laplaciano",
            "weight": 2.8,
            "intensity": 1.5
        },
        "justificativa": "Máxima nitidez para detecção de defeitos, contraste local para áreas heterogêneas, suavização mínima"
    }
}

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
            st.session_state.contexto_aplicado = None
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
    def aplicar_preset_contextual(contexto_nome):
        """Aplica preset otimizado para o contexto específico"""
        try:
            if st.session_state.normalized_image is None:
                st.warning("⚠️ Carregue uma imagem primeiro!")
                return False
            
            contexto = CONTEXTOS_VERIDIIA[contexto_nome]
            params = contexto["parametros"]
            
            # Aplicar pipeline híbrido com os parâmetros do contexto
            ImageProcessingSystem.save_state()
            
            resultado = ImageProcessingSystem.apply_hybrid_processing(
                use_smoothing=params["use_smoothing"],
                sigma=params["sigma"],
                use_clahe=params["use_clahe"],
                clip_limit=params["clip_limit"],
                tile_size=params["tile_size"],
                use_sharpening=params["use_sharpening"],
                sharp_method=params["sharp_method"],
                weight=params["weight"],
                intensity=params["intensity"]
            )
            
            if resultado:
                st.session_state.contexto_aplicado = contexto_nome
                ImageProcessingSystem.log_action(f"Preset '{contexto_nome}' aplicado")
                st.success(f"✅ Preset {contexto['icone']} {contexto_nome} aplicado com sucesso!")
                return True
            
            return False
            
        except Exception as e:
            st.error(f"❌ Erro ao aplicar preset: {str(e)}")
            return False
    
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
            st.session_state.contexto_aplicado = None
            
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
                result = np.clip(filtered, 0, 255).astype(np.uint8)
            elif filter_type == 'Mediana':
                result = np.zeros_like(img)
                for i in range(3):
                    result[:,:,i] = median(img[:,:,i], footprint=np.ones((kernel_radius, kernel_radius)))
            else:
                return False
            
            st.session_state.preview_image = result
            ImageProcessingSystem.log_action(f"Pré-processamento: {filter_type} (raio={kernel_radius}, σ={sigma})")
            st.success(f"✅ {filter_type} aplicado!")
            return True
            
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
            return False
    
    @staticmethod
    def apply_sharpening(sharp_method, weight, threshold, intensity):
        """Aplica realce de nitidez"""
        try:
            if st.session_state.processed_image is None:
                st.warning("⚠️ Carregue uma imagem primeiro!")
                return False
            
            img = st.session_state.processed_image.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            edges_before = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges_before > 0) / edges_before.size
            
            adjusted_weight = weight
            adjusted_intensity = intensity
            oversharpening_risk = False
            
            if edge_density > 0.20:
                adjusted_weight = min(weight * 0.6, 1.5)
                adjusted_intensity = min(intensity * 0.9, 1.3)
                oversharpening_risk = True
            
            if sharp_method == 'Laplaciano':
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
                laplacian = np.uint8(np.absolute(laplacian))
                sharpened = np.zeros_like(img)
                for i in range(3):
                    sharpened[:,:,i] = cv2.addWeighted(img[:,:,i], 1.0, laplacian, adjusted_weight, 0)
                result = sharpened
            elif sharp_method == 'Sobel':
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx**2 + sobely**2)
                sobel = np.uint8(np.absolute(sobel))
                sharpened = np.zeros_like(img)
                for i in range(3):
                    sharpened[:,:,i] = cv2.addWeighted(img[:,:,i], 1.0, sobel, adjusted_weight, 0)
                result = sharpened
            elif sharp_method == 'Alta Frequência':
                blurred = cv2.GaussianBlur(img, (0, 0), 3)
                result = cv2.addWeighted(img, adjusted_intensity, blurred, -(adjusted_intensity-1), 0)
            else:
                return False
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            st.session_state.preview_image = result
            
            log_msg = f"Nitidez: {sharp_method} (peso={adjusted_weight:.1f})"
            if oversharpening_risk:
                log_msg += f" [Ajustado de {weight:.1f}]"
                st.warning("⚠️ Risco de oversharpening detectado! Parâmetros ajustados automaticamente.")
            
            ImageProcessingSystem.log_action(log_msg)
            st.success(f"✅ Nitidez {sharp_method} aplicada!")
            return True
            
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
            return False
    
    @staticmethod
    def apply_contrast(contrast_method, clip_limit, tile_size):
        """Aplica equalização de contraste"""
        try:
            if st.session_state.processed_image is None:
                st.warning("⚠️ Carregue uma imagem primeiro!")
                return False
            
            img = st.session_state.processed_image.copy()
            
            if contrast_method == 'CLAHE (Local)':
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            elif contrast_method == 'Global':
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                equalized = cv2.equalizeHist(gray)
                result = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            else:
                return False
            
            st.session_state.preview_image = result
            ImageProcessingSystem.log_action(f"Contraste: {contrast_method}")
            st.success(f"✅ Contraste {contrast_method} aplicado!")
            return True
            
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
            return False
    
    @staticmethod
    def apply_hybrid_processing(use_smoothing, sigma, use_clahe, clip_limit, tile_size, 
                                use_sharpening, sharp_method, weight, intensity):
        """Pipeline híbrido com seleção opcional de técnicas"""
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
            c.drawString(50, height - 50, "Relatório de Processamento de Imagens v4.0")
            
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
                    c.drawString(70, y_position, "⚠ ATENÇÃO - Algumas métricas abaixo do esperado")
                    y_position -= 15
                    if not m['psnr_ok']:
                        c.drawString(70, y_position, "- PSNR baixo: considere reduzir processamento")
                        y_position -= 15
                    if not m['ssim_ok']:
                        c.drawString(70, y_position, "- SSIM baixo: estrutura muito alterada")
                        y_position -= 15
                    if not m['lc_ok']:
                        c.drawString(70, y_position, "- LC baixo: aplicar CLAHE")
                        y_position -= 15
                    if not m['edge_ok']:
                        c.drawString(70, y_position, "- Edge fora do range: ajustar nitidez")
            
            c.save()
            os.unlink(temp_orig_path)
            os.unlink(temp_proc_path)
            
            pdf_buffer.seek(0)
            return pdf_buffer
            
        except Exception as e:
            st.error(f"❌ Erro ao gerar PDF: {str(e)}")
            return None

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal"""
    
    sistema = ImageProcessingSystem()
    
    st.title("🏙️ Sistema de Processamento de Imagens v4.0 - Projeto Véridia")
    st.markdown("### Análise e realce avançado")
    
    with st.sidebar:
        st.header("👤 Usuário")
        st.session_state.user = st.text_input("Nome", value=st.session_state.user)
        user_role = st.selectbox("Nível", ["Operador", "Administrador"])
        
        st.divider()
        
        st.header("📊 Sistema")
        st.info(f"""
        **Versão:** 4.0
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
        
        if st.session_state.contexto_aplicado:
            st.divider()
            contexto = CONTEXTOS_VERIDIIA[st.session_state.contexto_aplicado]
            st.success(f"""
            **Contexto Ativo:**  
            {contexto['icone']} {st.session_state.contexto_aplicado}
            """)
    
    # Criar tabs - NOVA TAB VÉRIDIA COMO PRIMEIRA
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏙️ Véridia", "📤 Upload", "🔧 Processamento", "📊 Análise", 
        "📈 Métricas", "📄 Relatório", "⚡ Híbrido"
    ])
    
    # ========================================================================
    # TAB 0: PROJETO VÉRIDIA - CONTEXTOS
    # ========================================================================
    with tab0:
        st.header("🏙️ Projeto Véridia - Contextos Especializados")
        
        st.markdown("""
        <div style='background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px;'>
        <h3 style='color: #2C3E50;'>Bem-vindo!</h3>
        <p style='color: #555;'>
        Este sistema foi desenvolvido para atender às necessidades específicas da cidade de Véridia 
        nos setores de <strong>Educação</strong>, <strong>Saúde</strong> e <strong>Indústria</strong>.
        </p>
        <p style='color: #555;'>
        Cada contexto possui parâmetros otimizados para seu caso de uso específico, garantindo 
        resultados profissionais com um único clique.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.normalized_image is None:
            st.warning("⚠️ Carregue uma imagem na aba 'Upload' primeiro para aplicar os presets contextuais!")
        else:
            st.success("✅ Imagem carregada! Escolha um contexto abaixo:")
        
        st.divider()
        
        # Criar um card para cada contexto
        for contexto_nome, contexto_info in CONTEXTOS_VERIDIIA.items():
            with st.expander(f"{contexto_info['icone']} **{contexto_nome}**", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Descrição:**  \n{contexto_info['descricao']}")
                    
                    st.markdown("**Casos de Uso:**")
                    for caso in contexto_info['casos_uso']:
                        st.markdown(f"• {caso}")
                    
                    st.markdown(f"**Justificativa Técnica:**  \n*{contexto_info['justificativa']}*")
                
                with col2:
                    st.markdown("**Parâmetros:**")
                    params = contexto_info['parametros']
                    
                    if params['use_smoothing']:
                        st.write(f"✓ Suavização (σ={params['sigma']})")
                    else:
                        st.write("✗ Suavização")
                    
                    if params['use_clahe']:
                        st.write(f"✓ CLAHE (clip={params['clip_limit']})")
                    else:
                        st.write("✗ CLAHE")
                    
                    if params['use_sharpening']:
                        st.write(f"✓ Nitidez ({params['sharp_method']})")
                        st.write(f"  • Peso: {params['weight']}")
                    else:
                        st.write("✗ Nitidez")
                    
                    st.divider()
                    
                    if st.button(
                        f"🚀 Aplicar {contexto_nome}", 
                        type="primary",
                        key=f"btn_{contexto_nome}",
                        use_container_width=True,
                        disabled=(st.session_state.normalized_image is None)
                    ):
                        if ImageProcessingSystem.aplicar_preset_contextual(contexto_nome):
                            st.rerun()
        
        st.divider()
        
        # Informações adicionais
        st.markdown("""
        <div style='background: #F8F9F9; border-left: 4px solid #3498DB; padding: 15px; margin-top: 20px;'>
        <h4 style='color: #2C3E50; margin-top: 0;'>💡 Como usar os contextos:</h4>
        <ol style='color: #555;'>
        <li>Carregue uma imagem na aba <strong>Upload</strong></li>
        <li>Escolha o contexto apropriado acima</li>
        <li>Clique em <strong>Aplicar</strong> para processar automaticamente</li>
        <li>Vá para a aba <strong>Métricas</strong> para avaliar os resultados</li>
        <li>Se necessário, ajuste manualmente na aba <strong>Híbrido</strong></li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 1: UPLOAD
    # ========================================================================
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
    
    # ========================================================================
    # TAB 2: PROCESSAMENTO
    # ========================================================================
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
                        ["CLAHE (Local)", "Global"],
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
                        if st.button("👁️ Preview", key="prev_contr", use_container_width=True):
                            ImageProcessingSystem.apply_contrast(contrast_method, clip_limit, tile_size)
                    with col2:
                        if st.button("✅ Aplicar", key="app_contr", use_container_width=True):
                            if ImageProcessingSystem.apply_contrast(contrast_method, clip_limit, tile_size):
                                ImageProcessingSystem.confirm_preview()
            
            with col_preview:
                st.subheader("📺 Visualização")
                
                if st.session_state.preview_image is not None:
                    st.image(st.session_state.preview_image, caption="Preview", use_container_width=True)
                elif st.session_state.processed_image is not None:
                    st.image(st.session_state.processed_image, caption="Atual", use_container_width=True)
                else:
                    st.info("👆 Aplique um filtro acima")
    
    # ========================================================================
    # TAB 3: ANÁLISE
    # ========================================================================
    with tab3:
        if st.session_state.normalized_image is None or st.session_state.processed_image is None:
            st.warning("⚠️ Carregue e processe uma imagem primeiro")
        else:
            st.header("📊 Análise Comparativa")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original")
                st.image(st.session_state.normalized_image, use_container_width=True)
            
            with col2:
                st.subheader("Processada")
                st.image(st.session_state.processed_image, use_container_width=True)
            
            st.divider()
            st.subheader("📈 Histogramas")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            for i, (img, title) in enumerate([(st.session_state.normalized_image, "Original"), 
                                              (st.session_state.processed_image, "Processada")]):
                for c, color in enumerate(['red', 'green', 'blue']):
                    axes[i, 0].hist(img[:,:,c].ravel(), bins=256, color=color, alpha=0.5)
                axes[i, 0].set_title(f"{title} - RGB")
                axes[i, 0].set_xlim([0, 256])
                
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                axes[i, 1].hist(gray.ravel(), bins=256, color='gray')
                axes[i, 1].set_title(f"{title} - Grayscale")
                axes[i, 1].set_xlim([0, 256])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # ========================================================================
    # TAB 4: MÉTRICAS
    # ========================================================================
    with tab4:
        st.header("📈 Métricas de Qualidade")
        
        if st.button("📊 Calcular Métricas", type="primary"):
            ImageProcessingSystem.calculate_metrics()
        
        if st.session_state.metrics:
            m = st.session_state.metrics
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "PSNR",
                    f"{m['PSNR']:.2f} dB",
                    delta="OK" if m['psnr_ok'] else "Baixo",
                    delta_color="normal" if m['psnr_ok'] else "inverse"
                )
            
            with col2:
                st.metric(
                    "SSIM",
                    f"{m['SSIM']:.3f}",
                    delta="OK" if m['ssim_ok'] else "Baixo",
                    delta_color="normal" if m['ssim_ok'] else "inverse"
                )
            
            with col3:
                st.metric(
                    "Local Contrast",
                    f"{m['LC']:.3f}",
                    delta="OK" if m['lc_ok'] else "Baixo",
                    delta_color="normal" if m['lc_ok'] else "inverse"
                )
            
            with col4:
                st.metric(
                    "Edge Sharpness",
                    f"{m['Edge_Sharpness']:.3f}",
                    delta="OK" if m['edge_ok'] else "Fora",
                    delta_color="normal" if m['edge_ok'] else "inverse"
                )
            
            st.divider()
            
            all_ok = m['psnr_ok'] and m['ssim_ok'] and m['lc_ok'] and m['edge_ok']
            
            if all_ok:
                st.success("✅ **APROVADO** - Todas as métricas dentro dos parâmetros!")
            else:
                st.warning("⚠️ **ATENÇÃO** - Algumas métricas fora dos parâmetros")
        else:
            st.info("👆 Clique no botão acima para calcular métricas")
    
    # ========================================================================
    # TAB 5: RELATÓRIO
    # ========================================================================
    with tab5:
        st.header("📄 Relatório")
        
        if st.session_state.processed_image is None:
            st.warning("⚠️ Processe uma imagem primeiro")
        else:
            st.subheader("📋 Histórico de Operações")
            
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
                    if st.session_state.contexto_aplicado:
                        st.write(f"• Contexto: {st.session_state.contexto_aplicado}")
                
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
                    st.session_state.contexto_aplicado = None
                    st.success("✅ Resetado!")
                    st.rerun()
            
            with col2:
                if st.button("📋 Limpar Histórico", use_container_width=True):
                    st.session_state.history = []
                    st.success("✅ Limpo!")
                    st.rerun()
    
    # ========================================================================
    # TAB 6: HÍBRIDO
    # ========================================================================
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
        <p><strong>Sistema de Processamento de Imagens v4.0 - Projeto Véridia</strong></p>
        <p>Python • OpenCV • scikit-image • Streamlit</p>
        <p>🎓 Educação • 🏥 Saúde • 🏭 Indústria</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
