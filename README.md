# Realce no Domínio Espacial — Grupo 6
https://colab.research.google.com/drive/1HJSyPczEs1jm-sBRuiffuSWQAYV_8_Ku?usp=sharing

## Descrição do Projeto

A cidade de Véridia deseja criar um sistema para analisar, transformar e realçar imagens digitais em diferentes contextos, como educação, saúde e indústria. A prefeitura contratou vocês para desenvolver um Sistema de Processamento de Imagens, permitindo que operadores e administradores processem imagens, apliquem filtros, transformações e análises de padrões, mantendo registro de resultados e relatórios de processamento.

O sistema deve ser implementado em Python, utilizando bibliotecas como:
- numpy
- cv2
- PIL
- skimage
- matplotlib

### Objetivos do sistema:

- Centralizar o processamento de imagens em uma única plataforma;
- Facilitar a aplicação de filtros e transformações de forma controlada;
- Garantir consistência e integridade dos resultados;
- Permitir análise detalhada de padrões e componentes de imagens;
- Incentivar pesquisa e aplicação de novas técnicas em processamento de imagens.

Nesta etapa, os grupos deverão implementar semanalmente as etapas de um sistema modular e inteligente capaz de analisar, transformar, comparar e gerar relatórios sobre imagens digitais.

O sistema deve ser desenvolvido em Python, utilizando bibliotecas como:
- numpy
- opencv (cv2)
- PIL
- matplotlib
- scikit-image
- scikit-learn
- reportlab

Opcionalmente:
- tensorflow
- streamlit

Cada grupo terá tarefas específicas e complementares, com atividades principais múltiplas e desafios avançados.

---

## Grupo 6 - Realce no Domínio Espacial

1. Implementar o que foi proposto na Unidade I.
2. Aplicar técnicas de realce de nitidez no domínio espacial.
3. Implementar equalização local de contraste (CLAHE).
4. Comparar resultados de realce global e local, e medir nitidez e variação de intensidade antes e depois do realce.
5. Criar visualização comparativa entre diferentes parâmetros de realce.
6. Desenvolver função híbrida combinando suavização e realce local.
7. Documentar.
8. Elaborar um artigo científico demonstrando todo o processo realizado.

---

## Plano de Implementação - Entregas

### SEMANAS (8,0) — OBJETIVOS

#### Semana 01 (14/10) — Estrutura e planejamento do módulo (1,0)

- Criar a base do projeto em Python no Google Colab ou ambiente local.
- Definir as funções principais e o fluxo de execução do módulo.
- Iniciar a configuração do repositório no GitHub e inserir o README inicial.

#### Semana 02 (28/10) — Implementação das funcionalidades principais (1,5)

- Desenvolver as funções centrais definidas na Unidade I.
- Testar o funcionamento com diferentes imagens.
- Registrar resultados iniciais e atualizar o repositório.

#### Semana 03 (04/11) — Aprimoramento e análise dos resultados (2,0)

- Realizar novos testes com outras imagens.
- Corrigir falhas de execução e aprimorar os resultados visuais.
- Inserir prints e tabelas de comparação no repositório.

#### Semana 04 (11/11) — Documentação, análise dos resultados e vídeo de demonstração (2,0)

- Elaborar a documentação parcial em formato .pdf, apresentando objetivos, metodologia, imagens usadas e resultados obtidos.
- Adicionar descrição técnica no README.md e no arquivo final.
- Produzir um vídeo curto (máx. 5 min) demonstrando o funcionamento do módulo e os resultados alcançados.
- Publicar o vídeo na pasta `/demo` e finalizar o repositório com commits organizados.

#### Semana 05 (18/11) — Revisão e entrega final (1,5)

- Revisar todo o código, limpar comentários, ajustar nomes de arquivos e garantir a execução correta do projeto.
- Entregar o link final do repositório.

---

## ENTREGA NO CLASSROOM (Google Sala de Aula)

- Cada grupo será avaliado semanalmente.
- Todos os integrantes devem ser adicionados como colaboradores do repositório.
- O link do repositório deverá ser postado no Google Classroom, conforme as datas de entrega.
- Alterações e commits serão utilizados como parte da avaliação de participação individual.
- As versões entregues devem estar funcionais, documentadas e acompanhadas do vídeo demonstrativo.
- Cada grupo deverá criar um repositório público no GitHub com o nome:

Processamento de Imagens_E01_GrupoX

*(Substituir “X” pelo número do grupo.)*

---

## Estrutura obrigatória do repositório

O repositório deverá conter, obrigatoriamente, as seguintes pastas e arquivos:

/src → códigos em Python (.ipynb ou .py)
/imagens → conjunto de imagens utilizadas no projeto (.png, .jpg, .jpeg)
/docs → documentação parcial (entregas semanais) e final em formato .PDF
/demo → vídeo curto (máximo de 5 minutos) demonstrando o funcionamento básico do sistema
README.md → arquivo explicativo com:
- Objetivo do módulo desenvolvido
- Bibliotecas utilizadas
- Instruções de execução
- Responsabilidades de cada integrante
- Prints ou exemplos de saída

### Sobre o vídeo

- O vídeo deve estar no formato `.mp4`
- Pode mostrar apenas a execução e os resultados obtidos, **sem necessidade de narração**

---

### 1. Funções Principais do Sistema

1.1. Aquisição de Imagem

Função: importar_imagem(caminho, tamanho=(512,512))
Entrada: Arquivo PNG/JPEG
Saída: Matriz de pixels normalizada
Objetivo: Carregar e padronizar as imagens.

1.2. Pré-processamento

Função: preprocessar_imagem(imagem, tipo_filtro, raio=None, sigma=None)
Entrada: Imagem original
Saída: Imagem filtrada e normalizada
Filtros possíveis: Mediana, Gaussiano
Parâmetros:

tipo_filtro = "mediana" | "gaussiano"

raio: tamanho do kernel

sigma: desvio padrão (0,5 a 2,0)

1.3. Processamento de Nitidez e Bordas

Funções:

aplicar_laplaciano(imagem, mascara, peso)

aplicar_sobel(imagem, limiar)

filtro_alta_frequencia(imagem, intensidade)
Entrada: Imagem pré-processada
Saída: Imagem realçada
Parâmetros:

Máscara 3x3

Peso

Limiar

Intensidade (máx. 1.5×)

1.4. Processamento de Contraste (Local)

Função: aplicar_CLAHE(imagem, bloco, clip_limit)
Entrada: Imagem realçada
Saída: Imagem com contraste aprimorado
Parâmetros:

Tamanho do bloco

Limite de clipagem (2.0 a 3.0)

1.5. Análise Visual

Função: comparar_imagens(original, processada, modo_visual)
Saída: Imagens lado a lado, gráficos ou diferença
Métricas opcionais: Mapas de borda, histograma

1.6. Avaliação Quantitativa

Funções:

calcular_PSNR(orig, proc)

calcular_SSIM(orig, proc)

calcular_LC(orig, proc)

calcular_edge_sharpness(orig, proc)
Saída: Valores numéricos
Critérios:

PSNR ≥ 30 dB

SSIM ≥ 0.85

1.7. Documentação

Função: gerar_relatorio(resultados, graficos, formato="PDF")
Saída: PDF com conclusões, imagens e métricas

### 2. Fluxo de Execução do Módulo

A seguir está a sequência recomendada da execução:

1. importar_imagem() 
      ↓
2. preprocessar_imagem()
      ↓
3. aplicar filtros de nitidez:
      ↳ aplicar_laplaciano() OU aplicar_sobel() OU filtro_alta_frequencia()
      ↓
4. aplicar_CLAHE()
      ↓
5. comparar_imagens()
      ↓
6. calcular métricas:
      ↳ calcular_PSNR(), calcular_SSIM(), calcular_LC(), calcular_edge_sharpness()
      ↓
7. gerar_relatorio()

