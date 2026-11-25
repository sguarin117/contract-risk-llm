# 📄 Análisis de riesgos en contratos financieros con LLMs

Código de mi Trabajo de Fin de Máster (TFM) del **Máster en Tecnologías del Sector Financiero: Fintech** de la **Universidad Carlos III de Madrid**.

La herramienta prepara contratos financieros en PDF para que un modelo de lenguaje grande (LLM) los lea y detecte riesgos. Cada riesgo detectado debe citar la sección, el ítem y la página del contrato que lo respalda, para que el resultado se pueda auditar.

## Contexto y problema

Revisar contratos financieros a mano (derivados, préstamos, acuerdos de colateral…) es lento y depende de la experiencia de quien los lee. Los LLMs pueden ayudar, pero si reciben el texto de un PDF mal extraído, o si responden en texto libre, sus conclusiones son difíciles de verificar.

El proyecto aborda ambos problemas con un pipeline de dos pasos:

1. **Extracción fiel del PDF** a un JSON intermedio que conserva la estructura legal del contrato y la procedencia de cada línea.
2. **Construcción de un paquete de prompt** con una taxonomía de riesgos y un esquema de salida JSON estricto (*Structured Outputs*), que obliga al modelo a justificar cada riesgo con evidencias trazables.

## Pipeline

```
contrato.pdf
   │  extract_contract_pdf_V3.py
   ▼
contrato.json (JSON intermedio) + contrato_context.xml (contexto para el LLM)
   │  build_contract_risk_prompt_V3.py
   ▼
contrato_prompt.json (paquete de prompt listo para enviar a una API de LLM)
```

### 1. `extract_contract_pdf_V3.py`: extracción del contrato

- Extracción del texto **respetando el layout** con PyMuPDF: conserva la geometría de cada línea y su procedencia (página y posición).
- Reconstruye cada línea a partir de los espacios horizontales entre fragmentos de texto, en lugar de insertar espacios a ciegas.
- **Detecta encabezados legales** (cláusulas, artículos, anexos, numeraciones como `1.2.3`) sin confundirlos con viñetas en minúscula.
- Elimina los números de página solo cuando están en los márgenes, para no borrar contenido del cuerpo que parezca un número.
- Mantiene el orden original del contenido dentro de cada sección.
- **OCR opcional** (Tesseract) para páginas escaneadas con poca capa de texto, y soporte para PDFs cifrados.
- Salidas: un JSON intermedio reproducible y un contexto con etiquetas tipo XML pensado para LLMs.

### 2. `build_contract_risk_prompt_V3.py`: paquete de prompt

- Define una **taxonomía de 14 categorías de riesgo financiero**: mercado, crédito de contraparte, liquidez, liquidación, operativo, legal/documental, cancelación anticipada, margen/colateral, valoración, idoneidad, fiscal/regulatorio, entre otros.
- Genera un **esquema JSON estricto** para la respuesta del modelo, compatible con *Structured Outputs* (Responses API y Chat Completions, también en modelos fine-tuned).
- Exige que cada riesgo incluya **evidencias trazables** a sección, ítem y página del contrato.
- No llama al LLM: produce un paquete auditable que puede enviarse a cualquier API compatible.

## Uso

Requiere Python 3.10+.

```bash
pip install -r requirements.txt

# 1. Extraer el contrato a JSON intermedio y contexto para el LLM
python extract_contract_pdf_V3.py contrato.pdf --out_json contrato.json --out_context contrato_context.xml

#    Opcional: OCR para PDFs escaneados (requiere Tesseract instalado)
python extract_contract_pdf_V3.py contrato.pdf --use_ocr_when_needed --ocr_language eng+spa

# 2. Construir el paquete de prompt para el análisis de riesgos
python build_contract_risk_prompt_V3.py contrato.json --out_json contrato_prompt.json
```

Si se ejecutan sin argumentos, ambos scripts piden las rutas de forma interactiva. El segundo busca por defecto los JSON de entrada en la carpeta `JSON_Docs/`, junto al script.

## Herramientas

Python · PyMuPDF · Tesseract OCR · JSON Schema / Structured Outputs · Prompt engineering

## Autor

**Sebastián Camilo Guarín Novoa**: [LinkedIn](https://www.linkedin.com/in/sebastian-guarin) · [GitHub](https://github.com/sguarin117)
