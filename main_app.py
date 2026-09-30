"""
LLM Playground — Streamlit + Groq + Hugging Face
Autor: Jean Kennan (Altai Digital)

Funcionalidades:
  1. Ingreso de API keys (Groq / Hugging Face) desde la barra lateral.
  2. Tokenización con distintos modelos: token IDs + color por token.
  3. Generación de texto con modelos GPT vía Groq (parámetros ajustables).
  4. Distintos modelos de EMBEDDING.
  5. Similitud de coseno entre frases.
"""
import os
import hashlib
import numpy as np
import streamlit as st

st.set_page_config(page_title="LLM Playground", page_icon="🧠", layout="wide")

# ------------------------------------------------------------------
# Catálogos de modelos
# ------------------------------------------------------------------
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "gemma2-9b-it",
]

TOKENIZER_MODELS = {
    "GPT-2 (BPE)": "gpt2",
    "BERT (WordPiece)": "bert-base-uncased",
    "RoBERTa (BPE)": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
}

EMBEDDING_MODELS = {
    "MiniLM (384d, rápido)": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet (768d, preciso)": "sentence-transformers/all-mpnet-base-v2",
    "Multilingual MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


# ------------------------------------------------------------------
# Utilidades cacheadas (se cargan una sola vez por modelo)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando tokenizador…")
def load_tokenizer(model_name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


@st.cache_resource(show_spinner="Cargando modelo de embeddings…")
def load_embedder(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def color_for_token(token_id):
    """Color estable y legible por token, derivado de su ID (hash -> HSL)."""
    h = int(hashlib.md5(str(token_id).encode()).hexdigest(), 16)
    hue = h % 360
    return f"hsl({hue}, 70%, 82%)"


def render_colored_tokens(tokens, ids):
    """HTML con cada token en su color + su ID debajo."""
    piezas = []
    for tok, tid in zip(tokens, ids):
        limpio = tok.replace("Ġ", "␣").replace("##", "").replace("▁", "␣")
        limpio = (limpio.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")) or "∅"
        color = color_for_token(tid)
        piezas.append(
            f'<span style="background:{color};border-radius:6px;padding:3px 6px;'
            f'margin:2px;display:inline-block;font-family:monospace;">'
            f'{limpio}<br><small style="color:#444;">{tid}</small></span>'
        )
    return '<div style="line-height:2.6;">' + "".join(piezas) + "</div>"


def cosine_similarity(a, b):
    """Similitud de coseno entre dos vectores."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------------
# Barra lateral — API keys
# ------------------------------------------------------------------
st.sidebar.title("🔑 Configuración")
groq_key = st.sidebar.text_input("Groq API Key", type="password",
                                 value=os.environ.get("GROQ_API_KEY", ""))
hf_key = st.sidebar.text_input("HuggingFace Token (opcional)", type="password",
                               value=os.environ.get("HF_TOKEN", ""))
if hf_key:
    os.environ["HF_TOKEN"] = hf_key
st.sidebar.markdown("---")
st.sidebar.caption("Groq → generación GPT · HF → tokenizadores y embeddings")

st.title("🧠 LLM Playground")

tab_tok, tab_gen, tab_emb = st.tabs(
    ["🎨 Tokenización", "💬 Generación (Groq)", "📐 Embeddings & Similitud"]
)

# ------------------------------------------------------------------
# TAB 1 — Tokenización con color
# ------------------------------------------------------------------
with tab_tok:
    st.subheader("Tokenización: token IDs + color por token")
    nombre_tok = st.selectbox("Modelo de tokenización", list(TOKENIZER_MODELS))
    texto = st.text_area("Texto a tokenizar", "Hola mundo, esto es Inteligencia Artificial.")
    if st.button("Tokenizar", key="btok"):
        tok = load_tokenizer(TOKENIZER_MODELS[nombre_tok])
        ids = tok.encode(texto, add_special_tokens=False)
        tokens = tok.convert_ids_to_tokens(ids)
        c1, c2 = st.columns(2)
        c1.metric("Nº de tokens", len(ids))
        c2.metric("Caracteres", len(texto))
        st.markdown(render_colored_tokens(tokens, ids), unsafe_allow_html=True)
        st.caption("␣ = espacio previo · cada color = un token distinto")

# ------------------------------------------------------------------
# TAB 2 — Generación con Groq (parámetros)
# ------------------------------------------------------------------
with tab_gen:
    st.subheader("Generación de texto con GPT (Groq)")
    modelo = st.selectbox("Modelo Groq", GROQ_MODELS)
    col1, col2, col3 = st.columns(3)
    temperature = col1.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = col2.slider("Max tokens", 16, 2048, 512, 16)
    top_p = col3.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
    system_prompt = st.text_input("System prompt", "Eres un asistente útil y conciso.")
    prompt = st.text_area("Prompt", "Explica qué es el aprendizaje por refuerzo en 3 frases.")

    if st.button("Generar", key="bgen"):
        if not groq_key:
            st.error("Ingresa tu Groq API Key en la barra lateral.")
        else:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                resp = client.chat.completions.create(
                    model=modelo,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                st.markdown("**Respuesta:**")
                st.write(resp.choices[0].message.content)
                if resp.usage:
                    st.caption(f"Tokens — prompt: {resp.usage.prompt_tokens} · "
                               f"respuesta: {resp.usage.completion_tokens}")
            except Exception as e:
                st.error(f"Error llamando a Groq: {e}")

# ------------------------------------------------------------------
# TAB 3 — Embeddings + similitud de coseno
# ------------------------------------------------------------------
with tab_emb:
    st.subheader("Similitud de coseno entre frases")
    nombre_emb = st.selectbox("Modelo de embedding", list(EMBEDDING_MODELS))
    frase1 = st.text_input("Frase 1", "El gato duerme en el sofá.")
    frase2 = st.text_input("Frase 2", "Un felino descansa en el sillón.")
    if st.button("Comparar", key="bemb"):
        embedder = load_embedder(EMBEDDING_MODELS[nombre_emb])
        v1, v2 = embedder.encode([frase1, frase2])
        sim = cosine_similarity(v1, v2)
        st.metric("Similitud de coseno", f"{sim:.4f}")
        st.progress(max(0.0, min(1.0, (sim + 1) / 2)))
        st.caption(f"Dimensión del embedding: {len(v1)}  ·  1.0 = idénticas, 0 = sin relación")
