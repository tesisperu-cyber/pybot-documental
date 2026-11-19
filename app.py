"""
PYBOT DOCUMENTAL
Gestión y consulta documental con respuestas que citan sus fuentes (RAG simplificado).

Flujo: subir documentos -> trocear -> embeddings (CPU) -> índice FAISS -> búsqueda
semántica -> respuesta con LLM (Groq) citando fragmentos [n].

Todo vive en la sesión del navegador: no se guarda nada en disco ni en base de datos.
"""

import hashlib
import io
import re

import numpy as np
import streamlit as st

# ----------------------------- Configuración ---------------------------------
APP_NAME = "PYBOT DOCUMENTAL"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384 dim, multilingüe
DEFAULT_LLM = "llama-3.3-70b-versatile"  # cámbialo en la barra lateral si Groq lo renombra
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40
MAX_FILE_MB = 10
MAX_FILES = 5
MAX_CHUNKS = 6000
FREE_QUESTIONS = 15  # preguntas por sesión si se usa la clave del dueño (secrets)

st.set_page_config(page_title=APP_NAME, page_icon="📚", layout="wide")


# ----------------------------- Estado de sesión ------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("chunks", [])      # lista de dicts: doc, page, text
    ss.setdefault("index", None)     # índice FAISS
    ss.setdefault("hashes", {})      # sha1 -> nombre de archivo
    ss.setdefault("messages", [])    # historial del chat
    ss.setdefault("n_questions", 0)


# ----------------------------- Embeddings ------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de embeddings (solo la primera vez)...")
def load_embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL)


def embed(texts):
    import faiss

    model = load_embedder()
    vecs = np.array(list(model.embed(texts)), dtype="float32")
    faiss.normalize_L2(vecs)  # con vectores normalizados, producto interno = coseno
    return vecs


# ----------------------------- Lectura de archivos ---------------------------
def extract_pages(name, data):
    """Devuelve una lista de (página|None, texto)."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    if ext == "docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        return [(None, "\n".join(p.text for p in doc.paragraphs))]
    return [(None, data.decode("utf-8", errors="replace"))]  # txt, md


def chunk_pages(doc_name, pages):
    chunks = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for page, text in pages:
        words = re.sub(r"\s+", " ", text).strip().split(" ")
        if not words or words == [""]:
            continue
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + CHUNK_WORDS])
            if len(piece) > 40:
                chunks.append({"doc": doc_name, "page": page, "text": piece})
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


# ----------------------------- Ingesta ---------------------------------------
def ingest(files):
    import faiss

    ss = st.session_state
    for f in files[:MAX_FILES]:
        data = f.getvalue()
        h = hashlib.sha1(data).hexdigest()
        if h in ss.hashes:
            continue
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            st.warning(f"«{f.name}» supera {MAX_FILE_MB} MB y se omitió.")
            continue
        try:
            chunks = chunk_pages(f.name, extract_pages(f.name, data))
        except Exception as e:  # archivo corrupto o protegido
            st.warning(f"No se pudo leer «{f.name}»: {e}")
            continue
        if not chunks:
            st.warning(f"«{f.name}» no tiene texto extraíble (¿PDF escaneado?).")
            continue
        if len(ss.chunks) + len(chunks) > MAX_CHUNKS:
            st.warning("Se alcanzó el límite de fragmentos de esta sesión.")
            break
        vecs = embed([c["text"] for c in chunks])
        if ss.index is None:
            ss.index = faiss.IndexFlatIP(vecs.shape[1])
        ss.index.add(vecs)
        ss.chunks.extend(chunks)
        ss.hashes[h] = f.name


def reset_all():
    for k in ("chunks", "index", "hashes", "messages", "n_questions"):
        st.session_state.pop(k, None)
    init_state()


# ----------------------------- Búsqueda --------------------------------------
def search(query, k, only_docs):
    ss = st.session_state
    if ss.index is None or not ss.chunks:
        return []
    q = embed([query])
    n = min(len(ss.chunks), max(k * 6, 30))
    scores, ids = ss.index.search(q, n)
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        c = ss.chunks[i]
        if only_docs and c["doc"] not in only_docs:
            continue
        results.append({**c, "score": float(score)})
        if len(results) == k:
            break
    return results


def source_label(r):
    page = f", pág. {r['page']}" if r.get("page") else ""
    return f"{r['doc']}{page}"


def render_sources(results):
    with st.expander(f"📎 Fuentes ({len(results)})"):
        for n, r in enumerate(results, 1):
            st.markdown(f"**[{n}] {source_label(r)}** · similitud {r['score']:.2f}")
            st.caption(r["text"])


# ----------------------------- Respuesta con LLM -----------------------------
SYSTEM_PROMPT = (
    "Eres PYBOT DOCUMENTAL, un asistente que responde SOLO con la información de los "
    "fragmentos de contexto entregados. Responde en el idioma de la pregunta. "
    "Cita las fuentes con el formato [n] junto a cada afirmación. Si el contexto no "
    "contiene la respuesta, di claramente que no aparece en los documentos cargados. "
    "No inventes datos, cifras ni referencias."
)


def answer(question, results, api_key, model):
    from groq import Groq

    context = "\n\n".join(f"[{n}] ({source_label(r)})\n{r['text']}" for n, r in enumerate(results, 1))
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"},
        ],
    )
    return resp.choices[0].message.content


def resolve_key(user_key):
    """Devuelve (clave, es_del_dueño)."""
    if user_key.strip():
        return user_key.strip(), False
    try:
        return st.secrets["GROQ_API_KEY"], True
    except Exception:
        return "", False


# ----------------------------- Interfaz --------------------------------------
def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.title("📚 " + APP_NAME)
        st.caption("Sube tus documentos y consúltalos con respuestas que citan sus fuentes.")

        st.markdown("### 🔑 Ingresa tu API Key de Groq")
        user_key = st.text_input(
            "API Key de Groq",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
            help="Se usa solo en esta sesión y no se guarda. Si la dejas vacía se usa la clave de la app, con un límite de preguntas.",
        )
        st.caption("¿No tienes una? Créala gratis en [console.groq.com/keys](https://console.groq.com/keys).")
        _key, _is_owner = resolve_key(user_key)
        if user_key.strip():
            st.success("Usando tu API Key.", icon="✅")
        elif _key:
            st.info(f"Usando la clave de la app (máx. {FREE_QUESTIONS} preguntas por sesión).", icon="ℹ️")
        else:
            st.warning("Sin API Key: solo búsqueda semántica.", icon="⚠️")

        st.divider()
        files = st.file_uploader(
            f"Documentos (PDF, DOCX, TXT, MD · máx. {MAX_FILES} archivos de {MAX_FILE_MB} MB)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )
        if files and st.button("Indexar documentos", type="primary", use_container_width=True):
            with st.spinner("Procesando e indexando..."):
                ingest(files)

        if ss.hashes:
            st.markdown("**Documentos indexados**")
            for name in ss.hashes.values():
                st.markdown(f"- {name}")
            st.caption(f"{len(ss.chunks)} fragmentos en el índice")
            if st.button("🗑️ Borrar todo", use_container_width=True):
                reset_all()
                st.rerun()

        st.divider()
        k = st.slider("Fragmentos a recuperar", 2, 10, 5)
        only_docs = st.multiselect("Limitar a documentos", sorted(set(ss.hashes.values())))

        st.divider()
        model = st.text_input("Modelo de Groq", value=DEFAULT_LLM)
    return k, only_docs, user_key, model


def tab_search(k, only_docs):
    q = st.text_input("Buscar en tus documentos", placeholder="Ej.: plazo del contrato")
    if q:
        if not st.session_state.chunks:
            st.info("Primero sube e indexa al menos un documento.")
            return
        results = search(q, k, only_docs)
        if not results:
            st.warning("Sin resultados.")
        for n, r in enumerate(results, 1):
            with st.container(border=True):
                st.markdown(f"**[{n}] {source_label(r)}** · similitud {r['score']:.2f}")
                st.write(r["text"])


def tab_chat(k, only_docs, user_key, model):
    ss = st.session_state

    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                render_sources(m["sources"])

    prompt = st.chat_input("Pregunta algo sobre tus documentos...")
    if not prompt:
        if ss.messages:
            md = "\n\n".join(
                f"**{'Tú' if m['role'] == 'user' else APP_NAME}:** {m['content']}" for m in ss.messages
            )
            st.download_button("⬇️ Descargar conversación (.md)", md, "conversacion.md")
        return

    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not ss.chunks:
            reply, sources = "Primero sube e indexa al menos un documento en la barra lateral.", []
        else:
            sources = search(prompt, k, only_docs)
            key, is_owner = resolve_key(user_key)
            if not sources:
                reply = "No encontré fragmentos relevantes en los documentos."
            elif not key:
                reply = "No hay clave de Groq configurada, así que te muestro solo los fragmentos más relevantes."
            elif is_owner and ss.n_questions >= FREE_QUESTIONS:
                reply = (
                    f"Llegaste al límite de {FREE_QUESTIONS} preguntas gratuitas por sesión. "
                    "Pega tu propia clave de Groq en la barra lateral para continuar."
                )
            else:
                try:
                    with st.spinner("Generando respuesta..."):
                        reply = answer(prompt, sources, key, model)
                    ss.n_questions += 1
                except Exception as e:
                    reply = f"No se pudo generar la respuesta ({type(e).__name__}). Revisa la clave y el modelo."
        st.markdown(reply)
        if sources:
            render_sources(sources)
    ss.messages.append({"role": "assistant", "content": reply, "sources": sources})


def main():
    init_state()
    k, only_docs, user_key, model = sidebar()

    st.header(APP_NAME)
    if not user_key.strip() and not resolve_key("")[0]:
        st.warning("🔑 Ingresa tu API Key de Groq en la barra lateral para activar las respuestas con IA.")
    if not st.session_state.hashes:
        st.info("👈 Sube uno o más documentos y pulsa **Indexar documentos** para empezar.")

    t1, t2 = st.tabs(["💬 Preguntar", "🔎 Buscar"])
    with t1:
        tab_chat(k, only_docs, user_key, model)
    with t2:
        tab_search(k, only_docs)

    st.caption(
        "Tus archivos se procesan solo en esta sesión y no se guardan. "
        "Inspirado en el proyecto AgenteIADocumental de Blandskron (MIT)."
    )


main()
