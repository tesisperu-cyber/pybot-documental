# 📚 PYBOT DOCUMENTAL

Aplicación web para **consultar tus documentos con IA**. Sube archivos, busca por significado y haz preguntas: las respuestas citan los fragmentos exactos de donde salió la información.

Hecha con [Streamlit](https://streamlit.io), FAISS y Groq. Todo corre en un solo archivo (`app.py`).

## ✨ Qué hace

- **Sube documentos** en PDF, DOCX, TXT y MD (hasta 5 archivos de 10 MB por sesión).
- **Búsqueda semántica:** encuentra fragmentos por significado, no solo por palabras exactas. Funciona bien en español.
- **Chat con fuentes:** responde usando solo el contenido de tus documentos y cita cada afirmación como [1], [2]... con el archivo y la página.
- **Filtro por documento** para limitar la búsqueda a los archivos que elijas.
- **Descarga la conversación** en formato Markdown.
- **Privacidad:** los archivos se procesan en la sesión del navegador y no se guardan en disco ni en base de datos.

## 🧠 Cómo funciona

1. El texto de cada documento se divide en fragmentos con solapamiento.
2. Cada fragmento se convierte en un vector con un modelo de embeddings multilingüe ligero (`fastembed`, ejecutado en CPU).
3. Los vectores se guardan en un índice **FAISS** en memoria.
4. Al preguntar, se recuperan los fragmentos más parecidos a la pregunta.
5. Esos fragmentos se envían a un modelo de **Groq**, que redacta la respuesta citando las fuentes.

Sin API Key de Groq la app sigue funcionando como buscador semántico.

## 🔑 API Key de Groq

La app pide **"Ingresa tu API Key de Groq"** en la barra lateral. Puedes crear una gratis en [console.groq.com/keys](https://console.groq.com/keys).

- La clave que pegue cada visitante se usa solo en su sesión y no se almacena.
- Si el dueño de la app configura una clave en *Secrets*, los visitantes sin clave pueden hacer hasta **15 preguntas por sesión** (se cambia en la constante `FREE_QUESTIONS` de `app.py`).

## 🚀 Ejecutar en local

Requisitos: Python 3.10 o superior.

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La primera vez se descarga el modelo de embeddings (unos cientos de MB), por lo que el arranque inicial tarda un poco.

Opcional: para no escribir la clave cada vez, crea el archivo `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "tu_clave"
```

## ☁️ Publicar en Streamlit Community Cloud

1. Sube `app.py`, `requirements.txt` y este `README.md` a un repositorio de GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io), crea una nueva app y elige el repositorio y `app.py` como archivo principal.
3. En **Settings → Secrets** agrega:

   ```toml
   GROQ_API_KEY = "tu_clave"
   ```

4. Despliega.

> ⚠️ Nunca subas `.streamlit/secrets.toml` ni tu clave a GitHub. Agrega `.streamlit/secrets.toml` a tu `.gitignore`.

## ⚙️ Configuración rápida

Estas constantes están al inicio de `app.py`:

| Constante | Qué controla | Valor por defecto |
| --- | --- | --- |
| `EMBED_MODEL` | Modelo de embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| `DEFAULT_LLM` | Modelo de Groq para responder | `openai/gpt-oss-120b` |
| `CHUNK_WORDS` / `CHUNK_OVERLAP` | Tamaño y solapamiento de fragmentos | 180 / 40 palabras |
| `MAX_FILES` / `MAX_FILE_MB` | Límites de carga | 5 archivos / 10 MB |
| `FREE_QUESTIONS` | Preguntas gratis con la clave de la app | 15 |

Si Groq cambia o retira el modelo (como pasó con `llama-3.3-70b-versatile`, retirado el 16-ago-2026), escribe otro en el campo **Modelo de Groq** de la barra lateral, o define `GROQ_MODEL = "nombre/del-modelo"` en *Secrets* sin tocar el código. Lista vigente: [console.groq.com/docs/models](https://console.groq.com/docs/models).

> Si ves `NotFoundError` al preguntar, casi siempre es el nombre del modelo.

## ⚠️ Limitaciones

- Los **PDF escaneados** (imágenes sin texto) no se pueden leer; harían falta OCR.
- El índice vive en memoria: al cerrar o recargar la página se pierde y hay que volver a indexar.
- Streamlit Community Cloud tiene memoria limitada, así que conviene no subir documentos muy grandes.
- Las respuestas dependen del modelo de lenguaje: revisa siempre las fuentes citadas antes de usarlas en trabajos formales.

## 🙌 Créditos

Idea inspirada en el proyecto [AgenteIADocumental](https://github.com/Blandskron/AgenteIADocumental) de Blandskron (licencia MIT). Esta versión es una reescritura independiente y simplificada para Streamlit, sin código copiado del original.

## 📄 Licencia

Agrega aquí la licencia que prefieras para tu repositorio (por ejemplo, MIT) creando un archivo `LICENSE`.
