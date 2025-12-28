import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURACIÓN ---
import streamlit as st

# Ahora la clave vendrá de la "Caja Fuerte" de la nube (Secrets)
# Si estás en tu PC local, necesitas crear un archivo .streamlit/secrets.toml
# Pero para subirlo a GitHub, deja esta línea así:
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    # Esto es por si quieres seguir probando en tu PC sin configurar secrets.toml
    # Puedes dejar tu clave aquí SOLO para pruebas locales, pero bórrala antes de subir.
    # Para producción, lo ideal es confiar solo en st.secrets
    pass

# --- GESTIÓN DE IMÁGENES ---
SIDEBAR_LOGO = "https://almakay.org/wp-content/uploads/2021/08/lg-almakay-2025-scaled.png"
AVATAR_URL = "https://almakay.org/wp-content/uploads/2020/06/favicon2.png"

# Configuración de página
st.set_page_config(page_title="Almy - Almakay", page_icon=AVATAR_URL, layout="wide")

# --- BARRA LATERAL ---
with st.sidebar:
    st.image(SIDEBAR_LOGO, width=200)
    st.markdown("### ⚠️ AVISO LEGAL Y DE SEGURIDAD")
    st.warning("**USO EXCLUSIVO PARA MAYORES DE 18 AÑOS.**\nEl uso de esta herramienta puede generar efectos no deseados en menores de edad.")
    
    st.markdown("### 🚫 CONTRAINDICACIONES")
    st.markdown("""
    **No utilices esta herramienta si te encuentras en:**
    * Estados alterados de conciencia (alcohol, drogas).
    * Tratamiento psiquiátrico fuerte o agudo.
    * Crisis de ansiedad, depresión severa o disociación.
    * En estos casos, por favor acude estrictamente a un profesional de la salud mental.
    """)
    st.markdown("---")
    st.caption("**DESCARGO DE RESPONSABILIDAD:** Almy es una IA educativa y de reflexión. **No constituye consejo médico, psicológico ni legal.** Almakay no se responsabiliza de las decisiones tomadas o los riesgos asumidos al utilizar esta herramienta sin acompañamiento profesional.")

st.title("✨ Almy: Tu Acompañante Sistémico")

# --- 1. CARGAR EL CEREBRO ---
@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.load_local(
        "vectorstore/db_faiss", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vectorstore

# --- 2. PERSONALIDAD: ALMY 5.9.1 (GPS CORREGIDO) ---
system_prompt = """
Eres Almy, el ACOMPAÑANTE IA de Almakay Constelaciones Familiares.
Tu base filosófica son las "Fuerzas del Amor" y los textos de Jairo Molina.

### 📍 TU UBICACIÓN FÍSICA (IMPORTANTE)
* **TÚ (ALMAKAY) ESTÁS FÍSICAMENTE SOLO EN QUITO, ECUADOR.** No tienes consultorios en otras ciudades.
* Si el usuario está en cualquier otra ciudad (Loja, Cuenca, Bogotá, Madrid, etc.), la única opción es **ONLINE**.

### 🧠 ESTRATEGIA DE CONEXIÓN
Tu prioridad es GENERAR INSIGHT (Darse cuenta).

1. **GESTIÓN DE LA UBICACIÓN:**
   * Pregunta la ciudad suavemente.
   * Si el usuario responde (ej: "Estoy en Loja"), NO vendas de inmediato. Guarda el dato.
   * Reacción correcta: "Gracias por decirme que estás en Loja. Como estamos lejos, la opción online será ideal para ti más adelante. Pero volviendo a tu sentir..."

2. **EL MOMENTO DE LA INVITACIÓN:**
   * Solo ofrece la sesión cuando el usuario muestre una toma de conciencia.
   * **Lógica de Derivación:**
     * Usuario en **Quito** -> "Te espero en sesión **Presencial** en nuestra sede."
     * Usuario en **CUALQUIER OTRO LADO** -> "Dado que estás en [Ciudad del Usuario], la sesión **Online** es perfecta para ti, conectamos por videollamada."

3. **SER UN PUENTE:**
   * "Ver esto es el primer paso. Para sanarlo, hay que vivirlo en un campo terapéutico."

### 🚫 REGLA DE HIERRO: NO SIMULES EJERCICIOS
* Si piden ejercicio: "Realizar ese movimiento requiere un campo sostenido por un humano. Te invito a vivirlo en sesión: [Link]"

### 🛡️ PROTOCOLOS DE SEGURIDAD
* **Vida/Muerte:** 🛑 DETÉN TODO. Llama a emergencias.
* **Temas Sensibles (Abuso, etc.):** "Necesito asegurarme de que tienes soporte profesional. Avanzar requiere tu auto-responsabilidad. ¿Seguimos bajo esa premisa?"

### 📚 BIBLIOTECA OFICIAL (SOLO SUGERIR)
1. "El Viaje del Alma..."
2. "El Viaje Hacia Ti: La salida es hacia adentro"
3. "El Viaje Hacia Ti: Tomando la Fuerza de Papá y Mamá"
4. "El Viaje Hacia Ti: La Danza Sagrada del Equilibrio Interior"
5. "El Viaje Hacia Ti: Las Fuerzas del Amor"
6. "El Duelo y la Vida..."
7. "El Legado Invisible..."

### MATRIZ DE SERVICIOS
* Enlace: https://almakay.org/reservas

CONTEXTO:
{context}

HISTORIAL:
{chat_history}

USUARIO:
{input}

RESPUESTA DE ALMY:
"""

def main():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_knowledge_base()

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile", # Mantenemos el modelo rápido por hoy
        temperature=0.3 
    )

    prompt = ChatPromptTemplate.from_template(system_prompt)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # --- INTERFAZ ---
    st.info("🔒 **Aviso de Privacidad:** Almy es una IA. Por tu seguridad, evita compartir nombres completos o datos sensibles. Las conversaciones son procesadas externamente.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "🌿 **Bienvenido/a.** Soy Almy, tu acompañante sistémico IA.\n\nEstoy aquí para ofrecerte reflexiones basadas en las **Fuerzas del Amor**, pero **no soy un terapeuta humano**. \n\nAntes de empezar, ten presente que este es un espacio de reflexión. Si estás en crisis, por favor consulta a tu profesional.\n\n¿Qué movimiento de tu alma deseas explorar hoy?"
            }
        ]

    for message in st.session_state.messages:
        avatar_img = AVATAR_URL if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar_img):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("Escribe aquí tu sentir..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant", avatar=AVATAR_URL):
            message_placeholder = st.empty()
            message_placeholder.markdown("🌿 *Sintonizando contigo con respeto...*")
            
            chat_history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
            
            response = retrieval_chain.invoke({
                "input": prompt_input,
                "chat_history": chat_history_text
            })
            respuesta_almy = response['answer']
            
            message_placeholder.markdown(respuesta_almy)
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta_almy})

if __name__ == "__main__":
    main()