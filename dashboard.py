# dashboard.py (modificado)
import streamlit as st
# graphviz not used — avoid extra dependency
import time
from ai_core_v2 import AdvancedAutonomousScholar  # ← CAMBIO AQUÍ

st.set_page_config(layout="wide", page_title="Dashboard del Erudito Autónomo")

st.title("🤖 Dashboard del Erudito Autónomo Avanzado")
st.markdown("Panel de control neurogenético con memoria episódica")


if 'scholar_ai' not in st.session_state:
    st.session_state.scholar_ai = None
    st.session_state.log_messages = [
        "Sistema listo. Configura e inicia el aprendizaje."
    ]
    st.session_state.is_running = False


def logger(message):
    st.session_state.log_messages.insert(0, message)
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages.pop()

 
with st.sidebar:
    st.header("⚙️ Controles")
    initial_topic = st.text_input(
        "Tema Inicial", "Inteligencia Artificial"
    )
    
    novelty_threshold = st.slider("Umbral de Novedad", 0.1, 0.9, 0.5)
    
    if st.button("🚀 Iniciar", key="start"):
        st.session_state.is_running = True
        st.session_state.log_messages = [f"Iniciando con '{initial_topic}'..."]
        st.session_state.scholar_ai = AdvancedAutonomousScholar(
            initial_topic=initial_topic,
            logger_callback=logger,
        )
        st.session_state.scholar_ai.brain.novelty_threshold = novelty_threshold
        st.rerun()

    if st.button("⏸️ Pausar", key="pause"):
        st.session_state.is_running = False
        st.rerun()

if st.session_state.scholar_ai:
    ai = st.session_state.scholar_ai
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Métricas")
        stats = ai.get_stats()
        st.metric("Módulos", stats['total_modules'])
        st.metric("Memorias", stats['memory_stats']['total_memories'])
        st.metric("Conceptos", stats['processed_topics'])
        st.metric("Parámetros", f"{stats['total_parameters']:,}")
        st.metric("Frontera", stats['frontier_size'])
        
        status = (
            "🟢 Aprendiendo"
            if st.session_state.is_running
            else "⏸️ Pausado"
        )
        st.write(status)

    with col2:
        st.subheader("📝 Log")
        log_text = "\n".join(st.session_state.log_messages[:20])
        st.text_area(
            "Registro (accesible)",
            log_text,
            height=300,
            label_visibility="hidden",
        )

    st.header("🧠 Arquitectura Modular")
    if ai.module_map:
        cols = st.columns(min(len(ai.module_map), 4))
        for i, (mod_id, topics) in enumerate(sorted(ai.module_map.items())):
            with cols[i % 4]:
                st.subheader(f"Módulo {mod_id}")
                st.info(f"{len(topics)} conceptos")
                for topic in topics[:3]:
                    st.write(f"• {topic}")

    st.header("💬 Chat con el Sistema")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input(
            "Escribe tu pregunta:",
            "¿Qué es un agente inteligente?",
        )
        submitted = st.form_submit_button("Enviar")
    if submitted and user_msg.strip():
        result = ai.answer_question(user_msg.strip(), top_k=5)
        st.session_state.chat_history.append({
            'role': 'user',
            'text': user_msg.strip()
        })
        st.session_state.chat_history.append({
            'role': 'assistant',
            'text': result
        })

    # Render chat
    for entry in st.session_state.chat_history[-10:]:
        if entry['role'] == 'user':
            st.markdown(f"**Tú:** {entry['text']}")
        else:
            res = entry['text']
            if isinstance(res, dict):
                st.markdown(f"**Sistema:** {res.get('question','')}")
                items = res.get('items', [])
                for it in items[:5]:
                    st.write(
                        f"• {it['topic']} (Módulo {it['module_id']}, "
                        f"sim {it['similarity']:.2f})"
                    )
                    links = it.get('links', [])
                    if links:
                        st.caption("Enlaces:")
                        st.code("\n".join(links[:3]))
            else:
                st.markdown(f"**Sistema:** {res}")

    if st.session_state.is_running:
        keep_running = ai.learn_one_step()
        if not keep_running:
            st.session_state.is_running = False
            st.success("Aprendizaje completado")
        time.sleep(0.5)
        st.rerun()
else:
    st.info("Configura e inicia desde la barra lateral")

