# dashboard.py (modificado)
import streamlit as st
import glob
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
    st.session_state.auto_mode = False


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
    st.session_state.auto_mode = st.checkbox(
        "Aprendizaje automático continuo", value=False
    )
    st.markdown("---")
    st.subheader("📂 Sesiones")
    load_json = st.text_input(
        "Ruta JSON (checkpoint)",
        "data/checkpoints/session_YYYYMMDD_HHMMSS.json",
    )
    load_weights = st.text_input(
        "Ruta Pesos (.pt)", "data/checkpoints/brain_YYYYMMDD_HHMMSS.pt"
    )
    reencode = st.checkbox("Re-encodar memorias (más lento)", value=True)
    # Descubrir archivos disponibles
    try:
        json_files = sorted(
            glob.glob("data/checkpoints/session_*.json"), reverse=True
        )
    except Exception:
        json_files = []
    try:
        weight_files = sorted(
            glob.glob("data/checkpoints/brain_*.pt"), reverse=True
        )
    except Exception:
        weight_files = []

    if json_files:
        sel_json = st.selectbox("Elegir JSON disponible", json_files)
    else:
        sel_json = ""
    if weight_files:
        sel_w = st.selectbox("Elegir Pesos disponibles", weight_files)
    else:
        sel_w = ""
    if st.button("Cargar sesión") and st.session_state.scholar_ai:
        ok = st.session_state.scholar_ai.load_session(
            json_path=load_json.strip(),
            weights_path=(
                load_weights.strip() if load_weights.strip() else None
            ),
            reencode_memories=bool(reencode),
        )
        if ok:
            st.success("Sesión cargada correctamente.")
        else:
            st.error("No se pudo cargar la sesión. Verifica las rutas.")
    if st.button("Cargar desde listado") and st.session_state.scholar_ai:
        if not sel_json:
            st.error("No hay JSON seleccionado.")
        else:
            warg = sel_w if sel_w else None
            ok2 = st.session_state.scholar_ai.load_session(
                json_path=sel_json,
                weights_path=warg,
                reencode_memories=bool(reencode),
            )
            if ok2:
                st.success("Sesión cargada desde el listado.")
            else:
                st.error("No se pudo cargar desde el listado.")
    
    if st.button("🚀 Iniciar", key="start"):
        # En manual: no iniciar corriendo automáticamente
        st.session_state.is_running = bool(st.session_state.auto_mode)
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

    st.header("🧭 Control Manual de Aprendizaje")
    frontier_list = list(ai.learning_frontier)
    priority_list = [t for _, t in sorted(ai.priority_queue, reverse=True)]

    colm1, colm2 = st.columns(2)
    with colm1:
        st.subheader("Frontera actual")
        selected_frontier = st.multiselect(
            "Selecciona temas a aprender (frontera)",
            frontier_list,
            max_selections=10,
        )
    with colm2:
        st.subheader("Prioridades")
        selected_priority = st.multiselect(
            "Selecciona temas a aprender (prioridad)",
            priority_list,
            max_selections=10,
        )

    cols_actions = st.columns(3)
    with cols_actions[0]:
        if st.button("Aprender seleccionados"):
            to_learn = list(
                dict.fromkeys(selected_frontier + selected_priority)
            )
            learned = 0
            for t in to_learn[:20]:
                if ai.learn_topic(t):
                    learned += 1
            st.success(f"Aprendidos {learned} tema(s)")
            st.rerun()
    with cols_actions[1]:
        new_topic = st.text_input(
            "Agregar tema a la frontera",
            "Aprendizaje automático",
        )
        if st.button("Agregar") and new_topic.strip():
            ai.add_to_frontier(new_topic.strip())
            st.info(f"Agregado '{new_topic.strip()}' a la frontera")
            st.rerun()
    with cols_actions[2]:
        if st.button("Aprender 1 paso automático"):
            if ai.learn_one_step():
                st.rerun()
            else:
                st.success("Frontera agotada")

    st.header("💬 Chat con el Sistema")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_suggestions' not in st.session_state:
        st.session_state.chat_suggestions = []

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
        # Guardar sugerencias de temas a aprender desde el chat
        sugg = (
            result.get('suggested_next', [])
            if isinstance(result, dict)
            else []
        )
        # Agregar primeros de frontera y prioridades como opciones extra
        extra_frontier = list(ai.learning_frontier)[:5]
        extra_priority = [t for _, t in sorted(
            ai.priority_queue, reverse=True
        )][:5]
        all_opts = list(dict.fromkeys(sugg + extra_frontier + extra_priority))
        # Limitar a 10 sugerencias
        all_opts = all_opts[:10]
        st.session_state.chat_suggestions = [
            t for t in all_opts if t not in ai.processed_topics
        ]

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

    # Selección de aprendizaje desde el chat
    if st.session_state.chat_suggestions:
        st.subheader("🔎 Sugerencias para aprender (desde chat)")
        sel_chat = st.multiselect(
            "Selecciona temas a aprender",
            st.session_state.chat_suggestions,
            max_selections=10,
        )
        if st.button("Aprender seleccionados (chat)") and sel_chat:
            learned = 0
            for t in sel_chat:
                if ai.learn_topic(t):
                    learned += 1
            st.session_state.chat_history.append({
                'role': 'assistant',
                'text': f"Aprendidos {learned} tema(s) desde chat."
            })
            # Refrescar sugerencias eliminando los ya procesados
            st.session_state.chat_suggestions = [
                t for t in st.session_state.chat_suggestions
                if t not in ai.processed_topics
            ]
            st.rerun()

    st.divider()
    save_col, auto_col = st.columns([1, 1])
    with save_col:
        if st.button("💾 Guardar sesión"):
            paths = ai.save_session()
            st.success(
                f"Checkpoint: {paths['json']} | Pesos: {paths['weights']}"
            )
    with auto_col:
        st.write(
            "Modo automático: " + (
                "🟢 ON" if st.session_state.is_running else "⚪ OFF"
            )
        )

    if st.session_state.is_running and st.session_state.auto_mode:
        keep_running = ai.learn_one_step()
        if not keep_running:
            st.session_state.is_running = False
            st.success("Aprendizaje completado")
        time.sleep(0.5)
        st.rerun()
else:
    st.info("Configura e inicia desde la barra lateral")

