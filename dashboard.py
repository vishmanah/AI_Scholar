# dashboard.py (modificado)
import streamlit as st
import glob
import csv
import io
import os
from datetime import datetime
import logging
# graphviz not used — avoid extra dependency
import time
from ai_core_v2 import AdvancedAutonomousScholar  # ← CAMBIO AQUÍ

logger_mod = logging.getLogger(__name__)

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
    # Modos y auto-guardado
    st.session_state.strict_mode = False
    st.session_state.auto_save_enabled = False
    st.session_state.auto_save_every = 20
    st.session_state.last_saved_processed = 0
    # Modo simple (UI reducida)
    st.session_state.simple_mode = True
    st.session_state.simple_suggestions = []


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
    # Modo simple primero; si está activo, ocultar controles avanzados
    st.session_state.simple_mode = st.checkbox(
        "Modo simple (seleccionar temas sugeridos)",
        value=st.session_state.simple_mode,
    )
    if st.session_state.simple_mode:
        # Forzar estados: sin auto, estricto on, no corriendo
        st.session_state.auto_mode = False
        st.session_state.is_running = False
        st.session_state.strict_mode = True
        st.info(
            "Modo simple activo: automático desactivado y modo estricto activo"
        )
    else:
        st.session_state.auto_mode = st.checkbox(
            "Aprendizaje automático continuo", value=bool(
                st.session_state.get('auto_mode', False)
            )
        )
        st.session_state.strict_mode = st.checkbox(
            "Modo estricto (solo aprende temas seleccionados)",
            value=bool(st.session_state.get('strict_mode', False)),
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
    progress_every = st.slider(
        "Frecuencia log re-encodado (items)", 10, 1000, 100, step=10
    )
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
            progress_log_every=int(progress_every),
        )
        if ok:
            st.success("Sesión cargada correctamente.")
            try:
                stats_loaded = (
                    st.session_state.scholar_ai.get_stats()
                )
                st.session_state.last_saved_processed = (
                    stats_loaded['processed_topics']
                )
            except (AttributeError, KeyError, TypeError) as e:
                # Log targeted, keep visibility in UI
                logger_mod.warning(
                    "Failed to read 'processed_topics' after load_session: %s",
                    e,
                    exc_info=True,
                )
                st.warning(
                    "No se pudo leer 'processed_topics' tras cargar sesión: "
                    f"{e}"
                )
                st.session_state.last_saved_processed = 0
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
                progress_log_every=int(progress_every),
            )
            if ok2:
                st.success("Sesión cargada desde el listado.")
                try:
                    stats_loaded2 = (
                        st.session_state.scholar_ai.get_stats()
                    )
                    st.session_state.last_saved_processed = (
                        stats_loaded2['processed_topics']
                    )
                except (AttributeError, KeyError, TypeError) as e:
                    logger_mod.warning(
                        "Failed to read 'processed_topics' after list load: "
                        "%s",
                        e,
                        exc_info=True,
                    )
                    st.warning(
                        "No se pudo leer 'processed_topics' tras cargar desde "
                        f"listado: {e}"
                    )
                    st.session_state.last_saved_processed = 0
            else:
                st.error("No se pudo cargar desde el listado.")

    st.markdown("---")
    st.subheader("🔍 Validación de Pesos")
    if (
        st.button("Validar compatibilidad (sin cargar)")
        and st.session_state.scholar_ai
    ):
        jp = load_json.strip() if load_json.strip() else sel_json
        wp = load_weights.strip() if load_weights.strip() else sel_w
        if not jp or not wp:
            st.warning("Especifica JSON y Pesos (o elige desde los listados)")
        else:
            okv, msg = (
                st.session_state.scholar_ai.validate_weights_compatibility(
                    json_path=jp,
                    weights_path=wp,
                )
            )
            if okv:
                st.success(msg)
            else:
                st.error(msg)
    
    if st.button("🚀 Iniciar", key="start"):
        # En manual: no iniciar corriendo automáticamente
        st.session_state.is_running = bool(st.session_state.auto_mode)
        st.session_state.log_messages = [f"Iniciando con '{initial_topic}'..."]
        st.session_state.scholar_ai = AdvancedAutonomousScholar(
            initial_topic=initial_topic,
            logger_callback=logger,
        )
        st.session_state.scholar_ai.brain.novelty_threshold = novelty_threshold
    # En modo simple: forzar aprendizaje solo por selección
    # y desactivar auto
        if st.session_state.simple_mode:
            st.session_state.auto_mode = False
            st.session_state.is_running = False
            st.session_state.strict_mode = True
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

    # UI simplificada: seleccionar temas sugeridos y aprender
    if st.session_state.simple_mode:
        st.header("🎯 Selecciona temas sugeridos")

        def build_simple_suggestions():
            # Combina prioridades, frontera y (si hay) sugerencias del chat
            prio = [t for _, t in sorted(ai.priority_queue, reverse=True)]
            front = list(ai.learning_frontier)
            chat_sugg = (
                st.session_state.get('chat_suggestions', []) or []
            )
            combined = list(dict.fromkeys(prio + front + chat_sugg))
            # Intento opcional de extraer sugerencias vía Q&A
            try:
                q = f"Sugiere temas relacionados con {initial_topic}"
                ans = ai.answer_question(q, top_k=5)
                if isinstance(ans, dict):
                    extra = ans.get('suggested_next', []) or []
                    combined = list(dict.fromkeys(combined + extra))
            except Exception:
                pass
            # Filtrar procesados y limitar
            combined = [
                t for t in combined if t not in ai.processed_topics
            ]
            return combined[:20]

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            if st.button("🔎 Buscar temas"):
                st.session_state.simple_suggestions = (
                    build_simple_suggestions()
                )
            # Búsqueda en Internet (Wikipedia)
            q_online = st.text_input(
                "Buscar en Internet (Wikipedia)",
                value=initial_topic,
            )
            if st.button("🌐 Buscar en Internet") and q_online.strip():
                try:
                    online = ai.suggest_topics_online(
                        q_online.strip(),
                        limit=20,
                    )
                except Exception as e:
                    online = []
                    logger_mod.warning(
                        "Fallo en sugerencias online para '%s': %s",
                        q_online,
                        e,
                        exc_info=True,
                    )
                # Mezclar con las actuales y deduplicar
                merged = list(
                    dict.fromkeys(online + st.session_state.simple_suggestions)
                )
                st.session_state.simple_suggestions = merged[:20]
        with col_s2:
            st.info(
                "En Modo simple: automático desactivado y "
                "modo estricto activado"
            )

        if not st.session_state.simple_suggestions:
            # Prellenar en primer render si hay datos
            st.session_state.simple_suggestions = build_simple_suggestions()

        sel_simple = st.multiselect(
            "Temas sugeridos",
            st.session_state.simple_suggestions,
            max_selections=10,
        )
        if st.button("Aprender seleccionados (simple)") and sel_simple:
            learned = 0
            for t in sel_simple:
                if ai.learn_topic(t):
                    learned += 1
            st.success(f"Aprendidos {learned} tema(s) en modo simple")
            # Eliminar ya procesados y quizá autoguardar
            st.session_state.simple_suggestions = [
                t for t in st.session_state.simple_suggestions
                if t not in ai.processed_topics
            ]
            # Auto-guardado si corresponde
            try:
                # maybe_autosave se define más abajo en la UI completa;
                # aquí replicamos mínimo
                if st.session_state.auto_save_enabled:
                    stats_now = ai.get_stats()
                    proc_now = stats_now.get('processed_topics', 0)
                    if (
                        proc_now - st.session_state.last_saved_processed
                    ) >= int(st.session_state.auto_save_every):
                        paths = ai.save_session()
                        st.session_state.last_saved_processed = proc_now
                        st.success(
                            "Auto-guardado: "
                            f"{paths['json']} | {paths['weights']}"
                        )
            except Exception:
                pass
            st.rerun()

        # Ocultar el resto de la UI avanzada
        st.stop()

    # Auto-guardado utilitario
    def maybe_autosave(ai_obj):
        """Guardar sesión automáticamente cada N temas procesados."""
        try:
            proc = ai_obj.get_stats()['processed_topics']
        except Exception:
            return
        if (
            st.session_state.auto_save_enabled
            and (proc - st.session_state.last_saved_processed)
                >= int(st.session_state.auto_save_every)
        ):
            paths = ai_obj.save_session()
            st.session_state.last_saved_processed = proc
            st.success(
                f"Auto-guardado: {paths['json']} | {paths['weights']}"
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

    # Controles de Auto-guardado
    st.subheader("💾 Auto-guardado")
    cols_av = st.columns(2)
    with cols_av[0]:
        st.session_state.auto_save_enabled = st.checkbox(
            "Activar auto-guardado",
            value=st.session_state.auto_save_enabled,
        )
    with cols_av[1]:
        st.session_state.auto_save_every = st.slider(
            "Cada N temas",
            5,
            200,
            st.session_state.auto_save_every,
            step=5,
        )

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
            maybe_autosave(ai)
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
            if st.session_state.strict_mode:
                st.warning(
                    "Modo estricto activo: usa selección manual o desde chat."
                )
            else:
                if ai.learn_one_step():
                    maybe_autosave(ai)
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
            maybe_autosave(ai)
            st.rerun()

    st.divider()
    save_col, auto_col = st.columns([1, 1])
    with save_col:
        if st.button("💾 Guardar sesión"):
            paths = ai.save_session()
            st.success(
                f"Checkpoint: {paths['json']} | Pesos: {paths['weights']}"
            )
            try:
                st.session_state.last_saved_processed = (
                    ai.get_stats()['processed_topics']
                )
            except Exception as e:
                logger_mod.exception(
                    "Failed to update last_saved_processed from "
                    "ai.get_stats(): %s",
                    e,
                )
    with auto_col:
        st.write(
            "Modo automático: " + (
                "🟢 ON" if st.session_state.is_running else "⚪ OFF"
            )
        )

    # Exportar memorias a CSV
    st.subheader("📤 Exportar Memorias a CSV")
    if st.button("Exportar CSV"):
        rows = []
        for node in ai.brain.episodic_memory.memories.values():
            rows.append([
                node.topic,
                node.module_id,
                node.access_count,
                node.importance_score,
                node.timestamp,
            ])
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow([
            "topic",
            "module_id",
            "access_count",
            "importance",
            "timestamp",
        ])
        writer.writerows(rows)
        csv_data = csv_buf.getvalue()
        os.makedirs('data/checkpoints', exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f'data/checkpoints/memories_{ts}.csv'
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_data)
        st.success(f"Exportado: {csv_path}")
        st.download_button(
            "Descargar CSV",
            data=csv_data,
            file_name=f"memories_{ts}.csv",
            mime="text/csv",
        )

    if (
        st.session_state.is_running
        and st.session_state.auto_mode
        and not st.session_state.strict_mode
    ):
        keep_running = ai.learn_one_step()
        if not keep_running:
            st.session_state.is_running = False
            st.success("Aprendizaje completado")
        else:
            maybe_autosave(ai)
        time.sleep(0.5)
        st.rerun()
else:
    st.info("Configura e inicia desde la barra lateral")

