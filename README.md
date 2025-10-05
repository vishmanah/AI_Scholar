# AI_Scholar

Proyecto: Erudito Autónomo — sistema neurogenético avanzado para extracción y consolidación de conocimiento (prototipo).

Contenido principal:
- `ai_core_v2.py`: implementación principal avanzada (recomendada).
- `ai_core.py`: versión previa / más simple (mantenida para referencia).
- `dashboard.py`: interfaz Streamlit para controlar y visualizar el erudito.
- `main.py`: CLI simple para ejecutar pasos de aprendizaje y guardar checkpoints.

Requisitos:
- Python 3.10+
- Crear un entorno virtual e instalar `pip install -r requirements.txt`.

Uso rápido:
- Ejecutar CLI:
```powershell
python .\main.py
```
- Ejecutar dashboard (Streamlit):
```powershell
streamlit run .\dashboard.py
```

Notas:
- `ai_core_v2.py` es la implementación recomendada: más robusta, con atención y memoria episódica.
- El extractor usa `sentence-transformers` y descarga modelos la primera vez.
- Se crea automáticamente `data/checkpoints/` al guardar desde `main.py`.

Documentación:
- **[`ANALISIS_PROYECTO.md`](ANALISIS_PROYECTO.md)**: Análisis completo y detallado del proyecto (arquitectura, componentes, algoritmos, casos de uso, roadmap y más).

Contribuir:
- Abrir issues o PRs con mejoras. Añadir tests y CI antes de cambios mayores.
