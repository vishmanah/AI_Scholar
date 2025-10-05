# main.py
from ai_core_v2 import AdvancedAutonomousScholar
import json
import os
from datetime import datetime


def main():
    print("="*60)
    print("SISTEMA NEUROGENÉTICO AVANZADO")
    print("="*60)
    
    # Configuración
    topic = input("Tema inicial (Enter = 'Inteligencia Artificial'): ").strip()
    if not topic:
        topic = "Inteligencia Artificial"
    
    steps = input("Número de pasos (Enter = 30): ").strip()
    steps = int(steps) if steps else 30
    
    # Inicializar
    scholar = AdvancedAutonomousScholar(initial_topic=topic)
    
    # Aprender
    for step in range(steps):
        if not scholar.learn_one_step():
            print("\nFrontera agotada")
            break
        
        if step % 5 == 0:
            stats = scholar.get_stats()
            print(f"\n--- Paso {step} ---")
            print(f"Módulos: {stats['total_modules']}")
            print(f"Memorias: {stats['memory_stats']['total_memories']}")
            print(f"Procesados: {stats['processed_topics']}")
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    final_stats = scholar.get_stats()
    print(f"Total módulos: {final_stats['total_modules']}")
    print(f"Total memorias: {final_stats['memory_stats']['total_memories']}")
    print(f"Total conceptos: {final_stats['processed_topics']}")
    
    print("\nDistribución por módulo:")
    for mod_id, topics in scholar.module_map.items():
        print(f"\nMódulo {mod_id} ({len(topics)} conceptos):")
        print(f"  {', '.join(topics[:5])}")
    
    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"data/checkpoints/session_{timestamp}.json"
    # Asegurar que la carpeta exista
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'stats': final_stats,
            'module_map': {k: v for k, v in scholar.module_map.items()}
        }, f, indent=2, default=str)
    print(f"\nGuardado en: {save_path}")


if __name__ == "__main__":
    main()
