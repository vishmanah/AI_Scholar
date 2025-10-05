import os
import sys
import json
from datetime import datetime

import torch

# Ensure project root on sys.path for pytest
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_scholar_smoke(monkeypatch):
    """Simple smoke test: patch the extractor to avoid network calls and
    verify one learning step completes and stats are returned."""

    # Patch the knowledge extractor to avoid external calls
    def fake_get_knowledge_package(self, topic, depth=1):
        emb = torch.randn(384)
        links = ["Test_Link_1", "Test_Link_2"]
        metadata = {"num_sentences": 3, "num_links": 2}
        return emb, links, metadata

    monkeypatch.setattr(
        'ai_core_v2.AdvancedKnowledgeExtractor.get_knowledge_package',
        fake_get_knowledge_package,
    )

    # Import locally after ensuring PROJECT_ROOT is on sys.path
    from ai_core_v2 import AdvancedAutonomousScholar

    scholar = AdvancedAutonomousScholar(initial_topic="Prueba")
    ok = scholar.learn_one_step()
    assert ok is True
    stats = scholar.get_stats()
    assert 'total_modules' in stats
    assert stats['processed_topics'] >= 1


def main_demo(runs: int = 5):
    """Runnable demo (not run by pytest): initializes scholar and runs a few
    learning steps, then saves stats to data/checkpoints/session_*.json."""
    try:
        from ai_core_v2 import AdvancedAutonomousScholar

        scholar = AdvancedAutonomousScholar(
            initial_topic="Inteligencia Artificial"
        )
        for _ in range(runs):
            scholar.learn_one_step()

        stats = scholar.get_stats()

        # Ensure checkpoints dir
        save_dir = os.path.join(PROJECT_ROOT, 'data', 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'session_{timestamp}.json')

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(
                {'stats': stats},
                f,
                default=str,
                indent=2,
                ensure_ascii=False,
            )

        print('Saved demo session to:', save_path)
    except Exception as e:
        print('Demo failed:', e)


if __name__ == '__main__':
    main_demo()
