import torch

from ai_core_v2 import AdvancedAutonomousScholar


def test_scholar_smoke(monkeypatch):
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

    scholar = AdvancedAutonomousScholar(initial_topic="Prueba")
    ok = scholar.learn_one_step()
    assert ok is True
    stats = scholar.get_stats()
    assert 'total_modules' in stats
    assert stats['processed_topics'] >= 1
