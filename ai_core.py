# ai_core.py

import random
# collections.deque was not needed in this module

import torch
import torch.nn as nn
import wikipediaapi
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeurogenicModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [12, 10]

        layers = []
        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(sizes[-1], output_size))
        self.network = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        return self.network(x)


class SelfOrganizingModularNetwork(nn.Module):
    def __init__(self, input_size, output_size, novelty_threshold=0.6):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gatekeeper_neurons = torch.empty((0, input_size), device=device)
        self.modules_list = nn.ModuleList()
        self.novelty_threshold = novelty_threshold

    def get_gate_winner(self, x):
        if self.gatekeeper_neurons.shape[0] == 0:
            return -1, float('inf')

        with torch.no_grad():
            x_norm = x / (x.norm() + 1e-8)
            gates_norm = self.gatekeeper_neurons / (
                self.gatekeeper_neurons.norm(dim=1, keepdim=True) + 1e-8
            )

            similarities = torch.matmul(gates_norm, x_norm)
            best_similarity, winner_idx = torch.max(similarities, dim=0)

        return winner_idx.item(), (1 - best_similarity).item()

    def create_new_module(self, x_prototype):
        to_cat = x_prototype.unsqueeze(0)
        self.gatekeeper_neurons = torch.cat(
            (self.gatekeeper_neurons, to_cat),
            dim=0,
        )
        new_module = (
            NeurogenicModule(self.input_size, self.output_size)
            .to(device)
        )
        self.modules_list.append(new_module)

    def forward(self, x):
        winner_idx, distance = self.get_gate_winner(x)
        if winner_idx == -1 or distance > self.novelty_threshold:
            winner_idx = len(self.modules_list)
            self.create_new_module(x)

        return self.modules_list[winner_idx](x), winner_idx


class KnowledgeExtractor:
    """Extracts knowledge from Wikipedia and encodes with
    sentence-transformers.

    This is a lightweight extractor used by the older core. The newer
    `ai_core_v2.py` contains the preferred implementation.
    """

    def __init__(
        self, model_name='all-MiniLM-L6-v2', lang='es', logger_callback=print
    ):
        self.log = logger_callback
        self.log('Cargando modelo de lenguaje...')
        self.model = SentenceTransformer(model_name)
        self.wiki_api = wikipediaapi.Wikipedia(
            user_agent='MyCoolAIProject/1.0', language=lang
        )
        self.log('\u2713 Extractor de Conocimiento listo.')

    def get_knowledge_package(self, topic: str):
        try:
            page = self.wiki_api.page(topic)
            if not page.exists():
                return None, []

            links = list(page.links.keys())
            text = page.summary
            sentences = [
                s.strip()
                for s in text.split('.')
                if s.strip() and len(s) > 15
            ]
            if not sentences:
                return None, []

            with torch.no_grad():
                embeddings = self.model.encode(
                    sentences, convert_to_tensor=True, device=device
                )

            return torch.mean(embeddings, dim=0), links

        except Exception as e:
            self.log(f"  > Error extrayendo '{topic}': {e}")
            return None, []


class AutonomousScholar:
    def __init__(self, initial_topic: str, logger_callback=print):
        self.log = logger_callback
        self.brain = SelfOrganizingModularNetwork(
            input_size=384, output_size=1, novelty_threshold=0.6
        ).to(device)
        self.knowledge_extractor = KnowledgeExtractor(logger_callback=self.log)
        self.module_map = {}
        self.learning_frontier = [initial_topic]
        self.processed_topics = set()
        self.knowledge_graph = {}
        self.processed_topics_vectors = {}

    def learn_one_step(self):
        """Process a single concept (smoke-testable method)."""
        if not self.learning_frontier:
            self.log(
                '🏁 Frontera de aprendizaje agotada. El viaje ha terminado.'
            )
            return False

        topic = self.learning_frontier.pop(0)
        if topic in self.processed_topics:
            return True

        self.log(f'📚 Estudiando: "{topic}"')
        self.processed_topics.add(topic)

        vector, new_links = (
            self.knowledge_extractor.get_knowledge_package(topic)
        )
        if vector is None:
            return True

        num_modules_before = len(self.brain.modules_list)
        _, module_idx = self.brain(vector)

        if len(self.brain.modules_list) > num_modules_before:
            self.log(f'✨ ¡Nuevo Módulo {module_idx} creado para "{topic}"!')

        if module_idx not in self.module_map:
            self.module_map[module_idx] = []
        self.module_map[module_idx].append(topic)

        self.processed_topics_vectors[topic] = vector
        self.knowledge_graph[topic] = new_links[:5]

        random.shuffle(new_links)
        for link in new_links[:5]:
            if (
                link not in self.processed_topics
                and link not in self.learning_frontier
            ):
                self.learning_frontier.append(link)

        return True
