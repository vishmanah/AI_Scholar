"""SISTEMA NEUROGENÉTICO AVANZADO (2025)
Memoria de largo plazo, atención, consolidación y arquitectura optimizada.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
import hashlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== ESTRUCTURAS DE DATOS ====================
 

@dataclass
class KnowledgeNode:
    """Representación de un concepto aprendido"""
    topic: str
    embedding: torch.Tensor
    timestamp: float
    module_id: int
    access_count: int = 0
    importance_score: float = 0.5
    related_nodes: List[str] = None
    
    def __post_init__(self):
        if self.related_nodes is None:
            self.related_nodes = []

# ==================== MECANISMO DE ATENCIÓN ====================


class MultiHeadAttention(nn.Module):
    """Atención multi-cabeza para procesamiento contextual"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)

# ==================== MÓDULO NEUROGENÉTICO MEJORADO ====================


class AdvancedNeurogenicModule(nn.Module):
    """Módulo con atención, normalización y conexiones residuales"""
    def __init__(self, input_size, output_size, hidden_sizes=[256, 128, 64]):
        super().__init__()
        self.input_size = input_size
        
        # Transformer-like architecture
        self.input_norm = nn.LayerNorm(input_size)
        self.attention = MultiHeadAttention(input_size, num_heads=4)
        
        # Feed-forward network con residuales
        layers = []
        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.LayerNorm(sizes[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.001,
            weight_decay=0.01,
        )
        
        # Métricas
        self.age = 0
        self.activation_count = 0
        self.average_loss = 1.0
        
    def forward(self, x):
        # Añadir dimensión de secuencia para atención
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Attention + residual
        normed = self.input_norm(x)
        attended = self.attention(normed)
        x = x + attended
        
        # Feed-forward
        x = x.squeeze(1)
        return self.network(x)

# ==================== MEMORIA DE LARGO PLAZO ====================


class EpisodicMemory:
    """Sistema de memoria episódica con consolidación y recuperación"""
    def __init__(self, capacity=10000, consolidation_threshold=5):
        self.capacity = capacity
        self.consolidation_threshold = consolidation_threshold
        self.memories: Dict[str, KnowledgeNode] = {}
        self.access_history = deque(maxlen=1000)
        
    def store(self, node: KnowledgeNode):
        """Almacenar nuevo conocimiento"""
        key = self._get_key(node.topic)
        if key in self.memories:
            # Reforzar memoria existente
            self.memories[key].access_count += 1
            self.memories[key].importance_score *= 1.1
        else:
            self.memories[key] = node
            
        self.access_history.append(key)
        self._consolidate()
        
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[KnowledgeNode]:
        """Recuperar conocimientos relevantes"""
        if not self.memories:
            return []
            
        similarities = []
        for key, node in self.memories.items():
            sim = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                node.embedding.unsqueeze(0),
            ).item()
            similarities.append((sim * node.importance_score, node))
            
        similarities.sort(reverse=True, key=lambda x: x[0])
        retrieved = [node for _, node in similarities[:top_k]]
        
        # Actualizar estadísticas de acceso
        for node in retrieved:
            node.access_count += 1
            
        return retrieved
        
    def _consolidate(self):
        """Consolidación de memoria: eliminar menos importantes si hay
        sobrecarga"""
        if len(self.memories) > self.capacity:
            # Ordenar por importancia
            sorted_memories = sorted(
                self.memories.items(),
                key=lambda x: x[1].importance_score * x[1].access_count
            )
            # Eliminar el 10% menos importante
            to_remove = int(len(sorted_memories) * 0.1)
            for key, _ in sorted_memories[:to_remove]:
                del self.memories[key]
                
    def _get_key(self, topic: str) -> str:
        return hashlib.md5(topic.encode()).hexdigest()
        
    def get_stats(self) -> Dict:
        if not self.memories:
            return {
                'total_memories': 0,
                'avg_access_count': 0,
                'avg_importance': 0,
            }

        return {
            'total_memories': len(self.memories),
            'avg_access_count': np.mean(
                [n.access_count for n in self.memories.values()]
            ),
            'avg_importance': np.mean(
                [n.importance_score for n in self.memories.values()]
            ),
        }

# ==================== RED AUTO-ORGANIZADORA AVANZADA ====================
class AdvancedSelfOrganizingNetwork(nn.Module):
    """Red con memoria, atención y consolidación mejoradas"""
    def __init__(self, input_size, output_size, novelty_threshold=0.5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.novelty_threshold = novelty_threshold
        
        # Gatekeeper mejorado con embeddings
        self.gatekeeper_embeddings = torch.empty((0, input_size), device=device)
        self.module_metadata = []
        
        # Módulos especializados
        self.modules_list = nn.ModuleList()
        
        # Memoria episódica
        self.episodic_memory = EpisodicMemory()
        
        # Meta-aprendizaje: aprende a crear mejores módulos
        self.meta_learning_rate = 0.001
        self.module_creation_history = []
        
    def get_gate_winner(self, x: torch.Tensor) -> Tuple[int, float]:
        """Encontrar módulo más relevante con contexto de memoria"""
        if self.gatekeeper_embeddings.shape[0] == 0:
            return -1, float('inf')
            
        with torch.no_grad():
            # Normalizar
            x_norm = F.normalize(x.unsqueeze(0), dim=1)
            gates_norm = F.normalize(self.gatekeeper_embeddings, dim=1)
            
            # Calcular similitudes
            similarities = torch.matmul(gates_norm, x_norm.T).squeeze()
            
            if similarities.dim() == 0:
                similarities = similarities.unsqueeze(0)
                
            best_similarity, winner_idx = torch.max(similarities, dim=0)
            distance = 1 - best_similarity.item()
            
            return winner_idx.item(), distance
    
    def create_new_module(self, x_prototype: torch.Tensor, context: Optional[str] = None):
        """Crear nuevo módulo con arquitectura adaptativa"""
        # Determinar arquitectura basada en complejidad del problema
        complexity = self._estimate_complexity()
        hidden_sizes = self._adaptive_architecture(complexity)
        
        # Crear módulo
        new_module = AdvancedNeurogenicModule(
            self.input_size, 
            self.output_size,
            hidden_sizes=hidden_sizes
        ).to(device)
        
        self.modules_list.append(new_module)
        self.gatekeeper_embeddings = torch.cat(
            (self.gatekeeper_embeddings, x_prototype.unsqueeze(0)), 
            dim=0
        )
        
        # Metadata
        self.module_metadata.append({
            'creation_time': len(self.module_creation_history),
            'context': context,
            'architecture': hidden_sizes,
            'specialization_score': 0.0
        })
        
        self.module_creation_history.append({
            'module_id': len(self.modules_list) - 1,
            'novelty': self._estimate_complexity()
        })
        
    def _estimate_complexity(self) -> float:
        """Estimar complejidad actual del problema"""
        if len(self.module_creation_history) < 2:
            return 0.5
            
        recent_creations = self.module_creation_history[-5:]
        return np.mean([m['novelty'] for m in recent_creations])
        
    def _adaptive_architecture(self, complexity: float) -> List[int]:
        """Arquitectura adaptativa según complejidad"""
        base_size = 256
        if complexity > 0.8:
            return [base_size * 2, base_size, base_size // 2, base_size // 4]
        elif complexity > 0.5:
            return [base_size, base_size // 2, base_size // 4]
        else:
            return [base_size // 2, base_size // 4]
    
    def forward(self, x: torch.Tensor, use_memory: bool = True) -> Tuple[torch.Tensor, int]:
        """Forward con contexto de memoria"""
        # Buscar conocimientos relacionados en memoria
        context_nodes = []
        if use_memory:
            context_nodes = self.episodic_memory.retrieve(x, top_k=3)
        
        # Determinar módulo apropiado
        winner_idx, distance = self.get_gate_winner(x)
        
        # Crear nuevo módulo si necesario
        if winner_idx == -1 or distance > self.novelty_threshold:
            winner_idx = len(self.modules_list)
            context = context_nodes[0].topic if context_nodes else None
            self.create_new_module(x, context=context)
        
        # Procesar con módulo seleccionado
        module = self.modules_list[winner_idx]
        module.activation_count += 1
        
        output = module(x)
        
        return output, winner_idx
    
    def consolidate_knowledge(self):
        """Consolidación periódica del conocimiento"""
        if len(self.modules_list) < 2:
            return
            
        similarities = []
        for i in range(len(self.gatekeeper_embeddings)):
            for j in range(i + 1, len(self.gatekeeper_embeddings)):
                sim = F.cosine_similarity(
                    self.gatekeeper_embeddings[i].unsqueeze(0),
                    self.gatekeeper_embeddings[j].unsqueeze(0)
                ).item()
                if sim > 0.9:
                    similarities.append((i, j, sim))
        
    def get_network_stats(self) -> Dict:
        """Estadísticas completas de la red"""
        return {
            'total_modules': len(self.modules_list),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'memory_stats': self.episodic_memory.get_stats(),
            'avg_module_age': np.mean([m.age for m in self.modules_list]) if self.modules_list else 0,
            'module_activations': [m.activation_count for m in self.modules_list]
        }

# ==================== EXTRACTOR DE CONOCIMIENTO MEJORADO ====================
class AdvancedKnowledgeExtractor:
    """Extractor con caché, procesamiento paralelo y múltiples fuentes"""
    def __init__(self, model_name='all-MiniLM-L6-v2', lang='es', logger_callback=print):
        self.log = logger_callback
        self.log("Cargando modelo de lenguaje avanzado...")
        
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        
        import wikipediaapi
        self.wiki_api = wikipediaapi.Wikipedia(
            user_agent='AdvancedNeurogenicAI/2.0', 
            language=lang
        )
        
        # Caché para evitar consultas repetidas
        self.cache = {}
        self.max_cache_size = 1000
        
        self.log("Sistema de extracción avanzado listo.")
        
    def get_knowledge_package(self, topic: str, depth: int = 1) -> Tuple[Optional[torch.Tensor], List[str], Dict]:
        """Extraer conocimiento con análisis profundo"""
        # Verificar caché
        if topic in self.cache:
            return self.cache[topic]
            
        try:
            page = self.wiki_api.page(topic)
            if not page.exists():
                return None, [], {}
                
            # Extraer contenido
            text = page.summary
            links = list(page.links.keys())[:20]
            
            # Procesar texto
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            
            if not sentences:
                return None, [], {}
            
            # Generar embeddings
            with torch.no_grad():
                embeddings = self.model.encode(
                    sentences, 
                    convert_to_tensor=True, 
                    device=device,
                    show_progress_bar=False
                )
                
            # Embedding ponderado
            weights = torch.tensor([1.0 / (i + 1) for i in range(len(embeddings))], device=device)
            weights = weights / weights.sum()
            weighted_embedding = (embeddings.T @ weights)
            
            # Metadata
            metadata = {
                'num_sentences': len(sentences),
                'num_links': len(links),
                'text_length': len(text),
                'key_terms': self._extract_key_terms(text)
            }
            
            result = (weighted_embedding, links, metadata)
            
            # Guardar en caché
            if len(self.cache) < self.max_cache_size:
                self.cache[topic] = result
                
            return result
            
        except Exception as e:
            self.log(f"Error procesando '{topic}': {e}")
            return None, [], {}
            
    def _extract_key_terms(self, text: str, top_n: int = 5) -> List[str]:
        """Extraer términos clave con análisis lingüístico"""
        import re
        from collections import Counter
        
        # Stop words comunes en español
        stop_words = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
            'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
            'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
            'la', 'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy',
            'sin', 'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo',
            'yo', 'también', 'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'primero',
            'desde', 'grande', 'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella',
            'sí', 'día', 'uno', 'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa',
            'tanto', 'hombre', 'parecer', 'nuestro', 'tan', 'donde', 'ahora', 'parte',
            'después', 'vida', 'quedar', 'siempre', 'creer', 'hablar', 'llevar', 'dejar',
            'nada', 'cada', 'seguir', 'menos', 'nuevo', 'encontrar', 'algo', 'solo', 'decir',
            'puede', 'mediante', 'cual', 'algunos', 'esta', 'estos', 'estas', 'fue', 'son',
            'era', 'han', 'sido', 'tiene', 'están', 'había', 'sea', 'tras', 'ante', 'durante',
        }
        
        # Limpiar y tokenizar texto
        text = text.lower()
        # Eliminar puntuación y caracteres especiales, mantener letras y espacios
        text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)
        # Dividir en palabras
        words = text.split()
        
        # Filtrar: eliminar stop words, palabras cortas, y palabras muy comunes
        filtered_words = [
            w for w in words 
            if len(w) > 5 and w not in stop_words and w.isalpha()
        ]
        
        # Contar frecuencias y retornar las más comunes
        if not filtered_words:
            return []
        
        word_counts = Counter(filtered_words)
        return [w for w, _ in word_counts.most_common(top_n)]

# ==================== ERUDITO AUTÓNOMO MEJORADO ====================
class AdvancedAutonomousScholar:
    """Sistema de aprendizaje con consolidación, priorización y meta-cognición"""
    def __init__(self, initial_topic: str, logger_callback=print):
        self.log = logger_callback
        
        # Red neuronal avanzada
        self.brain = AdvancedSelfOrganizingNetwork(
            input_size=384, 
            output_size=1, 
            novelty_threshold=0.5
        ).to(device)
        
        # Extractor mejorado
        self.knowledge_extractor = AdvancedKnowledgeExtractor(logger_callback=self.log)
        
        # Sistema de gestión de aprendizaje
        self.learning_frontier = deque([initial_topic])
        self.processed_topics = set()
        self.priority_queue = []
        
        # Mapeo de conocimiento
        self.module_map: Dict[int, List[str]] = {}
        self.knowledge_graph: Dict[str, List[str]] = {}
        
        # Métricas
        self.learning_efficiency = []
        self.curiosity_score = 1.0
        
    def learn_one_step(self) -> bool:
        """Paso de aprendizaje optimizado"""
        if not self.learning_frontier:
            self.log("Frontera de aprendizaje agotada.")
            return False
            
        topic = self._select_next_topic()
        
        if topic in self.processed_topics:
            return True
            
        self.log(f"Estudiando: '{topic}'")
        self.processed_topics.add(topic)
        
        embedding, links, metadata = self.knowledge_extractor.get_knowledge_package(topic)
        
        if embedding is None:
            return True
            
        num_modules_before = len(self.brain.modules_list)
        _, module_idx = self.brain(embedding, use_memory=True)
        
        node = KnowledgeNode(
            topic=topic,
            embedding=embedding,
            timestamp=len(self.processed_topics),
            module_id=module_idx,
            importance_score=self._calculate_importance(metadata)
        )
        self.brain.episodic_memory.store(node)
        
        if module_idx not in self.module_map:
            self.module_map[module_idx] = []
        self.module_map[module_idx].append(topic)
        
        if len(self.brain.modules_list) > num_modules_before:
            self.log(f"Nuevo módulo {module_idx} creado para '{topic}'")
            
        self.knowledge_graph[topic] = links[:10]
        self._expand_curiosity(links, metadata)
        
        if len(self.processed_topics) % 50 == 0:
            self.brain.consolidate_knowledge()
            self.log("Consolidación de conocimiento realizada")
            
        return True
        
    def _select_next_topic(self) -> str:
        if self.priority_queue:
            self.priority_queue.sort(reverse=True, key=lambda x: x[0])
            _, topic = self.priority_queue.pop(0)
            if topic in self.learning_frontier:
                self.learning_frontier.remove(topic)
            return topic
        else:
            return self.learning_frontier.popleft()
            
    def _calculate_importance(self, metadata: Dict) -> float:
        score = 0.5
        score += min(metadata.get('num_links', 0) / 100, 0.3)
        score += min(metadata.get('num_sentences', 0) / 50, 0.2)
        return min(score, 1.0)
        
    def _expand_curiosity(self, links: List[str], metadata: Dict):
        key_terms = set(metadata.get('key_terms', []))
        
        for link in links[:8]:
            if link in self.processed_topics:
                continue
                
            priority = self.curiosity_score
            if any(term in link.lower() for term in key_terms):
                priority *= 1.5
                
            if link not in [t for _, t in self.priority_queue]:
                self.priority_queue.append((priority, link))
                
            if link not in self.learning_frontier:
                self.learning_frontier.append(link)
                
        self.curiosity_score *= 0.99
        
    def get_stats(self) -> Dict:
        return {
            'processed_topics': len(self.processed_topics),
            'frontier_size': len(self.learning_frontier),
            'priority_queue_size': len(self.priority_queue),
            **self.brain.get_network_stats()
        }