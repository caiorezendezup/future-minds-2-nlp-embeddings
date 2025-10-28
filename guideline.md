Vou criar uma estrutura hierárquica detalhada para sua aula sobre Embeddings na prática com exemplos em Python:

## 1. Recap sobre Embeddings

### 1.1 Conceitos Fundamentais
```python
# Exemplo conceitual: representação de palavras como vetores
import numpy as np

# Representação tradicional (one-hot encoding)
vocabulario = ["gato", "cachorro", "animal", "felino"]
gato_onehot = [1, 0, 0, 0]
cachorro_onehot = [0, 1, 0, 0]

# Representação com embeddings (vetores densos)
gato_embedding = np.array([0.2, 0.8, 0.1, 0.9])
cachorro_embedding = np.array([0.3, 0.7, 0.2, 0.8])
```

### 1.2 Vantagens dos Embeddings
```python
from sklearn.metrics.pairwise import cosine_similarity

# Demonstração de similaridade semântica
embeddings = {
    "gato": np.array([0.2, 0.8, 0.1, 0.9]),
    "felino": np.array([0.25, 0.75, 0.15, 0.85]),
    "cachorro": np.array([0.3, 0.7, 0.2, 0.8]),
    "carro": np.array([0.9, 0.1, 0.8, 0.2])
}

# Calculando similaridade
sim_gato_felino = cosine_similarity([embeddings["gato"]], [embeddings["felino"]])[0][0]
sim_gato_carro = cosine_similarity([embeddings["gato"]], [embeddings["carro"]])[0][0]

print(f"Similaridade gato-felino: {sim_gato_felino:.3f}")
print(f"Similaridade gato-carro: {sim_gato_carro:.3f}")
```

## 2. Normalização e Dimensionalidade dos Embeddings

### 2.1 Normalização de Vetores
```python
import numpy as np
from sklearn.preprocessing import normalize

class EmbeddingNormalizer:
    @staticmethod
    def l2_normalize(embedding):
        """Normalização L2 (norma euclidiana)"""
        return embedding / np.linalg.norm(embedding)
    
    @staticmethod
    def min_max_normalize(embedding):
        """Normalização Min-Max"""
        return (embedding - embedding.min()) / (embedding.max() - embedding.min())
    
    @staticmethod
    def z_score_normalize(embedding):
        """Normalização Z-Score"""
        return (embedding - embedding.mean()) / embedding.std()

# Exemplo prático
embedding_raw = np.array([1.5, -0.8, 2.3, -1.2, 0.9])
normalizer = EmbeddingNormalizer()

embedding_l2 = normalizer.l2_normalize(embedding_raw)
embedding_minmax = normalizer.min_max_normalize(embedding_raw)
embedding_zscore = normalizer.z_score_normalize(embedding_raw)

print(f"Original: {embedding_raw}")
print(f"L2 Normalizado: {embedding_l2}")
print(f"Min-Max Normalizado: {embedding_minmax}")
print(f"Z-Score Normalizado: {embedding_zscore}")
```

### 2.2 Redução de Dimensionalidade
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class DimensionalityReducer:
    def __init__(self):
        self.pca = None
        self.tsne = None
    
    def reduce_with_pca(self, embeddings, n_components=2):
        """Redução usando PCA"""
        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(embeddings)
    
    def reduce_with_tsne(self, embeddings, n_components=2):
        """Redução usando t-SNE"""
        self.tsne = TSNE(n_components=n_components, random_state=42)
        return self.tsne.fit_transform(embeddings)
    
    def plot_embeddings(self, embeddings_2d, labels=None):
        """Visualização dos embeddings reduzidos"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        
        if labels:
            for i, label in enumerate(labels):
                plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        
        plt.title("Visualização de Embeddings (2D)")
        plt.xlabel("Dimensão 1")
        plt.ylabel("Dimensão 2")
        plt.show()
```

## 3. Noção de Chunking (Divisão de Documentos Longos)

### 3.1 Estratégias de Chunking
```python
import re
from typing import List, Tuple

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Divisão por sentenças"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(current_chunk + sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_tokens(self, text: str) -> List[str]:
        """Divisão por tokens com sobreposição"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Divisão por parágrafos"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# Exemplo de uso
texto_exemplo = """
A inteligência artificial é uma área da ciência da computação que se concentra 
na criação de sistemas capazes de realizar tarefas que normalmente requerem 
inteligência humana. Isso inclui aprendizado, raciocínio, percepção e 
processamento de linguagem natural.

Os embeddings são uma técnica fundamental em IA que converte dados categóricos 
ou textuais em representações vetoriais densas. Essas representações capturam 
relações semânticas entre os dados de forma que itens similares tenham 
representações próximas no espaço vetorial.
"""

chunker = DocumentChunker(chunk_size=100, overlap=20)
chunks_sentences = chunker.chunk_by_sentences(texto_exemplo)
chunks_tokens = chunker.chunk_by_tokens(texto_exemplo)

print("Chunks por sentenças:")
for i, chunk in enumerate(chunks_sentences):
    print(f"Chunk {i+1}: {chunk}\n")
```

## 4. Introdução ao Processamento de Linguagem Natural (NLP) e Tokens

### 4.1 Tokenização
```python
import nltk
from transformers import AutoTokenizer
import spacy

class TextTokenizer:
    def __init__(self):
        # Baixar recursos necessários (executar apenas uma vez)
        # nltk.download('punkt')
        # nltk.download('stopwords')
        
        self.tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
        # self.nlp = spacy.load('en_core_web_sm')  # Para inglês
    
    def tokenize_simple(self, text: str) -> List[str]:
        """Tokenização simples por espaços"""
        return text.lower().split()
    
    def tokenize_nltk(self, text: str) -> List[str]:
        """Tokenização usando NLTK"""
        tokens = nltk.word_tokenize(text.lower())
        return tokens
    
    def tokenize_transformer(self, text: str) -> List[str]:
        """Tokenização usando transformers"""
        tokens = self.tokenizer_bert.tokenize(text)
        return tokens
    
    def get_token_ids(self, text: str) -> List[int]:
        """Conversão para IDs de tokens"""
        return self.tokenizer_bert.encode(text)

# Exemplo prático
tokenizer = TextTokenizer()
texto = "Embeddings são representações vetoriais de texto."

print("Tokenização simples:", tokenizer.tokenize_simple(texto))
print("Tokenização NLTK:", tokenizer.tokenize_nltk(texto))
print("Tokenização Transformer:", tokenizer.tokenize_transformer(texto))
print("Token IDs:", tokenizer.get_token_ids(texto))
```

### 4.2 Pré-processamento de Texto
```python
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """Limpeza básica do texto"""
        # Remover caracteres especiais
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Converter para minúsculas
        text = text.lower()
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remoção de stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Aplicar stemming"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Aplicar lemmatização"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_pipeline(self, text: str) -> List[str]:
        """Pipeline completo de pré-processamento"""
        # Limpeza
        clean_text = self.clean_text(text)
        # Tokenização
        tokens = clean_text.split()
        # Remover stop words
        tokens = self.remove_stopwords(tokens)
        # Lemmatização
        tokens = self.lemmatize_tokens(tokens)
        return tokens
```

## 5. Implementação Prática com Sentence-Transformers

### 5.1 Sistema de Busca com Embeddings
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Tuple, Dict
import uuid

class EmbeddingSearchSystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = {}
        self.embeddings = {}
        self.document_ids = []
    
    def add_document(self, text: str, metadata: Dict = None) -> str:
        """Adicionar documento ao sistema"""
        doc_id = str(uuid.uuid4())
        
        # Armazenar documento
        self.documents[doc_id] = {
            'text': text,
            'metadata': metadata or {}
        }
        
        # Gerar embedding
        embedding = self.model.encode([text])[0]
        self.embeddings[doc_id] = embedding
        self.document_ids.append(doc_id)
        
        return doc_id
    
    def remove_document(self, doc_id: str) -> bool:
        """Remover documento do sistema"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            del self.embeddings[doc_id]
            self.document_ids.remove(doc_id)
            return True
        return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Buscar documentos similares"""
        if not self.documents:
            return []
        
        # Gerar embedding da consulta
        query_embedding = self.model.encode([query])[0]
        
        # Calcular similaridades
        similarities = []
        for doc_id in self.document_ids:
            doc_embedding = self.embeddings[doc_id]
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append((doc_id, similarity, self.documents[doc_id]['text']))
        
        # Ordenar por similaridade
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def update_document(self, doc_id: str, new_text: str, new_metadata: Dict = None):
        """Atualizar documento existente"""
        if doc_id in self.documents:
            # Atualizar texto e metadata
            self.documents[doc_id]['text'] = new_text
            if new_metadata:
                self.documents[doc_id]['metadata'].update(new_metadata)
            
            # Regenerar embedding
            embedding = self.model.encode([new_text])[0]
            self.embeddings[doc_id] = embedding
            
            return True
        return False
    
    def save_system(self, filepath: str):
        """Salvar sistema em arquivo"""
        system_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'document_ids': self.document_ids
        }
        with open(filepath, 'wb') as f:
            pickle.dump(system_data, f)
    
    def load_system(self, filepath: str):
        """Carregar sistema de arquivo"""
        with open(filepath, 'rb') as f:
            system_data = pickle.load(f)
        
        self.documents = system_data['documents']
        self.embeddings = system_data['embeddings']
        self.document_ids = system_data['document_ids']

# Exemplo de uso prático
def exemplo_sistema_busca():
    # Inicializar sistema
    search_system = EmbeddingSearchSystem()
    
    # Adicionar documentos
    documentos = [
        "Python é uma linguagem de programação versátil e fácil de aprender.",
        "Machine Learning utiliza algoritmos para encontrar padrões em dados.",
        "Embeddings convertem texto em representações vetoriais densas.",
        "Deep Learning é um subcampo do Machine Learning que usa redes neurais.",
        "Natural Language Processing permite que computadores entendam texto humano."
    ]
    
    doc_ids = []
    for i, doc in enumerate(documentos):
        doc_id = search_system.add_document(doc, {'categoria': 'tecnologia', 'indice': i})
        doc_ids.append(doc_id)
        print(f"Documento adicionado: {doc_id}")
    
    # Realizar buscas
    queries = [
        "aprendizado de máquina",
        "programação em Python",
        "processamento de texto"
    ]
    
    for query in queries:
        print(f"\nBusca por: '{query}'")
        results = search_system.search(query, top_k=3)
        
        for i, (doc_id, similarity, text) in enumerate(results):
            print(f"{i+1}. Similaridade: {similarity:.3f}")
            print(f"   Texto: {text}")
            print(f"   ID: {doc_id}")
    
    # Demonstrar remoção
    print(f"\nRemovendo documento: {doc_ids[0]}")
    search_system.remove_document(doc_ids[0])
    
    # Busca após remoção
    print(f"\nBusca após remoção por: 'Python'")
    results = search_system.search("Python", top_k=3)
    for i, (doc_id, similarity, text) in enumerate(results):
        print(f"{i+1}. Similaridade: {similarity:.3f} - {text}")

# Executar exemplo
exemplo_sistema_busca()
```

### 5.2 Sistema Avançado com Chunking
```python
class AdvancedEmbeddingSystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 512):
        self.model = SentenceTransformer(model_name)
        self.chunker = DocumentChunker(chunk_size=chunk_size)
        self.search_system = EmbeddingSearchSystem(model_name)
    
    def add_long_document(self, text: str, doc_title: str = None, metadata: Dict = None) -> List[str]:
        """Adicionar documento longo com chunking automático"""
        chunks = self.chunker.chunk_by_sentences(text)
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'document_title': doc_title or f'Document_{len(self.search_system.documents)}',
                'chunk_index': i,
                'total_chunks': len(chunks),
                'is_chunk': True
            })
            
            chunk_id = self.search_system.add_document(chunk, chunk_metadata)
            chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    def search_with_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Busca com informações de contexto"""
        results = self.search_system.search(query, top_k)
        
        contextualized_results = []
        for doc_id, similarity, text in results:
            doc_info = self.search_system.documents[doc_id]
            
            result = {
                'id': doc_id,
                'text': text,
                'similarity': similarity,
                'metadata': doc_info['metadata']
            }
            
            # Adicionar contexto se for chunk
            if doc_info['metadata'].get('is_chunk', False):
                result['document_title'] = doc_info['metadata'].get('document_title')
                result['chunk_position'] = f"{doc_info['metadata']['chunk_index'] + 1}/{doc_info['metadata']['total_chunks']}"
            
            contextualized_results.append(result)
        
        return contextualized_results

# Exemplo com documento longo
def exemplo_documento_longo():
    advanced_system = AdvancedEmbeddingSystem(chunk_size=200)
    
    documento_longo = """
    A inteligência artificial (IA) é uma área da ciência da computação que se concentra 
    na criação de sistemas capazes de realizar tarefas que normalmente requerem 
    inteligência humana. Isso inclui aprendizado, raciocínio, percepção, 
    processamento de linguagem natural e tomada de decisões.
    
    O machine learning é um subcampo da IA que permite que os computadores aprendam 
    e melhorem automaticamente através da experiência, sem serem explicitamente 
    programados para cada tarefa específica. Os algoritmos de machine learning 
    constroem modelos baseados em dados de treinamento para fazer previsões 
    ou tomar decisões.
    
    Deep learning, por sua vez, é um subcampo do machine learning que utiliza 
    redes neurais artificiais com múltiplas camadas para modelar e compreender 
    dados complexos. Essas redes são inspiradas no funcionamento do cérebro humano 
    e são especialmente eficazes em tarefas como reconhecimento de imagem, 
    processamento de linguagem natural e reconhecimento de fala.
    
    Os embeddings são uma técnica fundamental utilizada em muitas aplicações 
    de IA e NLP. Eles convertem dados categóricos ou textuais em representações 
    vetoriais densas que capturam relações semânticas. Isso permite que algoritmos 
    de machine learning trabalhem mais efetivamente com dados textuais, 
    encontrando padrões e similaridades que não seriam óbvios em representações 
    mais simples.
    """
    
    # Adicionar documento longo
    chunk_ids = advanced_system.add_long_document(
        documento_longo, 
        "Introdução à Inteligência Artificial",
        {'autor': 'Sistema de Exemplos', 'categoria': 'educacional'}
    )
    
    print(f"Documento dividido em {len(chunk_ids)} chunks")
    
    # Realizar buscas
    queries = [
        "redes neurais",
        "machine learning algoritmos",
        "embeddings semântica"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Busca por: '{query}'")
        print('='*50)
        
        results = advanced_system.search_with_context(query, top_k=3)
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. Similaridade: {result['similarity']:.3f}")
            if 'document_title' in result:
                print(f"   Documento: {result['document_title']}")
                print(f"   Posição: {result['chunk_position']}")
            print(f"   Texto: {result['text'][:100]}...")

exemplo_documento_longo()
```

### 5.3 Métricas e Avaliação
```python
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import seaborn as sns

class EmbeddingEvaluator:
    def __init__(self, search_system: EmbeddingSearchSystem):
        self.search_system = search_system
    
    def evaluate_search_quality(self, test_queries: List[Tuple[str, List[str]]]) -> Dict:
        """Avaliar qualidade da busca usando queries de teste"""
        metrics = {
            'precision_at_k': [],
            'recall_at_k': [],
            'mrr': []  # Mean Reciprocal Rank
        }
        
        for query, relevant_docs in test_queries:
            results = self.search_system.search(query, top_k=10)
            result_ids = [doc_id for doc_id, _, _ in results]
            
            # Precision@K e Recall@K
            for k in [1, 3, 5, 10]:
                top_k_results = result_ids[:k]
                relevant_found = len(set(top_k_results) & set(relevant_docs))
                
                precision = relevant_found / k if k > 0 else 0
                recall = relevant_found / len(relevant_docs) if len(relevant_docs) > 0 else 0
                
                metrics['precision_at_k'].append((k, precision))
                metrics['recall_at_k'].append((k, recall))
            
            # Mean Reciprocal Rank
            for i, doc_id in enumerate(result_ids):
                if doc_id in relevant_docs:
                    metrics['mrr'].append(1 / (i + 1))
                    break
            else:
                metrics['mrr'].append(0)
        
        return metrics
    
    def plot_similarity_distribution(self, query: str, num_samples: int = 100):
        """Plotar distribuição de similaridades"""
        results = self.search_system.search(query, top_k=num_samples)
        similarities = [sim for _, sim, _ in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
        plt.title(f'Distribuição de Similaridades para: "{query}"')
        plt.xlabel('Similaridade (Cosseno)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_embedding_space(self):
        """Analisar o espaço de embeddings"""
        if len(self.search_system.embeddings) < 2:
            print("Não há embeddings suficientes para análise")
            return
        
        embeddings_matrix = np.array(list(self.search_system.embeddings.values()))
        
        # Estatísticas básicas
        print("Análise do Espaço de Embeddings:")
        print(f"Número de documentos: {len(embeddings_matrix)}")
        print(f"Dimensionalidade: {embeddings_matrix.shape[1]}")
        print(f"Norma média: {np.mean(np.linalg.norm(embeddings_matrix, axis=1)):.3f}")
        print(f"Desvio padrão da norma: {np.std(np.linalg.norm(embeddings_matrix, axis=1)):.3f}")
        
        # Matriz de similaridade
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='coolwarm', center=0)
        plt.title('Matriz de Similaridade entre Documentos')
        plt.show()
```

Esta estrutura hierárquica fornece uma base sólida para sua aula, progredindo dos conceitos básicos até implementações práticas avançadas. Cada seção inclui código funcional que pode ser executado e modificado pelos alunos para experimentação prática.