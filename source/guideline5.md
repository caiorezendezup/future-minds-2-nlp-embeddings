## 5. Implementação Prática com Sentence-Transformers

### 🎯 **Por que Sentence-Transformers?**

O **Sentence-Transformers** é uma biblioteca Python que facilita o uso de modelos pré-treinados para gerar embeddings de alta qualidade. Diferente de embeddings de palavras individuais (Word2Vec), ele gera embeddings para **frases e documentos completos**.

**Vantagens:**
- 🚀 **Plug-and-play**: Modelos pré-treinados prontos para uso
- 🎯 **Semântica contextual**: Entende o significado completo das frases
- 🌍 **Multilíngue**: Suporte a diversos idiomas
- ⚡ **Eficiente**: Otimizado para produção

### 5.1 Sistema de Busca com Embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Tuple, Dict
import uuid

class EmbeddingSearchSystem:
    """
    🔍 Sistema de Busca Semântica com Embeddings
    
    Este sistema demonstra como construir um mecanismo de busca que entende
    o SIGNIFICADO do texto, não apenas palavras-chave exatas.
    
    Exemplo: Buscar por "cachorro" também encontrará textos sobre "cão" ou "pet"
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        🏗️ Inicialização do Sistema
        
        Args:
            model_name: Nome do modelo Sentence-Transformers
            
        📊 Modelos Populares:
        - 'all-MiniLM-L6-v2': Rápido, 384 dimensões, boa qualidade geral
        - 'all-mpnet-base-v2': Melhor qualidade, 768 dimensões, mais lento  
        - 'paraphrase-multilingual': Suporte multilíngue
        """
        print(f"🔄 Carregando modelo: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # 📚 Estruturas de dados do sistema
        self.documents = {}      # doc_id -> {text, metadata}
        self.embeddings = {}     # doc_id -> embedding_vector  
        self.document_ids = []   # Lista ordenada de IDs
        
        print(f"✅ Sistema inicializado com modelo {model_name}")
        print(f"📐 Dimensões dos embeddings: {self.model.get_sentence_embedding_dimension()}")
    
    def add_document(self, text: str, metadata: Dict = None) -> str:
        """
        ➕ Adicionar Documento ao Sistema
        
        Processo:
        1. Gera ID único para o documento
        2. Armazena texto e metadados  
        3. Converte texto em embedding vetorial
        4. Armazena embedding para buscas futuras
        
        Args:
            text: Texto do documento
            metadata: Informações adicionais (categoria, autor, etc.)
            
        Returns:
            doc_id: Identificador único do documento
        """
        doc_id = str(uuid.uuid4())
        print(f"📝 Adicionando documento: {doc_id[:8]}...")
        
        # 💾 Armazenar documento e metadados
        self.documents[doc_id] = {
            'text': text,
            'metadata': metadata or {}
        }
        
        # 🧠 Gerar embedding (conversão texto → vetor)
        print(f"🔄 Gerando embedding para: '{text[:50]}...'")
        embedding = self.model.encode([text])[0]
        self.embeddings[doc_id] = embedding
        self.document_ids.append(doc_id)
        
        print(f"✅ Documento adicionado com {len(embedding)} dimensões")
        return doc_id
    
    def remove_document(self, doc_id: str) -> bool:
        """
        🗑️ Remover Documento do Sistema
        
        Remove completamente o documento de todas as estruturas:
        - Texto e metadados
        - Embedding vetorial
        - Lista de IDs
        """
        if doc_id in self.documents:
            print(f"🗑️ Removendo documento: {doc_id[:8]}...")
            
            del self.documents[doc_id]
            del self.embeddings[doc_id] 
            self.document_ids.remove(doc_id)
            
            print("✅ Documento removido com sucesso")
            return True
        
        print("❌ Documento não encontrado")
        return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        🔍 Buscar Documentos Similares
        
        Algoritmo de Busca Semântica:
        1. Converte consulta em embedding
        2. Calcula similaridade cosseno com todos os documentos
        3. Ordena por similaridade (maior = mais relevante)
        4. Retorna top_k mais similares
        
        Args:
            query: Texto da consulta
            top_k: Número máximo de resultados
            
        Returns:
            Lista de tuplas: (doc_id, similaridade, texto)
        """
        if not self.documents:
            print("⚠️ Nenhum documento no sistema")
            return []
        
        print(f"🔍 Buscando por: '{query}'")
        
        # 🧠 Converter consulta em embedding
        query_embedding = self.model.encode([query])[0]
        
        # 📊 Calcular similaridade com todos os documentos
        similarities = []
        for doc_id in self.document_ids:
            doc_embedding = self.embeddings[doc_id]
            
            # 📐 Similaridade cosseno: mede ângulo entre vetores
            # Valores: -1 (opostos) a 1 (idênticos)
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append((doc_id, similarity, self.documents[doc_id]['text']))
        
        # 📈 Ordenar por relevância (similaridade decrescente)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 Encontrados {len(similarities)} documentos")
        return similarities[:top_k]
    
    def update_document(self, doc_id: str, new_text: str, new_metadata: Dict = None):
        """
        🔄 Atualizar Documento Existente
        
        Importante: Quando o texto muda, o embedding DEVE ser recalculado
        porque a representação vetorial mudou!
        """
        if doc_id in self.documents:
            print(f"🔄 Atualizando documento: {doc_id[:8]}...")
            
            # Atualizar texto e metadata
            self.documents[doc_id]['text'] = new_text
            if new_metadata:
                self.documents[doc_id]['metadata'].update(new_metadata)
            
            # 🧠 CRÍTICO: Regenerar embedding para novo texto
            embedding = self.model.encode([new_text])[0]
            self.embeddings[doc_id] = embedding
            
            print("✅ Documento e embedding atualizados")
            return True
        
        print("❌ Documento não encontrado")
        return False
    
    def save_system(self, filepath: str):
        """
        💾 Persistir Sistema em Arquivo
        
        Salva todo o estado do sistema:
        - Documentos e metadados
        - Embeddings pré-computados (economiza tempo!)
        - Lista de IDs
        
        ⚠️ Nota: O modelo não é salvo - deve ser recarregado na inicialização
        """
        print(f"💾 Salvando sistema em: {filepath}")
        
        system_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'document_ids': self.document_ids
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_data, f)
        
        print(f"✅ Sistema salvo com {len(self.documents)} documentos")
    
    def load_system(self, filepath: str):
        """
        📂 Carregar Sistema de Arquivo
        
        Restaura estado completo do sistema, incluindo embeddings
        pré-computados (evita recalcular tudo!)
        """
        print(f"📂 Carregando sistema de: {filepath}")
        
        with open(filepath, 'rb') as f:
            system_data = pickle.load(f)
        
        self.documents = system_data['documents']
        self.embeddings = system_data['embeddings']
        self.document_ids = system_data['document_ids']
        
        print(f"✅ Sistema carregado com {len(self.documents)} documentos")

# 🚀 EXEMPLO PRÁTICO EDUCACIONAL
def exemplo_sistema_busca():
    """
    📚 Demonstração Completa do Sistema de Busca Semântica
    
    Este exemplo mostra:
    1. Como inicializar o sistema
    2. Adicionar documentos com metadados
    3. Realizar buscas semânticas
    4. Interpretar resultados de similaridade
    5. Gerenciar documentos (adicionar/remover)
    """
    
    print("=" * 60)
    print("🎓 DEMONSTRAÇÃO: SISTEMA DE BUSCA SEMÂNTICA")
    print("=" * 60)
    
    # 🏗️ Inicializar sistema
    search_system = EmbeddingSearchSystem()
    
    # 📚 Conjunto de documentos sobre tecnologia
    documentos = [
        "Python é uma linguagem de programação versátil e fácil de aprender.",
        "Machine Learning utiliza algoritmos para encontrar padrões em dados.",
        "Embeddings convertem texto em representações vetoriais densas.",
        "Deep Learning é um subcampo do Machine Learning que usa redes neurais.",
        "Natural Language Processing permite que computadores entendam texto humano."
    ]
    
    print(f"\n📝 Adicionando {len(documentos)} documentos...")
    doc_ids = []
    for i, doc in enumerate(documentos):
        doc_id = search_system.add_document(
            doc, 
            {'categoria': 'tecnologia', 'indice': i}
        )
        doc_ids.append(doc_id)
    
    # 🔍 Demonstrar buscas semânticas
    queries = [
        "aprendizado de máquina",      # Deve encontrar ML e DL
        "programação em Python",       # Deve encontrar Python
        "processamento de texto"       # Deve encontrar NLP
    ]
    
    print(f"\n🔍 Realizando {len(queries)} buscas semânticas...")
    for query in queries:
        print(f"\n{'='*50}")
        print(f"🔍 Consulta: '{query}'")
        print('='*50)
        
        results = search_system.search(query, top_k=3)
        
        for i, (doc_id, similarity, text) in enumerate(results):
            print(f"\n{i+1}. 📊 Similaridade: {similarity:.3f}")
            print(f"   📝 Texto: {text}")
            print(f"   🆔 ID: {doc_id[:8]}...")
            
            # 💡 Interpretação educacional da similaridade
            if similarity > 0.7:
                print("   ✅ Alta relevância")
            elif similarity > 0.5:
                print("   🟡 Relevância moderada")
            else:
                print("   🔴 Baixa relevância")
    
    # 🗑️ Demonstrar remoção de documento
    print(f"\n{'='*50}")
    print("🗑️ DEMONSTRAÇÃO: REMOÇÃO DE DOCUMENTO")
    print('='*50)
    
    print(f"Removendo documento: {doc_ids[0][:8]}...")
    search_system.remove_document(doc_ids[0])
    
    # 🔍 Busca após remoção
    print(f"\n🔍 Busca após remoção por: 'Python'")
    results = search_system.search("Python", top_k=3)
    
    print(f"📊 Resultados encontrados: {len(results)}")
    for i, (doc_id, similarity, text) in enumerate(results):
        print(f"{i+1}. Similaridade: {similarity:.3f}")
        print(f"   {text[:60]}...")
    
    # 💾 Demonstrar persistência
    print(f"\n{'='*50}")
    print("💾 DEMONSTRAÇÃO: SALVAR/CARREGAR SISTEMA")
    print('='*50)
    
    filename = "sistema_busca.pkl"
    search_system.save_system(filename)
    
    # Criar novo sistema e carregar dados
    novo_sistema = EmbeddingSearchSystem()
    novo_sistema.load_system(filename)
    
    print("✅ Sistema recarregado com sucesso!")
    
    # 📊 Estatísticas finais
    print(f"\n{'='*50}")
    print("📊 ESTATÍSTICAS DO SISTEMA")
    print('='*50)
    print(f"📚 Total de documentos: {len(search_system.documents)}")
    print(f"🧠 Dimensões dos embeddings: {len(list(search_system.embeddings.values())[0])}")
    print(f"💾 Tamanho médio dos embeddings: {np.mean([emb.nbytes for emb in search_system.embeddings.values()])} bytes")

# 🎯 PONTOS-CHAVE PARA FIXAÇÃO
def pontos_chave_educacionais():
    """
    📝 Conceitos Fundamentais Demonstrados
    """
    print("\n" + "="*60)
    print("🎯 CONCEITOS-CHAVE APRENDIDOS")
    print("="*60)
    
    conceitos = [
        "🔍 Busca Semântica: Encontra significado, não apenas palavras exatas",
        "📊 Similaridade Cosseno: Mede ângulo entre vetores (0-1 para embeddings normalizados)",
        "🧠 Embeddings Contextuais: Capturam significado completo de frases/documentos",
        "💾 Persistência: Salvar embeddings evita recálculos custosos",
        "🔄 Atualização Dinâmica: Texto novo = embedding novo",
        "📈 Ranking por Relevância: Ordenação por similaridade decrescente"
    ]
    
    for conceito in conceitos:
        print(f"  {conceito}")
    
    print(f"\n💡 DICA PRÁTICA:")
    print("  Em produção, use índices aproximados (FAISS, Annoy) para")
    print("  buscas rápidas em milhões de documentos!")

# 🚀 Executar demonstração completa
if __name__ == "__main__":
    exemplo_sistema_busca()
    pontos_chave_educacionais()
```

## 5.2 Sistema Avançado com Chunking

### 🎯 **Conceitos Fundamentais**

Este sistema demonstra como lidar com **documentos longos** em aplicações reais de NLP, onde textos excedem os limites de processamento de modelos de linguagem.

**Por que precisamos de chunking?**
- **Limitações de modelos**: GPT-3.5 (4K tokens), BERT (512 tokens)
- **Qualidade de busca**: Chunks menores = resultados mais precisos
- **Performance**: Processamento paralelo de fragmentos independentes

```python
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import uuid

class AdvancedEmbeddingSystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 512):
        """
        🏗️ INICIALIZAÇÃO DO SISTEMA AVANÇADO
        
        Args:
            model_name: Modelo de embedding (all-MiniLM-L6-v2 = rápido e eficiente)
            chunk_size: Tamanho máximo dos chunks em caracteres
        
        💡 Escolha do modelo:
        - all-MiniLM-L6-v2: 384 dimensões, rápido, boa qualidade geral
        - all-mpnet-base-v2: 768 dimensões, melhor qualidade, mais lento
        - multilingual: Para textos em múltiplos idiomas
        """
        self.model = SentenceTransformer(model_name)
        self.chunker = DocumentChunker(chunk_size=chunk_size)
        self.search_system = EmbeddingSearchSystem(model_name)
    
    def add_long_document(self, text: str, doc_title: str = None, metadata: Dict = None) -> List[str]:
        """
        📄 PROCESSAMENTO DE DOCUMENTOS LONGOS
        
        Esta função resolve o problema fundamental: como processar textos
        que excedem os limites dos modelos de linguagem?
        
        Estratégia:
        1. Dividir documento em chunks menores
        2. Gerar embedding para cada chunk
        3. Manter rastreabilidade (qual chunk pertence a qual documento)
        
        Returns:
            Lista de IDs dos chunks criados
        """
        # ETAPA 1: Divisão inteligente do documento
        chunks = self.chunker.chunk_by_sentences(text)
        chunk_ids = []
        
        # ETAPA 2: Processar cada chunk individualmente
        for i, chunk in enumerate(chunks):
            # 🏷️ METADADOS ENRIQUECIDOS
            # Preservar informação sobre origem e posição do chunk
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'document_title': doc_title or f'Document_{len(self.search_system.documents)}',
                'chunk_index': i,                    # Posição no documento original
                'total_chunks': len(chunks),         # Total de chunks do documento
                'is_chunk': True                     # Flag para identificar chunks
            })
            
            # ETAPA 3: Adicionar chunk ao sistema de busca
            chunk_id = self.search_system.add_document(chunk, chunk_metadata)
            chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    def search_with_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        🔍 BUSCA CONTEXTUALIZADA
        
        Diferencial: Além da similaridade, retorna informações contextuais
        que ajudam o usuário a entender de onde veio o resultado.
        
        Informações contextuais incluem:
        - Título do documento original
        - Posição do chunk no documento
        - Metadados adicionais
        """
        # ETAPA 1: Busca semântica tradicional
        results = self.search_system.search(query, top_k)
        
        # ETAPA 2: Enriquecimento com contexto
        contextualized_results = []
        for doc_id, similarity, text in results:
            doc_info = self.search_system.documents[doc_id]
            
            # 📊 ESTRUTURA DE RESULTADO ENRIQUECIDA
            result = {
                'id': doc_id,
                'text': text,
                'similarity': similarity,
                'metadata': doc_info['metadata']
            }
            
            # 🔗 ADICIONAR CONTEXTO PARA CHUNKS
            if doc_info['metadata'].get('is_chunk', False):
                result['document_title'] = doc_info['metadata'].get('document_title')
                result['chunk_position'] = f"{doc_info['metadata']['chunk_index'] + 1}/{doc_info['metadata']['total_chunks']}"
            
            contextualized_results.append(result)
        
        return contextualized_results
```

### 🚀 **Exemplo Prático Educacional**

```python
def exemplo_documento_longo():
    """
    📚 DEMONSTRAÇÃO COMPLETA: DO DOCUMENTO LONGO À BUSCA INTELIGENTE
    
    Este exemplo simula um cenário real onde você tem:
    - Um documento extenso sobre IA
    - Necessidade de fazer buscas específicas
    - Importância de manter contexto dos resultados
    """
    
    print("🎓 INICIANDO SISTEMA AVANÇADO DE EMBEDDINGS")
    print("=" * 60)
    
    # 🏗️ CONFIGURAÇÃO DO SISTEMA
    advanced_system = AdvancedEmbeddingSystem(chunk_size=200)
    
    # 📄 DOCUMENTO DE EXEMPLO (simulando artigo científico)
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
    
    # 📥 PROCESSAMENTO DO DOCUMENTO
    print("📥 Processando documento longo...")
    chunk_ids = advanced_system.add_long_document(
        documento_longo, 
        "Introdução à Inteligência Artificial",
        {'autor': 'Sistema de Exemplos', 'categoria': 'educacional'}
    )
    
    print(f"✅ Documento dividido em {len(chunk_ids)} chunks")
    print(f"📊 IDs dos chunks: {chunk_ids[:3]}..." if len(chunk_ids) > 3 else f"📊 IDs: {chunk_ids}")
    
    # 🔍 DEMONSTRAÇÃO DE BUSCAS DIVERSIFICADAS
    queries = [
        "redes neurais",              # Busca por conceito específico
        "machine learning algoritmos", # Busca por área + método
        "embeddings semântica"        # Busca por técnica + propriedade
    ]
    
    print(f"\n🎯 Realizando {len(queries)} buscas demonstrativas...")
    
    for query_idx, query in enumerate(queries, 1):
        print(f"\n{'='*50}")
        print(f"🔍 BUSCA {query_idx}: '{query}'")
        print('='*50)
        
        # 🚀 EXECUTAR BUSCA CONTEXTUALIZADA
        results = advanced_system.search_with_context(query, top_k=3)
        
        if not results:
            print("❌ Nenhum resultado encontrado")
            continue
        
        # 📊 ANÁLISE DOS RESULTADOS
        print(f"📈 {len(results)} resultados encontrados:")
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 RESULTADO {i}:")
            print(f"   🎯 Similaridade: {result['similarity']:.3f}")
            
            # 📍 INFORMAÇÕES CONTEXTUAIS
            if 'document_title' in result:
                print(f"   📚 Documento: {result['document_title']}")
            if 'chunk_position' in result:
                print(f"   📍 Posição: Chunk {result['chunk_position']}")
            
            # 📝 PREVIEW DO CONTEÚDO
            preview_text = result['text'][:100].replace('\n', ' ')
            print(f"   📝 Texto: {preview_text}...")
            
            # 🏷️ METADADOS ADICIONAIS
            metadata = result['metadata']
            if 'autor' in metadata:
                print(f"   👤 Autor: {metadata['autor']}")
            if 'categoria' in metadata:
                print(f"   🏷️ Categoria: {metadata['categoria']}")

# 🎓 EXECUTAR DEMONSTRAÇÃO
exemplo_documento_longo()
```

### 📊 **Análise Educacional dos Resultados**

```python
def analisar_resultados_educacional():
    """Análise detalhada para fins educacionais"""
    
    print("\n" + "="*60)
    print("📊 ANÁLISE EDUCACIONAL DOS RESULTADOS")
    print("="*60)
    
    print("""
    🎯 O QUE OBSERVAR NOS RESULTADOS:
    
    1. 📈 SCORES DE SIMILARIDADE:
       • 0.8-1.0: Correspondência muito alta (quase exata)
       • 0.6-0.8: Correspondência boa (semanticamente relacionada)
       • 0.4-0.6: Correspondência moderada (pode ser relevante)
       • 0.0-0.4: Correspondência baixa (pouco relevante)
    
    2. 🎪 CONTEXTO PRESERVADO:
       • Título do documento original mantido
       • Posição do chunk no documento (ex: "2/4" = chunk 2 de 4)
       • Metadados customizados preservados
    
    3. 🔍 QUALIDADE DA BUSCA:
       • "redes neurais" → deve encontrar parágrafo sobre deep learning
       • "algoritmos" → deve encontrar seção sobre machine learning
       • "embeddings" → deve encontrar parágrafo específico sobre embeddings
    
    4. ⚡ VANTAGENS DO CHUNKING:
       • Respostas mais precisas (chunks focados vs documento inteiro)
       • Melhor performance (embeddings menores)
       • Contexto preservado (sabemos de onde veio cada resultado)
    """)

analisar_resultados_educacional()
```

### 🎓 **Pontos-Chave**

1. **🧩 Chunking Inteligente**: Dividir por sentenças preserva significado
2. **🏷️ Metadados Ricos**: Rastreabilidade é fundamental em sistemas reais
3. **🔍 Busca Contextualizada**: Não basta encontrar, precisa saber de onde veio
4. **⚖️ Trade-offs**: Chunks menores = mais precisão, mas mais complexidade
5. **🚀 Escalabilidade**: Sistema funciona com documentos de qualquer tamanho

## 5.3 Métricas e Avaliação

### 🎯 **Por que Avaliar Sistemas de Embeddings?**

A avaliação é fundamental para:
- **📊 Medir performance**: Quantificar quão bem o sistema funciona
- **🔍 Detectar problemas**: Identificar falhas na recuperação de documentos
- **⚖️ Comparar modelos**: Escolher a melhor abordagem para seu caso
- **🎯 Otimizar parâmetros**: Ajustar configurações do sistema
- **📈 Monitorar qualidade**: Acompanhar performance ao longo do tempo

```python
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import numpy as np
from typing import List, Tuple, Dict

class EmbeddingEvaluator:
    """
    🎓 CLASSE EDUCACIONAL: Avaliação de Sistemas de Embeddings
    
    Esta classe implementa métricas essenciais para avaliar:
    - Qualidade da busca semântica
    - Distribuição de similaridades
    - Estrutura do espaço de embeddings
    """
    
    def __init__(self, search_system):
        """
        Inicializar avaliador
        
        Args:
            search_system: Sistema de busca com embeddings (EmbeddingSearchSystem)
        """
        self.search_system = search_system
    
    def evaluate_search_quality(self, test_queries: List[Tuple[str, List[str]]]) -> Dict:
        """
        🎯 MÉTODO PRINCIPAL: Avaliar qualidade da busca usando queries de teste
        
        📚 CONCEITOS-CHAVE:
        - Precision@K: Proporção de documentos relevantes nos top-K resultados
        - Recall@K: Proporção dos documentos relevantes que foram encontrados
        - MRR: Mean Reciprocal Rank - posição média do primeiro resultado relevante
        
        Args:
            test_queries: Lista de tuplas (query, [doc_ids_relevantes])
            
        Returns:
            Dict com métricas calculadas
            
        💡 EXEMPLO DE USO:
        test_queries = [
            ("machine learning", ["doc1", "doc3", "doc7"]),
            ("python programming", ["doc2", "doc5"])
        ]
        """
        metrics = {
            'precision_at_k': [],  # Lista de tuplas (k, precision)
            'recall_at_k': [],     # Lista de tuplas (k, recall)
            'mrr': []              # Lista de valores MRR por query
        }
        
        print("🔍 AVALIANDO QUALIDADE DA BUSCA")
        print("=" * 50)
        
        for i, (query, relevant_docs) in enumerate(test_queries):
            print(f"\nQuery {i+1}: '{query}'")
            print(f"Documentos relevantes esperados: {len(relevant_docs)}")
            
            # Buscar documentos
            results = self.search_system.search(query, top_k=10)
            result_ids = [doc_id for doc_id, _, _ in results]
            
            print(f"Documentos retornados: {len(result_ids)}")
            
            # 📊 PRECISION@K e RECALL@K
            for k in [1, 3, 5, 10]:
                top_k_results = result_ids[:k]
                relevant_found = len(set(top_k_results) & set(relevant_docs))
                
                # Precision@K = Relevantes encontrados / K
                precision = relevant_found / k if k > 0 else 0
                
                # Recall@K = Relevantes encontrados / Total de relevantes
                recall = relevant_found / len(relevant_docs) if len(relevant_docs) > 0 else 0
                
                metrics['precision_at_k'].append((k, precision))
                metrics['recall_at_k'].append((k, recall))
                
                print(f"  P@{k}: {precision:.3f} | R@{k}: {recall:.3f}")
            
            # 🎯 MEAN RECIPROCAL RANK (MRR)
            # Encontra a posição do primeiro documento relevante
            for j, doc_id in enumerate(result_ids):
                if doc_id in relevant_docs:
                    mrr_score = 1 / (j + 1)  # Posição 1 = 1.0, Posição 2 = 0.5, etc.
                    metrics['mrr'].append(mrr_score)
                    print(f"  MRR: {mrr_score:.3f} (primeiro relevante na posição {j+1})")
                    break
            else:
                # Nenhum documento relevante encontrado
                metrics['mrr'].append(0)
                print(f"  MRR: 0.000 (nenhum documento relevante encontrado)")
        
        # 📈 RESUMO DAS MÉTRICAS
        self._print_metrics_summary(metrics)
        return metrics
    
    def _print_metrics_summary(self, metrics: Dict):
        """Imprimir resumo das métricas calculadas"""
        print("\n" + "="*50)
        print("📊 RESUMO DAS MÉTRICAS")
        print("="*50)
        
        # Agrupar métricas por K
        precision_by_k = {}
        recall_by_k = {}
        
        for k, precision in metrics['precision_at_k']:
            if k not in precision_by_k:
                precision_by_k[k] = []
            precision_by_k[k].append(precision)
        
        for k, recall in metrics['recall_at_k']:
            if k not in recall_by_k:
                recall_by_k[k] = []
            recall_by_k[k].append(recall)
        
        # Calcular médias
        print("Precision@K (média):")
        for k in sorted(precision_by_k.keys()):
            avg_precision = np.mean(precision_by_k[k])
            print(f"  P@{k}: {avg_precision:.3f}")
        
        print("\nRecall@K (média):")
        for k in sorted(recall_by_k.keys()):
            avg_recall = np.mean(recall_by_k[k])
            print(f"  R@{k}: {avg_recall:.3f}")
        
        avg_mrr = np.mean(metrics['mrr'])
        print(f"\nMRR médio: {avg_mrr:.3f}")
        
        # Interpretação educacional
        print("\n🎓 INTERPRETAÇÃO:")
        if avg_mrr > 0.7:
            print("✅ Excelente: Sistema encontra documentos relevantes nas primeiras posições")
        elif avg_mrr > 0.5:
            print("🟡 Bom: Sistema tem performance razoável")
        else:
            print("❌ Ruim: Sistema precisa de melhorias")
    
    def plot_similarity_distribution(self, query: str, num_samples: int = 100):
        """
        📊 VISUALIZAÇÃO: Distribuição de similaridades
        
        🎯 OBJETIVO: Entender como as similaridades se distribuem
        
        📚 O QUE ANALISAR:
        - Distribuição normal: Boa diversidade de documentos
        - Pico à esquerda: Muitos documentos irrelevantes
        - Pico à direita: Documentos muito similares (possível duplicação)
        - Distribuição uniforme: Falta de discriminação semântica
        """
        print(f"📊 ANALISANDO DISTRIBUIÇÃO DE SIMILARIDADES")
        print(f"Query: '{query}'")
        print("-" * 40)
        
        results = self.search_system.search(query, top_k=num_samples)
        similarities = [sim for _, sim, _ in results]
        
        if not similarities:
            print("❌ Nenhum resultado encontrado!")
            return
        
        # Estatísticas descritivas
        print(f"Número de documentos analisados: {len(similarities)}")
        print(f"Similaridade média: {np.mean(similarities):.3f}")
        print(f"Similaridade mediana: {np.median(similarities):.3f}")
        print(f"Desvio padrão: {np.std(similarities):.3f}")
        print(f"Min: {np.min(similarities):.3f} | Max: {np.max(similarities):.3f}")
        
        # Determinar número apropriado de bins
        n_bins = min(20, max(3, len(set(similarities))))
        
        # Criar visualização
        plt.figure(figsize=(12, 8))
        
        # Histograma principal
        plt.subplot(2, 2, 1)
        n, bins, patches = plt.hist(similarities, bins=n_bins, alpha=0.7, 
                                   edgecolor='black', color='skyblue')
        plt.title(f'Distribuição de Similaridades\nQuery: "{query}"')
        plt.xlabel('Similaridade (Cosseno)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        
        # Adicionar linha da média
        plt.axvline(np.mean(similarities), color='red', linestyle='--', 
                   label=f'Média: {np.mean(similarities):.3f}')
        plt.legend()
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(similarities, vert=True)
        plt.title('Box Plot das Similaridades')
        plt.ylabel('Similaridade')
        plt.grid(True, alpha=0.3)
        
        # Distribuição cumulativa
        plt.subplot(2, 2, 3)
        sorted_sims = np.sort(similarities)
        cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
        plt.plot(sorted_sims, cumulative, marker='o', markersize=3)
        plt.title('Distribuição Cumulativa')
        plt.xlabel('Similaridade')
        plt.ylabel('Proporção Acumulada')
        plt.grid(True, alpha=0.3)
        
        # Top-K similarities
        plt.subplot(2, 2, 4)
        top_k = min(20, len(similarities))
        plt.plot(range(1, top_k + 1), similarities[:top_k], 
                marker='o', markersize=4, color='orange')
        plt.title(f'Top-{top_k} Similaridades')
        plt.xlabel('Ranking')
        plt.ylabel('Similaridade')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Interpretação educacional
        print("\n🎓 INTERPRETAÇÃO DA DISTRIBUIÇÃO:")
        if np.std(similarities) < 0.1:
            print("⚠️  Baixa variabilidade - documentos muito similares entre si")
        elif np.mean(similarities) < 0.3:
            print("⚠️  Similaridades baixas - possível problema de relevância")
        elif np.mean(similarities) > 0.8:
            print("⚠️  Similaridades muito altas - possível overfitting ou duplicação")
        else:
            print("✅ Distribuição saudável de similaridades")
    
    def analyze_embedding_space(self):
        """
        🔍 ANÁLISE AVANÇADA: Estrutura do espaço de embeddings
        
        🎯 OBJETIVO: Entender as propriedades geométricas do espaço vetorial
        
        📚 O QUE ANALISAMOS:
        - Dimensionalidade e densidade
        - Distribuição de normas (magnitudes dos vetores)
        - Matriz de similaridade entre todos os documentos
        - Clusters e padrões estruturais
        """
        print("🔍 ANÁLISE DO ESPAÇO DE EMBEDDINGS")
        print("=" * 50)
        
        if len(self.search_system.embeddings) < 2:
            print("❌ Não há embeddings suficientes para análise (mínimo: 2)")
            return
        
        # Converter embeddings para matriz numpy
        embeddings_matrix = np.array(list(self.search_system.embeddings.values()))
        doc_ids = list(self.search_system.embeddings.keys())
        
        # 📊 ESTATÍSTICAS BÁSICAS
        print("📊 ESTATÍSTICAS BÁSICAS:")
        print(f"  Número de documentos: {len(embeddings_matrix)}")
        print(f"  Dimensionalidade: {embeddings_matrix.shape[1]}")
        
        # Análise de normas (magnitudes dos vetores)
        norms = np.linalg.norm(embeddings_matrix, axis=1)
        print(f"  Norma média: {np.mean(norms):.3f}")
        print(f"  Desvio padrão da norma: {np.std(norms):.3f}")
        print(f"  Norma mín/máx: {np.min(norms):.3f} / {np.max(norms):.3f}")
        
        # Análise de distribuição por dimensão
        print(f"  Média por dimensão: {np.mean(np.mean(embeddings_matrix, axis=0)):.3f}")
        print(f"  Desvio padrão por dimensão: {np.mean(np.std(embeddings_matrix, axis=0)):.3f}")
        
        # 🎯 MATRIZ DE SIMILARIDADE
        print("\n🎯 CALCULANDO MATRIZ DE SIMILARIDADE...")
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Estatísticas da matriz de similaridade
        # Remover diagonal (similaridade consigo mesmo = 1.0)
        off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        print(f"  Similaridade média entre documentos: {np.mean(off_diagonal):.3f}")
        print(f"  Desvio padrão das similaridades: {np.std(off_diagonal):.3f}")
        print(f"  Similaridade mín/máx: {np.min(off_diagonal):.3f} / {np.max(off_diagonal):.3f}")
        
        # 📊 VISUALIZAÇÕES
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Heatmap da matriz de similaridade
        ax1 = axes[0, 0]
        im = ax1.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title('Matriz de Similaridade entre Documentos')
        ax1.set_xlabel('Documento ID')
        ax1.set_ylabel('Documento ID')
        plt.colorbar(im, ax=ax1)
        
        # 2. Distribuição das similaridades
        ax2 = axes[0, 1]
        # Determinar número apropriado de bins para similaridades
        n_bins_sim = min(30, max(5, len(set(off_diagonal))))
        ax2.hist(off_diagonal, bins=n_bins_sim, alpha=0.7, edgecolor='black', color='lightgreen')
        ax2.axvline(np.mean(off_diagonal), color='red', linestyle='--', 
                   label=f'Média: {np.mean(off_diagonal):.3f}')
        ax2.set_title('Distribuição das Similaridades')
        ax2.set_xlabel('Similaridade Cosseno')
        ax2.set_ylabel('Frequência')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribuição das normas
        ax3 = axes[1, 0]
        # Determinar número apropriado de bins para normas
        unique_norms = np.unique(norms)
        if len(unique_norms) < 3:
            # Se há poucas normas únicas, usar bar plot
            ax3.bar(range(len(unique_norms)), 
                   [np.sum(norms == norm) for norm in unique_norms],
                   alpha=0.7, color='orange')
            ax3.set_xticks(range(len(unique_norms)))
            ax3.set_xticklabels([f'{norm:.3f}' for norm in unique_norms])
            ax3.set_title('Distribuição das Normas dos Vetores')
            ax3.set_xlabel('Norma (Magnitude)')
            ax3.set_ylabel('Frequência')
        else:
            n_bins_norms = min(20, max(3, len(unique_norms)))
            ax3.hist(norms, bins=n_bins_norms, alpha=0.7, edgecolor='black', color='orange')
            ax3.axvline(np.mean(norms), color='red', linestyle='--', 
                       label=f'Média: {np.mean(norms):.3f}')
            ax3.set_title('Distribuição das Normas dos Vetores')
            ax3.set_xlabel('Norma (Magnitude)')
            ax3.set_ylabel('Frequência')
            ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Heatmap de correlação entre dimensões (amostra)
        ax4 = axes[1, 1]
        # Usar apenas primeiras 20 dimensões para visualização
        sample_dims = min(20, embeddings_matrix.shape[1])
        correlation_matrix = np.corrcoef(embeddings_matrix[:, :sample_dims].T)
        im4 = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title(f'Correlação entre Dimensões\n(Primeiras {sample_dims} dimensões)')
        ax4.set_xlabel('Dimensão')
        ax4.set_ylabel('Dimensão')
        plt.colorbar(im4, ax=ax4)
        
        plt.tight_layout()
        plt.show()
        
        # 🎓 INTERPRETAÇÃO EDUCACIONAL
        print("\n🎓 INTERPRETAÇÃO DOS RESULTADOS:")
        print("-" * 40)
        
        if np.std(norms) < 0.1:
            print("✅ Normas consistentes - embeddings bem normalizados")
        else:
            print("⚠️  Normas variáveis - considere normalização L2")
        
        if np.mean(off_diagonal) > 0.7:
            print("⚠️  Similaridades muito altas - possível falta de diversidade")
        elif np.mean(off_diagonal) < 0.1:
            print("⚠️  Similaridades muito baixas - documentos muito diferentes")
        else:
            print("✅ Distribuição saudável de similaridades")
        
        if np.std(off_diagonal) < 0.1:
            print("⚠️  Pouca variabilidade - embeddings podem estar saturados")
        else:
            print("✅ Boa variabilidade nas similaridades")
        
        # Detectar possíveis clusters
        high_similarity_pairs = np.sum(off_diagonal > 0.8)
        total_pairs = len(off_diagonal)
        cluster_ratio = high_similarity_pairs / total_pairs
        
        print(f"\n🔍 DETECÇÃO DE CLUSTERS:")
        print(f"  Pares com alta similaridade (>0.8): {high_similarity_pairs}/{total_pairs} ({cluster_ratio:.1%})")
        
        if cluster_ratio > 0.3:
            print("🎯 Possíveis clusters detectados - documentos agrupados por temas")
        else:
            print("📊 Distribuição uniforme - boa diversidade temática")

# 🚀 EXEMPLO DE USO EDUCACIONAL
def exemplo_avaliacao_completa():
    """Demonstração completa do sistema de avaliação"""
    print("🎓 DEMONSTRAÇÃO: SISTEMA DE AVALIAÇÃO DE EMBEDDINGS")
    print("=" * 60)
    
    # Simular sistema de busca com documentos de exemplo
    from sentence_transformers import SentenceTransformer
    import uuid
    
    class EmbeddingSearchSystem:
        def __init__(self):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.documents = {}
            self.embeddings = {}
            self.document_ids = []
        
        def add_document(self, text, metadata=None):
            doc_id = str(uuid.uuid4())
            self.documents[doc_id] = {'text': text, 'metadata': metadata or {}}
            embedding = self.model.encode([text])[0]
            self.embeddings[doc_id] = embedding
            self.document_ids.append(doc_id)
            return doc_id
        
        def search(self, query, top_k=5):
            if not self.documents:
                return []
            
            query_embedding = self.model.encode([query])[0]
            similarities = []
            
            for doc_id in self.document_ids:
                doc_embedding = self.embeddings[doc_id]
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarities.append((doc_id, similarity, self.documents[doc_id]['text']))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    
    search_system = EmbeddingSearchSystem()
    
    # Adicionar documentos de exemplo
    documentos_exemplo = [
        "Python é uma linguagem de programação versátil",
        "Machine learning utiliza algoritmos para encontrar padrões",
        "Deep learning é um subcampo do machine learning",
        "Redes neurais artificiais imitam o cérebro humano",
        "Processamento de linguagem natural permite compreender texto"
    ]
    
    doc_ids = []
    for doc in documentos_exemplo:
        doc_id = search_system.add_document(doc)
        doc_ids.append(doc_id)
    
    # Criar avaliador
    evaluator = EmbeddingEvaluator(search_system)
    
    # Definir queries de teste
    test_queries = [
        ("aprendizado de máquina", [doc_ids[1], doc_ids[2]]),  # ML e DL
        ("linguagem de programação", [doc_ids[0]]),             # Python
        ("inteligência artificial", [doc_ids[1], doc_ids[2], doc_ids[3]])  # ML, DL, NN
    ]
    
    # 1. Avaliar qualidade da busca
    print("\n1️⃣ AVALIAÇÃO DA QUALIDADE DA BUSCA")
    metrics = evaluator.evaluate_search_quality(test_queries)
    
    # 2. Analisar distribuição de similaridades
    print("\n2️⃣ ANÁLISE DE DISTRIBUIÇÃO DE SIMILARIDADES")
    evaluator.plot_similarity_distribution("machine learning")
    
    # 3. Analisar espaço de embeddings
    print("\n3️⃣ ANÁLISE DO ESPAÇO DE EMBEDDINGS")
    evaluator.analyze_embedding_space()

# Executar exemplo
exemplo_avaliacao_completa()
```

### 🎯 **Métricas-Chave Explicadas**

#### 📊 **Precision@K**
- **Fórmula**: `Precision@K = Documentos Relevantes nos Top-K / K`
- **Interpretação**: "Dos K documentos que retornei, quantos são realmente relevantes?"
- **Exemplo**: Se busco por "Python" e nos top-3 resultados, 2 são relevantes → P@3 = 2/3 = 0.67

#### 📈 **Recall@K**
- **Fórmula**: `Recall@K = Documentos Relevantes Encontrados / Total de Relevantes`
- **Interpretação**: "Dos documentos relevantes que existem, quantos consegui encontrar?"
- **Exemplo**: Se existem 5 docs relevantes e encontrei 3 nos top-10 → R@10 = 3/5 = 0.60

#### 🎯 **Mean Reciprocal Rank (MRR)**
- **Fórmula**: `MRR = 1/posição_primeiro_relevante`
- **Interpretação**: "Quão rápido encontro o primeiro resultado relevante?"
- **Exemplo**: Primeiro relevante na posição 2 → MRR = 1/2 = 0.50

### 💡 **Dicas Educacionais**

1. **🎯 Precision vs Recall Trade-off**: Alta precision pode significar baixo recall e vice-versa
2. **📊 MRR é crucial**: Usuários geralmente olham apenas os primeiros resultados
3. **🔍 Distribuição de similaridades**: Revela muito sobre a qualidade dos embeddings
4. **⚖️ Balance é importante**: Nem muito similares (duplicação) nem muito diferentes (irrelevância)

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