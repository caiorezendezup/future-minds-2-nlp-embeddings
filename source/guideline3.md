## 3. Noção de Chunking (Divisão de Documentos Longos)

### 🎯 **Por que Chunking é Fundamental?**

O **chunking** é uma técnica essencial no processamento de documentos longos, especialmente quando trabalhamos com:

- **🤖 Modelos de linguagem**: Que têm limites de tokens de entrada (ex: GPT-3.5 = 4K tokens)
- **🔍 Sistemas de busca semântica**: Chunks menores oferecem maior precisão na recuperação
- **💾 Armazenamento eficiente**: Embeddings de chunks são mais gerenciáveis que documentos completos
- **⚡ Performance**: Processamento paralelo de chunks independentes

### 🧠 **Conceitos-Chave**

- **Chunk Size**: Tamanho ideal do fragmento (geralmente 256-1024 tokens)
- **Overlap**: Sobreposição entre chunks para preservar contexto
- **Boundary Preservation**: Respeitar limites naturais do texto (sentenças, parágrafos)

### 3.1 Estratégias de Chunking

```python
import re
from typing import List, Tuple

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Inicializa o chunker com parâmetros configuráveis
        
        Args:
            chunk_size: Tamanho máximo do chunk em caracteres
            overlap: Número de caracteres de sobreposição entre chunks
        
        💡 Dica: chunk_size típico = 256-1024, overlap = 10-20% do chunk_size
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        📝 ESTRATÉGIA 1: Divisão por Sentenças
        
        ✅ Vantagens:
        - Preserva integridade semântica das frases
        - Mantém contexto completo dentro de cada chunk
        - Ideal para textos narrativos e artigos
        
        ❌ Desvantagens:
        - Chunks podem ter tamanhos muito variados
        - Sentenças muito longas podem exceder o limite
        
        🎯 Melhor uso: Documentos com sentenças bem estruturadas
        """
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:  # Pular sentenças vazias
                continue
                
            # Verificar se adicionar a sentença excede o limite
            potential_chunk = current_chunk + sentence + ". "
            
            if len(potential_chunk) < self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Salvar chunk atual se não estiver vazio
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Iniciar novo chunk com a sentença atual
                current_chunk = sentence + ". "
        
        # Adicionar último chunk se houver conteúdo
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_tokens(self, text: str) -> List[str]:
        """
        🔤 ESTRATÉGIA 2: Divisão por Tokens (Palavras) com Sobreposição
        
        ✅ Vantagens:
        - Chunks de tamanho consistente
        - Sobreposição preserva contexto entre chunks
        - Controle preciso sobre o tamanho
        
        ❌ Desvantagens:
        - Pode quebrar sentenças no meio
        - Sobreposição pode causar redundância
        
        🎯 Melhor uso: Quando precisão de tamanho é crucial
        """
        words = text.split()
        chunks = []
        
        # Iterar com step = chunk_size - overlap para criar sobreposição
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
            
            # Parar se não há mais palavras suficientes
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        📄 ESTRATÉGIA 3: Divisão por Parágrafos
        
        ✅ Vantagens:
        - Preserva estrutura lógica do documento
        - Mantém tópicos relacionados juntos
        - Respeita formatação original
        
        ❌ Desvantagens:
        - Parágrafos podem ser muito longos ou curtos
        - Dependente da qualidade da formatação
        
        🎯 Melhor uso: Documentos bem estruturados (artigos, papers)
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            potential_chunk = current_chunk + paragraph + "\n\n"
            
            if len(potential_chunk) < self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Salvar chunk atual se não estiver vazio
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Verificar se parágrafo sozinho excede limite
                if len(paragraph) > self.chunk_size:
                    # Dividir parágrafo longo em sentenças
                    sentences = re.split(r'[.!?]+', paragraph)
                    temp_chunk = ""
                    for sentence in sentences:
                        if sentence.strip():
                            if len(temp_chunk + sentence) < self.chunk_size:
                                temp_chunk += sentence.strip() + ". "
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = sentence.strip() + ". "
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def analyze_chunks(self, chunks: List[str]) -> None:
        """
        📊 Método auxiliar para analisar qualidade dos chunks
        """
        if not chunks:
            print("❌ Nenhum chunk foi gerado!")
            return
        
        sizes = [len(chunk) for chunk in chunks]
        word_counts = [len(chunk.split()) for chunk in chunks]
        
        print(f"📈 ANÁLISE DOS CHUNKS:")
        print(f"   Total de chunks: {len(chunks)}")
        print(f"   Tamanho médio: {sum(sizes)/len(sizes):.1f} caracteres")
        print(f"   Tamanho min/max: {min(sizes)}/{max(sizes)} caracteres")
        print(f"   Palavras médias por chunk: {sum(word_counts)/len(word_counts):.1f}")
        print(f"   Configuração: chunk_size={self.chunk_size}, overlap={self.overlap}")

# 🚀 EXEMPLO PRÁTICO EXPANDIDO
def exemplo_chunking_completo():
    """Demonstração completa das estratégias de chunking"""
    
    texto_exemplo = """
A inteligência artificial é uma área da ciência da computação que se concentra 
na criação de sistemas capazes de realizar tarefas que normalmente requerem 
inteligência humana. Isso inclui aprendizado, raciocínio, percepção e 
processamento de linguagem natural.

Os embeddings são uma técnica fundamental em IA que converte dados categóricos 
ou textuais em representações vetoriais densas. Essas representações capturam 
relações semânticas entre os dados de forma que itens similares tenham 
representações próximas no espaço vetorial.

O processamento de linguagem natural (NLP) utiliza essas técnicas para 
compreender e gerar texto humano. Aplicações incluem tradução automática, 
análise de sentimentos, chatbots e sistemas de recomendação baseados em conteúdo.
"""
    
    print("🎓 DEMONSTRAÇÃO: ESTRATÉGIAS DE CHUNKING")
    print("=" * 60)
    
    # Configurar chunker
    chunker = DocumentChunker(chunk_size=200, overlap=30)
    
    # Testar cada estratégia
    strategies = [
        ("Sentenças", chunker.chunk_by_sentences),
        ("Tokens", chunker.chunk_by_tokens),
        ("Parágrafos", chunker.chunk_by_paragraphs)
    ]
    
    for strategy_name, strategy_method in strategies:
        print(f"\n🔍 ESTRATÉGIA: {strategy_name.upper()}")
        print("-" * 40)
        
        chunks = strategy_method(texto_exemplo)
        chunker.analyze_chunks(chunks)
        
        print(f"\n📝 Primeiros 2 chunks:")
        for i, chunk in enumerate(chunks[:2]):
            print(f"   Chunk {i+1}: {chunk[:100]}...")
            
        # Demonstrar sobreposição (apenas para tokens)
        if strategy_name == "Tokens" and len(chunks) > 1:
            overlap_demo = set(chunks[0].split()) & set(chunks[1].split())
            print(f"   🔄 Palavras em sobreposição: {len(overlap_demo)}")

# Executar exemplo
exemplo_chunking_completo()
```

### 🎯 **Escolhendo a Estratégia Ideal**

| **Estratégia** | **Melhor Para** | **Evitar Quando** |
|----------------|-----------------|-------------------|
| **Sentenças** | Textos narrativos, artigos | Sentenças muito longas |
| **Tokens** | Controle preciso de tamanho | Contexto semântico é crucial |
| **Parágrafos** | Documentos estruturados | Parágrafos inconsistentes |

### 🔧 **Parâmetros Recomendados**

```python
# Para diferentes casos de uso:
CONFIGS = {
    'gpt-3.5': {'chunk_size': 3000, 'overlap': 300},    # ~4K tokens
    'bert': {'chunk_size': 400, 'overlap': 50},         # ~512 tokens
    'busca_semantica': {'chunk_size': 800, 'overlap': 100},
    'qa_system': {'chunk_size': 1500, 'overlap': 200}
}
```

### 💡 **Dicas Avançadas**

1. **📏 Medição em Tokens**: Use tokenizadores reais em vez de caracteres
2. **🔄 Overlap Inteligente**: Termine chunks em pontos naturais
3. **📊 Validação**: Sempre analise a qualidade dos chunks gerados
4. **⚖️ Trade-offs**: Chunks menores = mais precisão, chunks maiores = mais contexto
