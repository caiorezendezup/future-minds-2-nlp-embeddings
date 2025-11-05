## 4. Introdução ao Processamento de Linguagem Natural (NLP) e Tokens

### 🎯 **Por que NLP e Tokenização são Fundamentais?**

O **Processamento de Linguagem Natural (NLP)** é a ponte entre a linguagem humana e a compreensão computacional. Computadores não entendem texto diretamente - eles precisam converter palavras em números. A **tokenização** é o primeiro passo crucial nesse processo.

**Analogia**: Imagine que você precisa ensinar um alienígena a entender português. Primeiro, você dividiria as frases em palavras individuais (tokenização), depois explicaria o significado de cada palavra (embeddings).

### 4.1 Tokenização

**🔑 Conceito**: Tokenização é o processo de dividir texto em unidades menores chamadas **tokens** (palavras, subpalavras, caracteres).

```python
import nltk
from transformers import AutoTokenizer
import spacy
from typing import List

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("📥 Baixando recursos do NLTK...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    print("✅ Recursos baixados com sucesso!")

class TextTokenizer:
    def __init__(self):
        """
        Inicialização com diferentes tokenizadores
        
        💡 Dica: Execute uma vez para baixar recursos:
        nltk.download('punkt')
        nltk.download('stopwords')
        """
        # BERT tokenizer - usa subpalavras (subword tokenization)
        self.tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # SpaCy para processamento avançado (descomente se necessário)
        # self.nlp = spacy.load('en_core_web_sm')
    
    def tokenize_simple(self, text: str) -> List[str]:
        """
        🟢 MÉTODO 1: Tokenização Simples por Espaços
        
        ✅ Vantagens:
        - Rápido e simples
        - Não requer bibliotecas externas
        - Funciona bem para textos limpos
        
        ❌ Limitações:
        - Não trata pontuação adequadamente
        - Não considera contrações (don't → don't, não don + 't)
        - Sensível a espaços extras
        """
        return text.lower().split()
    
    def tokenize_nltk(self, text: str) -> List[str]:
        """
        🟡 MÉTODO 2: Tokenização com NLTK
        
        ✅ Vantagens:
        - Trata pontuação corretamente
        - Separa contrações (don't → ['do', "n't"])
        - Reconhece abreviações
        - Funciona com múltiplos idiomas
        
        🎯 Uso ideal: Textos gerais, análise linguística básica
        """
        tokens = nltk.word_tokenize(text.lower())
        return tokens
    
    def tokenize_transformer(self, text: str) -> List[str]:
        """
        🔵 MÉTODO 3: Tokenização de Transformers (BERT)
        
        ✅ Vantagens:
        - Usa subpalavras (lida com palavras raras/novas)
        - Consistente com modelos pré-treinados
        - Vocabulário fixo e otimizado
        - Trata palavras fora do vocabulário (OOV)
        
        🎯 Uso ideal: Quando usar embeddings de transformers
        
        Exemplo: "unhappiness" → ["un", "##happiness"]
        """
        tokens = self.tokenizer_bert.tokenize(text)
        return tokens
    
    def get_token_ids(self, text: str) -> List[int]:
        """
        🔢 Conversão para IDs Numéricos
        
        Por que precisamos de IDs?
        - Modelos de ML trabalham com números, não texto
        - Cada token tem um ID único no vocabulário
        - Permite processamento eficiente em batches
        
        Tokens especiais do BERT:
        - [CLS]: 101 (início da sequência)
        - [SEP]: 102 (separador/fim)
        - [PAD]: 0 (preenchimento)
        """
        return self.tokenizer_bert.encode(text)

# 🚀 EXEMPLO PRÁTICO EDUCACIONAL
def demonstrar_tokenizacao():
    """Demonstração comparativa dos métodos de tokenização"""
    
    tokenizer = TextTokenizer()
    
    # Texto com desafios comuns
    texto = "Embeddings são representações vetoriais de texto. Eles're muito úteis!"
    
    print("=" * 60)
    print("🎓 DEMONSTRAÇÃO: COMPARAÇÃO DE TOKENIZADORES")
    print("=" * 60)
    print(f"Texto original: '{texto}'")
    print()
    
    # Comparar métodos
    methods = [
        ("🟢 Tokenização Simples", tokenizer.tokenize_simple),
        ("🟡 Tokenização NLTK", tokenizer.tokenize_nltk),
        ("🔵 Tokenização BERT", tokenizer.tokenize_transformer)
    ]
    
    for name, method in methods:
        tokens = method(texto)
        print(f"{name}:")
        print(f"   Tokens: {tokens}")
        print(f"   Quantidade: {len(tokens)} tokens")
        print()
    
    # Mostrar IDs dos tokens
    token_ids = tokenizer.get_token_ids(texto)
    print("🔢 Token IDs (BERT):")
    print(f"   IDs: {token_ids}")
    print(f"   Quantidade: {len(token_ids)} tokens")
    
    # Explicar diferenças
    print("\n" + "=" * 60)
    print("📊 ANÁLISE COMPARATIVA")
    print("=" * 60)
    print("🟢 Simples: Não separou pontuação, manteve contração")
    print("🟡 NLTK: Separou pontuação, tratou melhor as palavras")
    print("🔵 BERT: Usou subpalavras, adicionou tokens especiais [CLS] e [SEP]")

# Executar demonstração
demonstrar_tokenizacao()
```

```python
from transformers import AutoTokenizer
from typing import List
import re

class TextTokenizerSimplified:
    def __init__(self):
        """Inicialização sem NLTK para evitar problemas de SSL"""
        self.tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_simple(self, text: str) -> List[str]:
        """🟢 MÉTODO 1: Tokenização Simples por Espaços"""
        return text.lower().split()

    def tokenize_regex(self, text: str) -> List[str]:
        """🟡 MÉTODO 2: Tokenização com Regex (substitui NLTK)"""
        # Remove pontuação e divide por espaços
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [token for token in tokens if token.strip()]

    def tokenize_transformer(self, text: str) -> List[str]:
        """🔵 MÉTODO 3: Tokenização de Transformers (BERT)"""
        tokens = self.tokenizer_bert.tokenize(text)
        return tokens

    def get_token_ids(self, text: str) -> List[int]:
        """🔢 Conversão para IDs Numéricos"""
        return self.tokenizer_bert.encode(text)

def demonstrar_tokenizacao_simplificada():
    """Demonstração sem NLTK"""
    tokenizer = TextTokenizerSimplified()
    
    texto = "Embeddings são representações vetoriais de texto. Eles're muito úteis!"
    
    print("=" * 60)
    print("🎓 DEMONSTRAÇÃO: COMPARAÇÃO DE TOKENIZADORES (SEM NLTK)")
    print("=" * 60)
    print(f"Texto original: '{texto}'")
    print()
    
    methods = [
        ("🟢 Tokenização Simples", tokenizer.tokenize_simple),
        ("🟡 Tokenização Regex", tokenizer.tokenize_regex),
        ("🔵 Tokenização BERT", tokenizer.tokenize_transformer)
    ]
    
    for name, method in methods:
        tokens = method(texto)
        print(f"{name}:")
        print(f"  Tokens: {tokens}")
        print(f"  Quantidade: {len(tokens)} tokens")
        print()
    
    token_ids = tokenizer.get_token_ids(texto)
    print("🔢 Token IDs (BERT):")
    print(f"  IDs: {token_ids}")
    print(f"  Quantidade: {len(token_ids)} tokens")

# Executar versão simplificada
demonstrar_tokenizacao_simplificada()
```

### 4.2 Pré-processamento de Texto

**🎯 Objetivo**: Limpar e padronizar texto para melhorar a qualidade dos embeddings e análises.

```python
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List

class TextPreprocessor:
    def __init__(self, language='english'):
        """
        Inicialização com recursos de pré-processamento
        
        Componentes principais:
        - Stop words: palavras muito comuns (the, and, is...)
        - Stemmer: reduz palavras ao radical (running → run)
        - Lemmatizer: reduz à forma canônica (better → good)
        """
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        🧹 ETAPA 1: Limpeza Básica do Texto
        
        Operações realizadas:
        1. Remove caracteres especiais e números
        2. Converte para minúsculas (case normalization)
        3. Remove espaços extras
        
        ✅ Por que fazer isso?
        - Reduz ruído nos dados
        - Padroniza formato
        - Melhora consistência dos embeddings
        """
        print(f"   📝 Texto original: '{text}'")
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        print(f"   🔧 Após remoção de especiais: '{text}'")
        
        # Converter para minúsculas
        text = text.lower()
        print(f"   📝 Após minúsculas: '{text}'")
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        print(f"   ✨ Texto limpo: '{text}'")
        
        return text
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        🚫 ETAPA 2: Remoção de Stop Words
        
        Stop words são palavras muito comuns que geralmente não carregam
        significado semântico importante: 'the', 'and', 'is', 'in', etc.
        
        ✅ Vantagens:
        - Reduz dimensionalidade
        - Foca em palavras com mais significado
        - Melhora eficiência computacional
        
        ❌ Cuidado:
        - Pode remover contexto importante em algumas tarefas
        - "Not good" → "good" (perde negação)
        """
        original_count = len(tokens)
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        removed_count = original_count - len(filtered_tokens)
        
        print(f"   🚫 Removidas {removed_count} stop words de {original_count} tokens")
        print(f"   📋 Tokens restantes: {filtered_tokens}")
        
        return filtered_tokens
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        🌱 ETAPA 3A: Stemming (Alternativa 1)
        
        Stemming remove sufixos para encontrar o "radical" da palavra.
        Algoritmo: Porter Stemmer (mais comum)
        
        Exemplos:
        - running, runs, ran → run
        - better, good → better, good (não conecta palavras relacionadas)
        
        ✅ Vantagens: Rápido, simples
        ❌ Limitações: Pode gerar palavras inexistentes, menos preciso
        """
        stemmed = [self.stemmer.stem(token) for token in tokens]
        print(f"   🌱 Stemming aplicado:")
        for original, stemmed_word in zip(tokens, stemmed):
            if original != stemmed_word:
                print(f"      {original} → {stemmed_word}")
        return stemmed
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        📚 ETAPA 3B: Lemmatização (Alternativa 2 - Recomendada)
        
        Lemmatização reduz palavras à sua forma canônica (lemma) usando
        conhecimento linguístico e dicionários.
        
        Exemplos:
        - running, runs, ran → run
        - better → good
        - mice → mouse
        
        ✅ Vantagens: Mais preciso, gera palavras reais
        ❌ Limitações: Mais lento, requer recursos linguísticos
        """
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        print(f"   📚 Lemmatização aplicada:")
        for original, lemma in zip(tokens, lemmatized):
            if original != lemma:
                print(f"      {original} → {lemma}")
        return lemmatized
    
    def preprocess_pipeline(self, text: str) -> List[str]:
        """
        🔄 Pipeline Completo de Pré-processamento
        
        Ordem das operações (importante!):
        1. Limpeza → 2. Tokenização → 3. Stop words → 4. Lemmatização
        
        💡 Dica: A ordem importa! Limpe antes de tokenizar,
        remova stop words antes de lemmatizar.
        """
        print(f"\n🔄 INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        print("=" * 50)
        
        # Etapa 1: Limpeza
        print("\n🧹 ETAPA 1: LIMPEZA")
        clean_text = self.clean_text(text)
        
        # Etapa 2: Tokenização simples
        print("\n✂️ ETAPA 2: TOKENIZAÇÃO")
        tokens = clean_text.split()
        print(f"   📝 Tokens: {tokens}")
        
        # Etapa 3: Remoção de stop words
        print("\n🚫 ETAPA 3: REMOÇÃO DE STOP WORDS")
        tokens = self.remove_stopwords(tokens)
        
        # Etapa 4: Lemmatização
        print("\n📚 ETAPA 4: LEMMATIZAÇÃO")
        tokens = self.lemmatize_tokens(tokens)
        
        print(f"\n✅ RESULTADO FINAL: {tokens}")
        return tokens

# 🚀 EXEMPLO PRÁTICO EDUCACIONAL
def demonstrar_preprocessamento():
    """Demonstração completa do pré-processamento"""
    
    preprocessor = TextPreprocessor()
    
    # Texto com vários desafios
    texto_exemplo = """
    The running dogs are better than cats! 
    They're playing in the beautiful gardens.
    """
    
    print("🎓 DEMONSTRAÇÃO: PRÉ-PROCESSAMENTO DE TEXTO")
    print("=" * 60)
    
    # Executar pipeline completo
    resultado = preprocessor.preprocess_pipeline(texto_exemplo)
    
    print("\n📊 RESUMO DO PROCESSAMENTO:")
    print("=" * 30)
    print(f"📝 Texto original: '{texto_exemplo.strip()}'")
    print(f"✅ Tokens finais: {resultado}")
    print(f"📊 Redução: {len(texto_exemplo.split())} → {len(resultado)} tokens")
    
    # Comparar stemming vs lemmatização
    print("\n🔍 COMPARAÇÃO: STEMMING vs LEMMATIZAÇÃO")
    print("=" * 40)
    
    tokens_exemplo = ['running', 'better', 'playing', 'beautiful']
    
    for token in tokens_exemplo:
        stemmed = preprocessor.stemmer.stem(token)
        lemmatized = preprocessor.lemmatizer.lemmatize(token)
        print(f"{token:10} → Stem: {stemmed:8} | Lemma: {lemmatized}")

# Executar demonstração
demonstrar_preprocessamento()
```

```python
import re
from typing import List

class TextPreprocessorSimplified:
    def __init__(self, language='english'):
        """
        Inicialização com recursos de pré-processamento simplificados
        
        Componentes principais:
        - Stop words: lista básica de palavras comuns
        - Stemming simples: remoção de sufixos básicos
        - Sem dependências externas
        """
        # Lista básica de stop words em inglês
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'they', 'their', 'them', 'this',
            'these', 'those', 'we', 'you', 'your', 'i', 'me', 'my', 'mine',
            'our', 'ours', 'she', 'her', 'hers', 'him', 'his'
        }
        
        # Regras básicas de stemming (sufixos comuns)
        self.stemming_rules = [
            ('ing', ''),      # running → runn
            ('ly', ''),       # quickly → quick
            ('ed', ''),       # played → play
            ('ies', 'y'),     # flies → fly
            ('ied', 'y'),     # tried → try
            ('ies', 'y'),     # studies → study
            ('s', ''),        # dogs → dog
        ]
    
    def clean_text(self, text: str) -> str:
        """
        🧹 ETAPA 1: Limpeza Básica do Texto
        
        Operações realizadas:
        1. Remove caracteres especiais e números
        2. Converte para minúsculas (case normalization)
        3. Remove espaços extras
        
        ✅ Por que fazer isso?
        - Reduz ruído nos dados
        - Padroniza formato
        - Melhora consistência dos embeddings
        """
        print(f"   📝 Texto original: '{text}'")
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        print(f"   🔧 Após remoção de especiais: '{text}'")
        
        # Converter para minúsculas
        text = text.lower()
        print(f"   📝 Após minúsculas: '{text}'")
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        print(f"   ✨ Texto limpo: '{text}'")
        
        return text
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        🚫 ETAPA 2: Remoção de Stop Words
        
        Stop words são palavras muito comuns que geralmente não carregam
        significado semântico importante: 'the', 'and', 'is', 'in', etc.
        
        ✅ Vantagens:
        - Reduz dimensionalidade
        - Foca em palavras com mais significado
        - Melhora eficiência computacional
        
        ❌ Cuidado:
        - Pode remover contexto importante em algumas tarefas
        - "Not good" → "good" (perde negação)
        """
        original_count = len(tokens)
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        removed_count = original_count - len(filtered_tokens)
        
        print(f"   🚫 Removidas {removed_count} stop words de {original_count} tokens")
        print(f"   📋 Tokens restantes: {filtered_tokens}")
        
        return filtered_tokens
    
    def simple_stem(self, word: str) -> str:
        """
        🌱 Stemming Simples com Regras Básicas
        
        Aplica regras simples de remoção de sufixos.
        Não é tão preciso quanto Porter Stemmer, mas funciona sem dependências.
        
        Exemplos:
        - running → runn
        - quickly → quick
        - played → play
        """
        for suffix, replacement in self.stemming_rules:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)] + replacement
        return word
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        🌱 ETAPA 3A: Stemming Simples (Alternativa 1)
        
        Stemming remove sufixos para encontrar o "radical" da palavra.
        Versão simplificada com regras básicas.
        
        ✅ Vantagens: Rápido, sem dependências externas
        ❌ Limitações: Menos preciso que algoritmos avançados
        """
        stemmed = [self.simple_stem(token) for token in tokens]
        print(f"   🌱 Stemming simples aplicado:")
        for original, stemmed_word in zip(tokens, stemmed):
            if original != stemmed_word:
                print(f"      {original} → {stemmed_word}")
        return stemmed
    
    def simple_lemmatize(self, word: str) -> str:
        """
        📚 Lemmatização Simples com Dicionário Básico
        
        Versão simplificada usando um pequeno dicionário de formas irregulares.
        """
        # Dicionário básico de formas irregulares comuns
        irregular_forms = {
            'better': 'good',
            'best': 'good',
            'worse': 'bad',
            'worst': 'bad',
            'mice': 'mouse',
            'children': 'child',
            'feet': 'foot',
            'teeth': 'tooth',
            'men': 'man',
            'women': 'woman',
            'running': 'run',
            'ran': 'run',
            'swimming': 'swim',
            'swam': 'swim',
            'flying': 'fly',
            'flew': 'fly'
        }
        
        return irregular_forms.get(word, word)
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        📚 ETAPA 3B: Lemmatização Simples (Alternativa 2 - Recomendada)
        
        Lemmatização reduz palavras à sua forma canônica usando
        um dicionário básico de formas irregulares.
        
        ✅ Vantagens: Mais preciso que stemming simples
        ❌ Limitações: Dicionário limitado, menos abrangente
        """
        lemmatized = [self.simple_lemmatize(token) for token in tokens]
        print(f"   📚 Lemmatização simples aplicada:")
        for original, lemma in zip(tokens, lemmatized):
            if original != lemma:
                print(f"      {original} → {lemma}")
        return lemmatized
    
    def preprocess_pipeline(self, text: str) -> List[str]:
        """
        🔄 Pipeline Completo de Pré-processamento Simplificado
        
        Ordem das operações (importante!):
        1. Limpeza → 2. Tokenização → 3. Stop words → 4. Lemmatização
        
        💡 Dica: A ordem importa! Limpe antes de tokenizar,
        remova stop words antes de lemmatizar.
        """
        print(f"\n🔄 INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO SIMPLIFICADO")
        print("=" * 60)
        
        # Etapa 1: Limpeza
        print("\n🧹 ETAPA 1: LIMPEZA")
        clean_text = self.clean_text(text)
        
        # Etapa 2: Tokenização simples
        print("\n✂️ ETAPA 2: TOKENIZAÇÃO")
        tokens = clean_text.split()
        print(f"   📝 Tokens: {tokens}")
        
        # Etapa 3: Remoção de stop words
        print("\n🚫 ETAPA 3: REMOÇÃO DE STOP WORDS")
        tokens = self.remove_stopwords(tokens)
        
        # Etapa 4: Lemmatização simples
        print("\n📚 ETAPA 4: LEMMATIZAÇÃO SIMPLES")
        tokens = self.lemmatize_tokens(tokens)
        
        print(f"\n✅ RESULTADO FINAL: {tokens}")
        return tokens

# 🚀 EXEMPLO PRÁTICO EDUCACIONAL
def demonstrar_preprocessamento_simplificado():
    """Demonstração completa do pré-processamento sem NLTK"""
    
    preprocessor = TextPreprocessorSimplified()
    
    # Texto com vários desafios
    texto_exemplo = """
    The running dogs are better than cats! 
    They're playing in the beautiful gardens.
    """
    
    print("🎓 DEMONSTRAÇÃO: PRÉ-PROCESSAMENTO DE TEXTO (VERSÃO SIMPLIFICADA)")
    print("=" * 70)
    
    # Executar pipeline completo
    resultado = preprocessor.preprocess_pipeline(texto_exemplo)
    
    print("\n📊 RESUMO DO PROCESSAMENTO:")
    print("=" * 30)
    print(f"📝 Texto original: '{texto_exemplo.strip()}'")
    print(f"✅ Tokens finais: {resultado}")
    print(f"📊 Redução: {len(texto_exemplo.split())} → {len(resultado)} tokens")
    
    # Comparar stemming vs lemmatização simples
    print("\n🔍 COMPARAÇÃO: STEMMING vs LEMMATIZAÇÃO SIMPLES")
    print("=" * 50)
    
    tokens_exemplo = ['running', 'better', 'playing', 'beautiful', 'quickly']
    
    for token in tokens_exemplo:
        stemmed = preprocessor.simple_stem(token)
        lemmatized = preprocessor.simple_lemmatize(token)
        print(f"{token:10} → Stem: {stemmed:8} | Lemma: {lemmatized}")

# Executar demonstração
demonstrar_preprocessamento_simplificado()
```

### 🎯 **Pontos-Chave para Fixação**

1. **Tokenização é fundamental**: É o primeiro passo para converter texto em dados processáveis
2. **Diferentes métodos, diferentes propósitos**: Simples para prototipagem, NLTK para análise geral, Transformers para modelos modernos
3. **Pré-processamento melhora qualidade**: Texto limpo gera embeddings mais consistentes
4. **Ordem importa**: Sempre siga a sequência lógica de processamento
5. **Trade-offs**: Mais processamento = mais lento, mas geralmente melhor qualidade

### 💡 **Dicas Práticas**

- **Para embeddings**: Use tokenização compatível com o modelo escolhido
- **Para análise exploratória**: NLTK é uma boa escolha
- **Para produção**: Considere performance vs. qualidade
- **Sempre valide**: Inspecione os resultados de cada etapa
