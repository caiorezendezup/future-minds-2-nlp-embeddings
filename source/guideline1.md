## 1. Recap sobre Embeddings

### 1.1 Conceitos Fundamentais
Embeddings são representações numéricas densas que capturam o significado semântico de palavras, frases ou documentos em um espaço vetorial de dimensão fixa. Eles resolvem limitações das representações tradicionais como one-hot encoding.

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

print("=== COMPARAÇÃO DE REPRESENTAÇÕES ===")
print(f"One-hot 'gato': {gato_onehot}")
print(f"Embedding 'gato': {gato_embedding}")
print(f"\nTamanho do vocabulário: {len(vocabulario)}")
print(f"Dimensões one-hot: {len(gato_onehot)} (esparso: {gato_onehot.count(0)}/{len(gato_onehot)} zeros)")
print(f"Dimensões embedding: {len(gato_embedding)} (denso: todos valores não-zero)")
```

**Principais diferenças:**
- **One-hot encoding**: Vetores esparsos com apenas um 1 e muitos 0s
- **Embeddings**: Vetores densos com valores reais que capturam relações semânticas
- **Dimensionalidade**: One-hot cresce com o vocabulário; embeddings têm tamanho fixo

### 1.2 Vantagens dos Embeddings

Os embeddings permitem que **palavras semanticamente similares tenham representações próximas** no espaço vetorial, algo impossível com one-hot encoding.

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

print("=== ANÁLISE DE SIMILARIDADE SEMÂNTICA ===")
print(f"Similaridade gato ↔ felino: {sim_gato_felino:.3f}")
print(f"Similaridade gato ↔ cachorro: {sim_gato_cachorro:.3f}")
print(f"Similaridade gato ↔ carro: {sim_gato_carro:.3f}")

# Interpretação dos resultados
print("\n=== INTERPRETAÇÃO ===")
if sim_gato_felino > sim_gato_cachorro:
    print("✓ 'Gato' é mais similar a 'felino' (mesmo conceito)")
if sim_gato_cachorro > sim_gato_carro:
    print("✓ 'Gato' é mais similar a 'cachorro' (ambos animais) que a 'carro'")

# Demonstração com one-hot (para comparação)
print("\n=== COMPARAÇÃO: ONE-HOT vs EMBEDDINGS ===")
onehot_gato = [1, 0, 0, 0]
onehot_felino = [0, 0, 0, 1]  # Posição diferente no vocabulário
onehot_similarity = cosine_similarity([onehot_gato], [onehot_felino])[0][0]
print(f"Similaridade one-hot gato ↔ felino: {onehot_similarity:.3f}")
print("❌ One-hot não captura que 'gato' e 'felino' são relacionados!")
```

### 1.3 Propriedades Matemáticas dos Embeddings

```python
# Demonstração de propriedades matemáticas
print("=== PROPRIEDADES MATEMÁTICAS ===")

def calculate_vector_properties(name, vector):
    norm = np.linalg.norm(vector)
    print(f"{name}:")
    print(f"  Vetor: {vector}")
    print(f"  Norma (magnitude): {norm:.3f}")
    print(f"  Normalizado: {vector/norm}")
    print()

for name, embedding in embeddings.items():
    calculate_vector_properties(name, embedding)

# Operações vetoriais
print("=== OPERAÇÕES VETORIAIS ===")
# Analogia: gato está para felino assim como cachorro está para...?
analogy_vector = embeddings["felino"] - embeddings["gato"] + embeddings["cachorro"]
print(f"Vetor analogia (felino - gato + cachorro): {analogy_vector}")

# Encontrar palavra mais próxima da analogia
similarities_analogy = {}
for word, vec in embeddings.items():
    if word != "cachorro":  # Excluir a palavra de entrada
        sim = cosine_similarity([analogy_vector], [vec])[0][0]
        similarities_analogy[word] = sim

closest_word = max(similarities_analogy, key=similarities_analogy.get)
print(f"Palavra mais próxima da analogia: {closest_word} (similaridade: {similarities_analogy[closest_word]:.3f})")
```

### 1.4 Vantagens dos Embeddings - Resumo

#### 🎯 **Principais Vantagens dos Embeddings**

1. **🎯 Capturam similaridade semântica**
   - Palavras com significados similares têm representações próximas no espaço vetorial
   - Exemplo: "gato" e "felino" terão alta similaridade cosseno

2. **📏 Dimensionalidade fixa**
   - Independente do tamanho do vocabulário
   - One-hot: cresce com vocabulário | Embeddings: tamanho fixo (ex: 300D)

3. **🔢 Representação densa**
   - Todos os valores são significativos (não há zeros desnecessários)
   - Uso eficiente do espaço de memória

4. **🧮 Operações matemáticas significativas**
   - Analogias: "rei" - "homem" + "mulher" ≈ "rainha"
   - Aritmética vetorial com significado semântico

5. **🚀 Eficiência computacional**
   - Processamento mais rápido em modelos de Machine Learning
   - Menor dimensionalidade que one-hot para vocabulários grandes

6. **🔄 Transferibilidade entre tarefas**
   - Embeddings pré-treinados podem ser reutilizados
   - Word2Vec, GloVe, FastText servem para múltiplas aplicações

#### 🛠️ **Aplicações Práticas**

- **Sistemas de recomendação**: Encontrar produtos/conteúdos similares
- **Busca semântica**: Buscar por significado, não apenas palavras-chave
- **Tradução automática**: Mapear conceitos entre idiomas
- **Análise de sentimentos**: Capturar nuances emocionais
- **Classificação de textos**: Categorização automática de documentos
- **Chatbots e assistentes virtuais**: Compreensão de intenções do usuário

#### 🔑 **Conceitos-chave para Fixação**

- **Similaridade Cosseno**: Mede o ângulo entre vetores (0 = ortogonais, 1 = idênticos)
- **Espaço Vetorial**: Embeddings vivem em um espaço onde distância = similaridade semântica
- **Densidade**: Cada dimensão contribui para o significado (vs. one-hot com muitos zeros)
- **Aprendizado**: Embeddings são aprendidos automaticamente a partir de grandes corpus de texto
