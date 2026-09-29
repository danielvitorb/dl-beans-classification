# Classificação de Grãos Secos com Deep Learning 🫘

Este repositório contém o desenvolvimento de uma Rede Neural Artificial (Multilayer Perceptron) focada na classificação multiclasse de grãos de feijão seco (*Dry Bean Dataset*) a partir de características morfológicas e geométricas extraídas de imagens. 

O projeto foi desenvolvido como parte da disciplina de **Aprendizado Profundo (Deep Learning)** do bacharelado em Tecnologia da Informação da Universidade Federal do Rio Grande do Norte (UFRN).

## 🎯 O Desafio
O objetivo principal foi construir um classificador robusto lidando com dois problemas centrais comuns em cenários reais:
1. **Desbalanceamento Extremo de Classes:** A classe majoritária possui cerca de 7x mais amostras que a minoritária.
2. **Escalas Distintas:** Atributos variando de pequenos fatores de forma a grandes áreas em pixels.

Além da atividade de modelagem base, o projeto conta com uma abordagem focada em maximização de Acurácia para uma **competição interna** da disciplina.

## 🧠 Arquitetura e Decisões de Engenharia

O modelo foi construído utilizando **PyTorch** e otimizado para execução via Metal Performance Shaders (MPS) em arquitetura Apple Silicon. 

Para garantir estabilidade, regularização e capacidade de generalização, as seguintes técnicas foram aplicadas:

### 1. Estabilidade do Treinamento
* **Input Normalization:** Aplicação de `StandardScaler` para centralizar features na média zero e variância unitária.
* **Batch Normalization:** Utilizado após cada camada densa (antes da ativação) para mitigar o *Internal Covariate Shift* e suavizar a superfície de perda (*Loss Landscape*).
* **Inicialização He (Kaiming):** Pesos inicializados especificamente para a função de ativação não-linear ReLU, prevenindo o desaparecimento de gradientes no início do treinamento.

### 2. Regularização (Combate ao Overfitting)
* **Dropout (20-30%):** Injeção de ruído durante o treino para quebrar a coadaptação dos neurônios, forçando a rede a aprender representações distribuídas.
* **Weight Decay (L2):** Utilização do otimizador `AdamW` com penalização L2 ($1e^{-4}$) para restringir a magnitude dos pesos e gerar fronteiras de decisão mais suaves.
* **Early Stopping:** Monitoramento da função de custo de validação, com paciência configurada para interromper o treinamento precocemente caso o modelo começasse a decorar os dados, restaurando os melhores pesos globais.

### 3. Tratamento de Desbalanceamento (Atividade Principal)
* **Função de Custo Ponderada:** Cálculo de pesos inversamente proporcionais à frequência das classes e aplicação direta na `CrossEntropyLoss`. Isso garantiu que a classe minoritária ('Bombay') tivesse um impacto matemático ~7x maior no cálculo do erro que a majoritária ('Dermason').
* **Métrica de Avaliação:** Substituição da Acurácia Global pelo **Macro F1-Score**, garantindo uma avaliação imparcial. O modelo base atingiu **0.94 de Macro F1-Score**.

### 4. Estratégia de Competição (Ensemble Learning)
Para a submissão no Kaggle/Competição focada estritamente em **Acurácia**, a estratégia foi readaptada:
* Remoção dos pesos na função de custo para otimizar o acerto global.
* Implementação de um **Conselho de Sábios (Ensemble de 10 Redes):** Treinamento de 10 modelos idênticos com diferentes sementes de inicialização. As previsões finais foram geradas através da média das probabilidades (*Soft Voting*), resultando em predições consideravelmente mais estáveis e precisas no conjunto de teste "cego".

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python 3
* **Deep Learning:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Análise e Manipulação de Dados:** Pandas, NumPy
* **Machine Learning Clássico:** Scikit-Learn (`StandardScaler`, `LabelEncoder`, `train_test_split`, `compute_class_weight`)
* **Visualização:** Matplotlib, Seaborn (Análise de Matriz de Confusão e Curvas de Aprendizado)

## 📁 Estrutura do Repositório
```text
📦 dl-beans-classification
 ┣ 📜 beans_train.csv                      # Dataset de treinamento (com rótulos)
 ┣ 📜 beans_test.csv                       # Dataset da competição (sem rótulos)          
 ┣ 📜 atividade_classificacao_graos.ipynb  # Pipeline completo de treino, validação e análise
 ┣ 📜 beans_submission.csv                 # Arquivo final de predições da competição gerado via Ensemble
 ┗ 📜 README.md