# 🚀 Classificador de Imagens CIFAR-10 com CNN 🧠

Este projeto é uma jornada completa de construção e otimização de uma Rede Neural Convolucional (CNN) em Python, usando Keras/TensorFlow. O objetivo é classificar corretamente as 10 categorias de imagens do famoso dataset CIFAR-10.

O projeto documenta todo o processo, desde uma linha de base simples até um modelo otimizado com mais de **71% de acurácia**.

---

## 📊 A Jornada da Otimização

Este projeto foi construído em fases, melhorando a acurácia passo a passo:

### 1️⃣ Fase 1: Análise Exploratória (EDA) 🖼️
* Carregamento do dataset CIFAR-10.
* Visualização das 10 classes (avião, gato, cachorro, etc.).
* Pré-processamento e normalização dos pixels das imagens.

### 2️⃣ Fase 2: Linha de Base (Random Forest) 🌳
Para entender a complexidade do problema, um modelo clássico de Machine Learning foi treinado.
* **Modelo:** `RandomForestClassifier` (do Scikit-learn)
* **Acurácia Obtida:** 📉 **~43.40%**
* **Conclusão:** Modelos clássicos não conseguem capturar o contexto espacial das imagens, justificando o uso de CNNs.

### 3️⃣ Fase 3: A Primeira CNN (Modelo V1) 🧠
Construção de uma CNN simples (2 camadas de convolução + 1 camada densa).
* **Modelo:** CNN V1 (10 épocas de treino)
* **Acurácia Obtida:** 📈 **~69.13%**
* **Conclusão:** Um salto enorme! A CNN provou ser muito superior ao entender os padrões visuais.

### 4️⃣ Fase 4: O Diagnóstico (Overfitting) 🩺
Uma tentativa de modelo mais profundo (V2) resultou em uma *pior* acurácia (68.09%). A análise gráfica mostrou um claro **overfitting**: o modelo estava "decorando" os dados de treino em vez de "aprender" a generalizar.

<img width="732" height="360" alt="image" src="https://github.com/user-attachments/assets/898af5e8-a287-42d0-9dda-2260119a9df7" />


### 5️⃣ Fase 5: A Otimização (Modelo V3) ✨
O modelo V1 (nosso melhor) foi re-treinado com uma técnica de regularização chave: **Early Stopping**.
* **Técnica:** `EarlyStopping` (paciência = 3)
* **Resultado:** O modelo parou automaticamente na melhor época (época 14), antes de começar o overfitting.

---

## 🏆 Resultado Final: Modelo V3

O modelo final (V3) alcançou o melhor desempenho, provando a eficácia da análise e otimização de hiperparâmetros.

* **Acurácia Final no Teste:** 🎯 **71.13%**

### Matriz de Confusão
A análise mostra que o modelo é forte na identificação de veículos, mas ainda apresenta alguma confusão entre animais (especialmente `gato` vs. `cachorro`).

<img width="706" height="619" alt="image" src="https://github.com/user-attachments/assets/5eae6a47-2698-4bf6-901d-2d99f1fde3a9" />


---

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **TensorFlow (Keras):** Para construir e treinar as CNNs.
* **Scikit-learn:** Para o modelo Random Forest e a Matriz de Confusão.
* **Matplotlib & Seaborn:** Para visualizar os dados e os resultados.
* **Numpy:** Para manipulação de arrays.

---

## 🚀 Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU-NOME/SEU-REPOSITORIO.git](https://github.com/SEU-NOME/SEU-REPOSITORIO.git)
    cd SEU-REPOSITORIO
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o script:**
    O script rodará todas as fases do projeto, desde a EDA até o treinamento final.
    ```bash
    python cifar10_classifier.py
    ```
