# -*- coding: utf-8 -*-
"""
Projeto: Classificador de Imagens CIFAR-10
Fase 1: Carregamento e Análise Exploratória de Dados (EDA)
"""

# 1. Importar as bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

print("Bibliotecas importadas com sucesso!")

# 2. Carregar o dataset CIFAR-10
# Os dados são automaticamente divididos em treinamento e teste
(X_treino, y_treino), (X_teste, y_teste) = cifar10.load_data()

print("\nDataset CIFAR-10 carregado com sucesso!")
print(f"Formato dos dados de treinamento (imagens): {X_treino.shape}")
print(f"Formato dos rótulos de treinamento: {y_treino.shape}")
print(f"Formato dos dados de teste (imagens): {X_teste.shape}")
print(f"Formato dos rótulos de teste: {y_teste.shape}")

# 3. Entender as classes (categorias)
# O dataset CIFAR-10 tem 10 classes. Vamos definir seus nomes para facilitar a visualização.
nomes_classes = [
    'avião', 'automóvel', 'pássaro', 'gato', 'cervo',
    'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão'
]
print("\nNomes das classes: ", nomes_classes)

# 4. Visualizar algumas imagens de exemplo
print("\nVisualizando algumas imagens de exemplo...")

# Cria uma figura com uma grade de 5x5 imagens
plt.figure(figsize=(10, 10))
for i in range(25):  # Mostra as primeiras 25 imagens
    plt.subplot(5, 5, i + 1)  # Define a posição da imagem na grade
    plt.xticks([])  # Remove os ticks do eixo x
    plt.yticks([])  # Remove os ticks do eixo y
    plt.grid(False)  # Remove a grade
    plt.imshow(X_treino[i])  # Mostra a imagem

    # Obtém o nome da classe usando o rótulo (y_treino[i][0])
    # y_treino é um array 2D, então precisamos [0] para pegar o valor escalar
    plt.xlabel(nomes_classes[y_treino[i][0]])
plt.suptitle("Primeiras 25 imagens do CIFAR-10", fontsize=16)
plt.show()

# 5. Verificar a distribuição das classes (Opcional, mas útil)
# Isso mostra quantas imagens temos de cada categoria
print("\nContagem de imagens por classe no conjunto de treinamento:")
unique, counts = np.unique(y_treino, return_counts=True)
for i, count in zip(unique, counts):
    print(f"  {nomes_classes[i]}: {count} imagens")


# --- FASE 2: PRÉ-PROCESSAMENTO E MODELO DE LINHA DE BASE ---

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time # Para marcar o tempo

print("\n--- Iniciando Fase 2: Modelo de Linha de Base (Random Forest) ---")

# 1. Normalização dos pixels
# Dividimos por 255.0 para garantir que o resultado seja float
X_treino_norm = X_treino / 255.0
X_teste_norm = X_teste / 255.0
print("Dados normalizados (valores de pixel agora entre 0 e 1).")

# 2. "Achatamento" (Flattening) das imagens
# Transformar (50000, 32, 32, 3) em (50000, 3072)
n_amostras_treino = X_treino_norm.shape[0]
n_amostras_teste = X_teste_norm.shape[0]

# Usamos .reshape() para achatar. -1 significa "calcule o tamanho automaticamente"
X_treino_flat = X_treino_norm.reshape(n_amostras_treino, -1)
X_teste_flat = X_teste_norm.reshape(n_amostras_teste, -1)

# O Scikit-learn prefere rótulos (y) como um vetor 1D (ex: [1, 2, 3])
# e não como um vetor coluna (ex: [[1], [2], [3]]). Usamos .ravel() para isso.
y_treino_flat = y_treino.ravel()
y_teste_flat = y_teste.ravel()

print(f"Formato dos dados de treino 'achatados': {X_treino_flat.shape}")
print(f"Formato dos rótulos de treino 'achatados': {y_treino_flat.shape}")

# 3. Usar um SUB-CONJUNTO para treinar mais rápido
# Vamos usar apenas as primeiras 10000 amostras de treino e 2000 de teste
N_AMOSTRAS_TREINO_RAPIDO = 10000
N_AMOSTRAS_TESTE_RAPIDO = 2000

X_treino_rapido = X_treino_flat[:N_AMOSTRAS_TREINO_RAPIDO]
y_treino_rapido = y_treino_flat[:N_AMOSTRAS_TREINO_RAPIDO]
X_teste_rapido = X_teste_flat[:N_AMOSTRAS_TESTE_RAPIDO]
y_teste_rapido = y_teste_flat[:N_AMOSTRAS_TESTE_RAPIDO]

print(f"\nUsando {N_AMOSTRAS_TREINO_RAPIDO} amostras para treino e {N_AMOSTRAS_TESTE_RAPIDO} para teste.")

# 4. Criar e Treinar o modelo Random Forest
print("Iniciando treinamento do Random Forest (isso pode levar alguns minutos)...")
start_time = time.time() # Marcar o início

# n_estimators=100 significa que ele usará 100 "árvores de decisão"
# n_jobs=-1 usa todos os processadores do seu PC para acelerar
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

# Treinando o modelo
rf_model.fit(X_treino_rapido, y_treino_rapido)

end_time = time.time() # Marcar o fim
print(f"Treinamento concluído em {end_time - start_time:.2f} segundos.")

# 5. Avaliar o modelo
print("Avaliando o modelo...")
y_predito = rf_model.predict(X_teste_rapido)

acuracia = accuracy_score(y_teste_rapido, y_predito)

print(f"\n--- Resultado da Linha de Base (Random Forest) ---")
print(f"Acurácia no conjunto de teste (com {N_AMOSTRAS_TESTE_RAPIDO} amostras): {acuracia * 100:.2f}%")
print("--------------------------------------------------")

# --- FASE 3: REDE NEURAL CONVOLUCIONAL (CNN) ---

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

print("\n--- Iniciando Fase 3: Rede Neural Convolucional (CNN) ---")

# 1. Preparar os Rótulos (y) para a CNN
# Vamos usar o formato 'One-Hot Encoding'
# Ex: O rótulo '3' (gato) vira [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
num_classes = 10
y_treino_cat = to_categorical(y_treino, num_classes)
y_teste_cat = to_categorical(y_teste, num_classes)

print("Rótulos (y) convertidos para formato One-Hot.")

# 2. Construir o modelo da CNN
# Usamos o Sequential, que é como empilhar blocos de LEGO
cnn_model = Sequential()

# --- Bloco 1: Convolução + Pooling ---
# O Detetive: 32 lupas (filtros), de tamanho 3x3
# A Ativação: 'relu' (o "porteiro" que liga a luz)
# input_shape: Apenas na primeira camada, dizemos o formato da imagem
cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     input_shape=(32, 32, 3)))
# O Resumo: Pega o maior valor em janelas 2x2
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

# --- Bloco 2: Convolução + Pooling ---
# Mais um conjunto de detetives para aprender padrões mais complexos
cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

# --- Preparação para o Júri Final ---
# Achata o mapa de características 2D em um vetor 1D
cnn_model.add(Flatten())

# --- O Júri Final (Camadas Totalmente Conectadas) ---
# Uma camada "escondida" de 64 neurônios para ajudar a pensar
cnn_model.add(Dense(units=64, activation='relu'))

# A camada de SAÍDA: 10 neurônios (um para cada classe)
# 'softmax' decide qual classe tem a maior probabilidade (dá o veredito final)
cnn_model.add(Dense(units=num_classes, activation='softmax'))

# 3. Compilar o modelo
# Aqui definimos como o modelo vai aprender
cnn_model.compile(
    optimizer='adam', # Um otimizador popular e eficiente
    loss='categorical_crossentropy', # Função de perda boa para classificação
    metrics=['accuracy'] # Queremos monitorar a acurácia
)

# Mostra um resumo da arquitetura do nosso modelo
print("\nArquitetura da CNN construída:")
cnn_model.summary()

# 4. Treinar a CNN
print("\nIniciando treinamento da CNN (isso VAI levar vários minutos)...")
start_time = time.time() # Marcar o início

# Vamos treinar com 10 "épocas" (epochs)
# Epochs = Quantas vezes o modelo verá o dataset de treino inteiro
# batch_size = Quantas imagens ele olha antes de atualizar os parâmetros
# validation_data = Usamos os dados de teste para ver o desempenho a cada época
cnn_model.fit(
    X_treino_norm, y_treino_cat, # Usamos os dados normalizados (NÃO achatados)
    epochs=10,
    batch_size=64,
    validation_data=(X_teste_norm, y_teste_cat) # Dados de teste para validar
)

end_time = time.time() # Marcar o fim
print(f"Treinamento da CNN concluído em {end_time - start_time:.2f} segundos.")

# 5. Avaliar o modelo CNN
print("\nAvaliando o modelo CNN no conjunto de teste...")
# model.evaluate retorna [perda, acurácia]
resultado = cnn_model.evaluate(X_teste_norm, y_teste_cat)
perda_final = resultado[0]
acuracia_final = resultado[1]

print(f"\n--- Resultado Final (CNN) ---")
print(f"Acurácia final da CNN no teste: {acuracia_final * 100:.2f}%")
print(f"Linha de Base (Random Forest): 43.40%") # Seu resultado anterior
print("-------------------------------")
if acuracia_final * 100 > 43.40:
    print("Sucesso! A CNN superou o modelo Random Forest.")
else:
    print("A CNN não superou o Random Forest (talvez precise de mais épocas ou ajustes).")

# --- FASE 4: OTIMIZAÇÃO DA CNN (V2) ---

print("\n--- Iniciando Fase 4: Otimização da CNN (Modelo V2) ---")

# 1. Construir o modelo V2 (Mais profundo e com Dropout)
cnn_model_v2 = Sequential()

# --- Bloco 1 ---
cnn_model_v2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn_model_v2.add(MaxPooling2D(pool_size=(2, 2)))

# --- Bloco 2 ---
cnn_model_v2.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model_v2.add(MaxPooling2D(pool_size=(2, 2)))

# --- NOVO BLOCO 3 ---
# Deixando a rede mais profunda para aprender padrões mais complexos
cnn_model_v2.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model_v2.add(MaxPooling2D(pool_size=(2, 2)))

# --- Preparação para o Júri Final ---
cnn_model_v2.add(Flatten())

# --- O Júri Final (com Dropout) ---
cnn_model_v2.add(Dense(64, activation='relu'))

# NOVO: Camada de Dropout.
# "Desliga" 50% dos neurônios aleatoriamente durante o treino
cnn_model_v2.add(Dropout(0.5))

# Camada de Saída (igual a antes)
cnn_model_v2.add(Dense(num_classes, activation='softmax'))

# 2. Compilar o modelo V2
cnn_model_v2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nArquitetura da CNN V2 (mais profunda):")
cnn_model_v2.summary()

# 3. Treinar a CNN V2
print("\nIniciando treinamento da CNN V2 (isso pode ser ainda MAIS demorado)...")
start_time = time.time()

# Vamos treinar por 15 épocas para dar mais tempo para aprender
history = cnn_model_v2.fit(
    X_treino_norm, y_treino_cat,
    epochs=15, # Aumentamos de 10 para 15
    batch_size=64,
    validation_data=(X_teste_norm, y_teste_cat)
)

end_time = time.time()
print(f"Treinamento da CNN V2 concluído em {end_time - start_time:.2f} segundos.")

# 4. Avaliar o modelo V2
print("\nAvaliando o modelo CNN V2 no conjunto de teste...")
resultado_v2 = cnn_model_v2.evaluate(X_teste_norm, y_teste_cat)
acuracia_v2 = resultado_v2[1]

print(f"\n--- Resultado Final (CNN V2) ---")
print(f"Acurácia final da CNN V2: {acuracia_v2 * 100:.2f}%")
print(f"Acurácia anterior (V1): 69.13%")
print("--------------------------------")

# --- FASE 5: ANÁLISE GRÁFICA DO TREINAMENTO V2 ---
import matplotlib.pyplot as plt

print("\n--- Iniciando Fase 5: Análise Gráfica (Plots) ---")

# 'history' guarda os dados do treinamento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, 16) # 15 épocas

# --- Gráfico 1: Acurácia ao longo das Épocas ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) # (1 linha, 2 colunas, gráfico 1)
plt.plot(epochs_range, acc, label='Acurácia de Treino')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação (Teste)')
plt.legend(loc='lower right')
plt.title('Acurácia de Treino vs. Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')

# --- Gráfico 2: Perda (Erro) ao longo das Épocas ---
plt.subplot(1, 2, 2) # (1 linha, 2 colunas, gráfico 2)
plt.plot(epochs_range, loss, label='Perda de Treino')
plt.plot(epochs_range, val_loss, label='Perda de Validação (Teste)')
plt.legend(loc='upper right')
plt.title('Perda de Treino vs. Validação')
plt.xlabel('Época')
plt.ylabel('Perda (Loss)')

plt.suptitle("Análise Gráfica do Treinamento da CNN V2")
plt.show()

# --- FASE 6: REFINANDO O MODELO V1 (O VENCEDOR) COM EARLY STOPPING ---
from tensorflow.keras.callbacks import EarlyStopping

print("\n--- Iniciando Fase 6: Modelo V3 (V1 + Early Stopping) ---")

# 1. Definir o "Callback" de Early Stopping
# Monitor = o que ele vai observar ('val_accuracy')
# Patience = 3. Significa: "Espere 3 épocas. Se a val_accuracy não melhorar, pare."
# Restore_best_weights = True. Pega o modelo da melhor época (não da última).
early_stopper = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1 # Mostra uma mensagem quando parar
)

# 2. Reconstruir o modelo V1 (nosso melhor modelo)
# (Precisamos recriar, pois o cnn_model já foi treinado)
cnn_model_v3 = Sequential()
cnn_model_v3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn_model_v3.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model_v3.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model_v3.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model_v3.add(Flatten())
cnn_model_v3.add(Dense(64, activation='relu'))
cnn_model_v3.add(Dense(num_classes, activation='softmax'))

# 3. Compilar o modelo V3
cnn_model_v3.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nArquitetura da CNN V3 (igual à V1):")
cnn_model_v3.summary()

# 4. Treinar a CNN V3 com o Early Stopping
print("\nIniciando treinamento da CNN V3 (com Early Stopping)...")
start_time = time.time()

# Vamos colocar 20 épocas, mas deixar o Early Stopping decidir quando parar!
history_v3 = cnn_model_v3.fit(
    X_treino_norm, y_treino_cat,
    epochs=20, # Máximo de 20 épocas
    batch_size=64,
    validation_data=(X_teste_norm, y_teste_cat),
    callbacks=[early_stopper] # AQUI ESTÁ A MÁGICA!
)

end_time = time.time()
print(f"Treinamento da CNN V3 concluído em {end_time - start_time:.2f} segundos.")

# 5. Avaliar o modelo V3
print("\nAvaliando o modelo CNN V3 no conjunto de teste...")
resultado_v3 = cnn_model_v3.evaluate(X_teste_norm, y_teste_cat)
acuracia_v3 = resultado_v3[1]

print(f"\n--- Resultado Final (CNN V3) ---")
print(f"Acurácia final da CNN V3 (Early Stop): {acuracia_v3 * 100:.2f}%")
print(f"Acurácia anterior (V1): 69.13%")
print(f"Acurácia anterior (V2): 68.09%")
print("--------------------------------")

# --- FASE 7: VISUALIZAÇÃO AVANÇADA DO MODELO ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np  # Já devemos ter, mas só para garantir

print("\n--- Iniciando Fase 7: Visualização Avançada ---")

# --- 7.1: GRÁFICOS DE TREINO (MODELO V3) ---
# Vamos verificar como foi o treino do V3 (o que parou cedo)

# 'history_v3' foi salvo na Fase 6
acc_v3 = history_v3.history['accuracy']
val_acc_v3 = history_v3.history['val_accuracy']
loss_v3 = history_v3.history['loss']
val_loss_v3 = history_v3.history['val_loss']

# Descobre quantas épocas realmente rodaram
epochs_ran_v3 = len(acc_v3)
epochs_range_v3 = range(1, epochs_ran_v3 + 1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Gráfico de Acurácia
plt.plot(epochs_range_v3, acc_v3, label='Acurácia de Treino')
plt.plot(epochs_range_v3, val_acc_v3, label='Acurácia de Validação (Teste)')
plt.legend(loc='lower right')
plt.title('Acurácia do Modelo V3 (com Early Stop)')
plt.xlabel('Época')
plt.ylabel('Acurácia')

plt.subplot(1, 2, 2)  # Gráfico de Perda
plt.plot(epochs_range_v3, loss_v3, label='Perda de Treino')
plt.plot(epochs_range_v3, val_loss_v3, label='Perda de Validação (Teste)')
plt.legend(loc='upper right')
plt.title('Perda do Modelo V3 (com Early Stop)')
plt.xlabel('Época')
plt.ylabel('Perda (Loss)')

plt.suptitle("Análise Gráfica do Treinamento da CNN V3")
plt.show()  # Mostra os gráficos do V3

# --- 7.2: MATRIZ DE CONFUSÃO ---
# O que o modelo está confundindo?

print("Calculando predições para a Matriz de Confusão...")
# Pega as probabilidades que o modelo previu para o conjunto de teste
y_pred_probs = cnn_model_v3.predict(X_teste_norm)

# Converte as probabilidades na classe com maior valor (o palpite final)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Pega os rótulos verdadeiros (não-categóricos, ex: 0, 1, 2...)
# y_teste já foi carregado na Fase 1
y_true_classes = y_teste.ravel()  # Achatamos para garantir o formato (10000,)

# Calcula a matriz
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Desenha a matriz com Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,  # Mostrar os números dentro de cada célula
    fmt='d',  # Formatar os números como inteiros
    xticklabels=nomes_classes,  # Nomes no eixo X (Fase 1)
    yticklabels=nomes_classes  # Nomes no eixo Y (Fase 1)
)
plt.title('Matriz de Confusão - Modelo V3 (71.13%)')
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Prevista')
plt.show()

# --- 7.3: EXEMPLOS DE ERROS ---
print("Buscando exemplos de classificações erradas...")

# Encontra os índices (posições) onde a predição foi DIFERENTE do rótulo verdadeiro
misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]

# Pega 25 exemplos aleatórios desses erros
try:
    # Tenta pegar 25, mas se houver menos, pega o que tiver
    size_to_show = min(25, len(misclassified_indices))
    random_error_indices = np.random.choice(misclassified_indices, size=size_to_show, replace=False)
except ValueError:
    print("Não foram encontrados erros suficientes para mostrar (ou não há erros!).")
    random_error_indices = misclassified_indices

plt.figure(figsize=(10, 10))
for i, img_index in enumerate(random_error_indices):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # Mostra a imagem original (X_teste, NÃO normalizado)
    plt.imshow(X_teste[img_index])

    # Pega os nomes das classes
    true_label = nomes_classes[y_true_classes[img_index]]
    pred_label = nomes_classes[y_pred_classes[img_index]]

    # Coloca o título em vermelho para destacar o erro
    plt.xlabel(f"Verdadeiro: {true_label}\nPrevisto: {pred_label}", color='red')

plt.suptitle("25 Classificações Erradas (Aleatórias)", fontsize=16, y=1.02)
plt.tight_layout()  # Ajusta o espaçamento
plt.show()

print("\n--- Análise de Visualização Concluída! ---")