# Importa as bibliotecas necessárias.
import pyttsx3  # Biblioteca para síntese de voz
import speech_recognition as sr  # Biblioteca para reconhecimento de voz
from playsound import playsound  # Biblioteca para reprodução de som
import webbrowser as wb  # Biblioteca para interação com o navegador
import tensorflow as tf  # Biblioteca para aprendizado de máquina
import librosa  # Biblioteca para processamento de áudio
import numpy as np  # Biblioteca para manipulação de arrays
import matplotlib.pyplot as plt  # Biblioteca para criação de gráficos
import seaborn as sns  # Biblioteca para visualização de dados
import random  # Biblioteca para geração de números aleatórios
import datetime  # Biblioteca para manipulação de data e hora

# Obtém a hora atual e a imprime.
hour = datetime.datetime.now().strftime('%H:%M')

# Obtém a data atual e a imprime.
date = datetime.date.today().strftime('%d/%B/%Y')
date = date.split('/')

# Configurações iniciais para visualização de dados com o Seaborn.
sns.set()

# Importa módulos personalizados para agenda e comandos/respostas.
from modules import carrega_agenda, comandos_respostas
comandos = comandos_respostas.comandos
respostas = comandos_respostas.respostas

# Nome para ser utilizado pela assistente virtual.
meu_nome = 'Agatha'

# Caminho do Google Chrome no seu PC.
chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'

# Função para realizar uma pesquisa no Google.
def search(frase):
    wb.get(chrome_path).open('https://www.google.com/search?q=' + frase)

# Tipos de modelo disponíveis.
MODEL_TYPES = ['EMOCAO']

# Função para carregar um modelo específico com base no tipo fornecido.
def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('models/speech_emotion_recognition.hdf5')
        model_dict = sorted(list(['neutra', 'calma', 'feliz', 'triste', 'nervosa', 'medo', 'nojo', 'surpreso']))
        SAMPLE_RATE = 48000 # Taxa comumente usada para  áudio em computadores.
    return model, model_dict, SAMPLE_RATE

#print(load_model_by_name('EMOCAO'))
#print(load_model_by_name('EMOCAO')[0].summary())

model_type  = "EMOCAO"
loaded_model = load_model_by_name(model_type)

# Função para prever emoções em um arquivo de áudio dividido em partes
def predict_sound(AUDIO, SAMPLE_RATE, plot=True):
    # Lista para armazenar os resultados das previsões
    results = []
    
    # Carregar os dados de áudio usando a biblioteca librosa
    # A função load do librosa carrega os dados de áudio do arquivo especificado em AUDIO.
    # Retorna dois valores: wav_data contendo os dados do áudio e sample_rate representando a taxa de amostragem do áudio.

    wav_data, sample_rate = librosa.load(AUDIO, sr=SAMPLE_RATE)

    # Remover silêncios e obter parte não silenciosa do áudio
    # A função effects.trim do librosa remove regiões silenciosas do sinal de áudio.
    # O parâmetro top_db define o limiar de decibéis para considerar uma região como silenciosa.
    # O resultado inclui um trecho (clip) sem as partes silenciosas e um índice (index) indicando as posições não silenciosas.

    clip, index = librosa.effects.trim(wav_data, top_db=60, frame_length=512, hop_length=64)

    # Dividir o áudio em partes usando a função frame da biblioteca TensorFlow
    # A função signal.frame do TensorFlow divide o sinal de áudio (clip) em frames sobrepostos.
    # Os parâmetros incluem a taxa de amostragem (sample_rate), tamanho do frame, e intervalo entre frames (hop_length).
    # pad_end=True garante que os frames alcancem o final do sinal, e pad_value=0 preenche os valores além do sinal com zeros.

    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate, pad_end=True, pad_value=0)



    # Iterar sobre as partes do áudio
    for i, data in enumerate(splitted_audio_data.numpy()):
        
        # Calcular os coeficientes cepstrais de frequência mel (MFCC)
        # A função feature.mfcc do librosa calcula os coeficientes cepstrais de frequência mel (MFCC) para a série temporal de áudio representada por 'data'.
        # Parâmetros incluem a taxa de amostragem (sr) e o número de coeficientes a serem calculados (n_mfcc).

        mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)

        # Redimensionar e normalizar os recursos MFCC
        # Os coeficientes MFCC são transpostos (T) e, em seguida, a média é calculada ao longo do eixo temporal para obter uma representação agregada.
        # O resultado é redimensionado em uma única linha usando reshape, criando um vetor unidimensional.

        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        # Adicionar uma dimensão para representar o batch (amostras agrupadas)
        # Introduz uma nova dimensão na representação dos recursos MFCC, que é necessária para a entrada do modelo de aprendizado profundo.

        mfccs_scaled_features = mfccs_scaled_features[:, :, np.newaxis]

        # Fazer previsões usando um modelo previamente carregado
        # Utiliza o modelo de aprendizado profundo (loaded_model[0]) para fazer previsões sobre os recursos MFCC redimensionados.
        # O batch_size define o número de amostras processadas simultaneamente durante a inferência.

        predictions = loaded_model[0].predict(mfccs_scaled_features, batch_size=32)

        
        # Se plot for True, exibir um gráfico de barras das previsões
        if plot:
            plt.figure(figsize=(len(splitted_audio_data), 5))
            plt.barh(loaded_model[1], predictions[0])
            plt.tight_layout()
            plt.show()

        # Converter as previsões em rótulos emocionais e armazenar os resultados
        # A função argmax(axis=1) encontra o índice do valor máximo ao longo do eixo 1 (colunas) no array 'predictions'.
        # Isso é feito para determinar a classe predita, que representa a emoção prevista.
        predictions = predictions.argmax(axis=1)

        # Converter os índices para o tipo de dados inteiro e nivelar o array para uma única dimensão.
        # A conversão para int é necessária para indexar o array de rótulos emocionais do modelo carregado (loaded_model[1]).
        predictions = predictions.astype(int).flatten()
        
        # Converter os índices preditos em rótulos emocionais usando o array de rótulos do modelo carregado
        # Utiliza os índices preditos (predictions) para acessar o array de rótulos emocionais (loaded_model[1]).
        # Isso resulta na obtenção dos rótulos emocionais correspondentes às previsões.

        predictions = loaded_model[1][predictions[0]]

        # Indexar os rótulos emocionais usando os índices preditos e armazenar os resultados.
        # O array 'predictions' agora contém as emoções previstas para cada parte do áudio.
        results.append(predictions)


        # Exibir o resultado para cada parte do áudio
        result_str = 'PARTE ' + str(i) + ': ' + str(predictions).upper()
        print(result_str)

    # Contar ocorrências de cada rótulo emocional e imprimir
    count_results = [[results.count(x), x] for x in set(results)]
    print(count_results)

    # Imprimir o rótulo emocional mais frequente
    print(max(count_results))
    return max(count_results)

# Exemplo de uso da função
#predict_sound('Audio/triste.wav', loaded_model[2], plot=False)

# Função para reproduzir música no YouTube com base na emoção fornecida como argumento.
def play_music_youtube(emocao):
    # Inicializa a variável de controle para reprodução como Falsa
    play = False
    
    # Verifica se a emoção é 'triste' ou 'medo'
    if emocao == 'triste' or emocao == 'medo':
        # Abre a URL do YouTube para uma música relacionada à tristeza ou medo
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=k32IPg4dbz0&ab_channel=Amelhorm%C3%BAsicainstrumental')
        # Define a variável de controle para reprodução como Verdadeira
        play = True
    
    # Verifica se a emoção é 'nervosa' ou 'surpreso'
    if emocao == 'nervosa' or emocao == 'surpreso':
        # Abre a URL do YouTube para uma música relacionada à nervosismo ou surpresa
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=pWjmpSD-ph0&ab_channel=CassioToledo')
        # Define a variável de controle para reprodução como Verdadeira
        play = True
    
    # Retorna o status de reprodução (True se uma música foi aberta, False caso contrário)
    return play

#play_music_youtube('triste')
#emocao = predict_sound('Audio/triste.wav', loaded_model[2], plot=False)
#print(emocao)
#play_music_youtube(emocao[1])

