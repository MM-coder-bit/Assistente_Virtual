# Importa as bibliotecas necessárias.
import pyttsx3  # Biblioteca para síntese de voz
import speech_recognition as sr  # Biblioteca para reconhecimento de voz
from playsound import playsound  # Biblioteca para reprodução de som
import webbrowser as wb  # Biblioteca para interação com o navegador
import tensorflow as tf  # Biblioteca para aprendizado de máquina
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
        SAMPLE_RATE = 48000
    return model, model_dict, SAMPLE_RATE

print(load_model_by_name('EMOCAO'))
print(load_model_by_name('EMOCAO')[0].summary())
