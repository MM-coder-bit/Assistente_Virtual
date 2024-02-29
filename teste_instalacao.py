# Importa a função playsound da biblioteca playsound para reprodução de arquivos de som.
from playsound import playsound

# Define os caminhos para três arquivos de som (n1, n2, n3).
n1 = 'Help\\curso_assistente-20240226T221933Z-001\\curso_assistente\\n1.mp3'
n2 = 'Help\\curso_assistente-20240226T221933Z-001\\curso_assistente\\n2.mp3'
n3 = 'Help\\curso_assistente-20240226T221933Z-001\\curso_assistente\\n3.mp3'

# Reproduz cada arquivo de som usando a função playsound.
playsound(n1)
playsound(n2)
playsound(n3)

# Importa a biblioteca speech_recognition e imprime a versão.
import speech_recognition
print('Speech_Recognition: ', speech_recognition.__version__)

# Importa a biblioteca pyttsx3 para síntese de voz e fala uma frase de teste.
import pyttsx3
pyttsx3.speak('Testando a Biblioteca')

# Importa a biblioteca tensorflow e imprime a versão.
import tensorflow
print('TensorFlow: ', tensorflow.__version__)

# Importa a biblioteca librosa para análise de áudio e imprime a versão.
import librosa
print('Librosa:', librosa.__version__)

# Importa a biblioteca matplotlib para criação de gráficos e imprime a versão.
import matplotlib
print('Matplotlib:', matplotlib.__version__)

# Importa a biblioteca seaborn para visualização de dados e imprime a versão.
import seaborn
print('Seaborn:', seaborn.__version__)

# Importa a biblioteca pyaudio para lidar com entrada e saída de áudio e imprime a versão.
import pyaudio
print('pyaudio:', pyaudio.__version__)
