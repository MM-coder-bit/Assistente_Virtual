from playsound import playsound

n1 = 'Help\\curso_assistente-20240226T221933Z-001\\curso_assistente\\n1.mp3'
n2 = 'Help\\curso_assistente-20240226T221933Z-001\\curso_assistente\\n2.mp3'
n3 = 'Help\\curso_assistente-20240226T221933Z-001\\curso_assistente\\n3.mp3'

playsound(n1)
playsound(n2)
playsound(n3)

import speech_recognition
print('Speech_Recognition: ', speech_recognition.__version__)

import pyttsx3
pyttsx3.speak('Testando a Biblioteca')

import tensorflow
print('TensorFlow: ', tensorflow.__version__)

import librosa
print('Librosa:', librosa.__version__)

import matplotlib
print('Matplotlib:', matplotlib.__version__)

import seaborn
print('Seaborn:', seaborn.__version__)

import pyaudio
print('pyaudio:', pyaudio.__version__)
