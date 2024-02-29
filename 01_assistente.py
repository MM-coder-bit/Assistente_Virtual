import pyttsx3
import speech_recognition as sr
from playsound import playsound
import webbrowser as wb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime

hour = datetime.datetime.now().strftime('%H:%M')
print(hour)

date = datetime.date.today().strftime('%d/%B/%Y')
date = date.split('/')

sns.set()

from modules import carrega_agenda, comandos_respostas
comandos = comandos_respostas.comandos
respostas = comandos_respostas.respostas

meu_nome = 'Agatha'
chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'  # Caminho do Google Chrome no seu PC

def search(frase):
    wb.get(chrome_path).open('https://www.google.com/search?q=' + frase)

search('Linguagem Python')