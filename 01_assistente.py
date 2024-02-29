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
#print(hour)

date = datetime.date.today().strftime('%d/%B/%Y')
date = date.split('/')
#print(date)

sns.set()