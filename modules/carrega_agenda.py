# Importa as bibliotecas necessárias.
import datetime
import pandas as pd

# Obtém a hora atual.
hora_atual = datetime.datetime.now()

# Obtém a hora e o minuto da hora atual.
hora_atual, minuto_atual = datetime.datetime.time(hora_atual).hour, datetime.datetime.time(hora_atual).minute

# Obtém a data atual.
data_atual = datetime.datetime.date(datetime.datetime.today())

# Define o caminho para a planilha de agenda.
planilha_agenda = 'Agenda\\agenda.xlsx'

# Lê a planilha de agenda utilizando a biblioteca pandas.
agenda = pd.read_excel(planilha_agenda)

# Inicializa listas vazias para armazenar dados da agenda.
descricao, responsavel, hora_agenda = [], [], []

# Itera sobre as linhas da agenda.
for index, row in agenda.iterrows():
    data = datetime.datetime.date(row['data'])
    hora_completa = datetime.datetime.strptime(str(row['hora']), '%H:%M:%S')
    hora = datetime.datetime.time(hora_completa).hour

    # Verifica se a data na agenda é igual à data atual e se a hora é maior ou igual à hora atual.
    if data_atual == data:
        if hora >= hora_atual:
            descricao.append(row['descricao'])
            responsavel.append(row['responsavel'])
            hora_agenda.append(row['hora'])

# Função que carrega os dados da agenda.
def carrega_agenda():
    if descricao:
        return descricao, responsavel, hora_agenda
    else:
        return False
    
#print(descricao)
#print(responsavel)
#print(hora_agenda)
