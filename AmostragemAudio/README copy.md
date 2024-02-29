# Amostragem de Áudio

### Um breve  resumo sobre a amostragem de áudio:

- **Sinal Áudio:**  O sinal elétrico é gerado por um dispositivo que capta ou reproduz som, como um microfone ou alto-falante. Amostragem refere-se ao processo de capturar esse sinal, enquanto a redução de amostragem é o procedimento pelo qual se realiza a diminuição da taxa de amostras. Essa redução é um passo significativo no processo de amostragem.

### Imagem representativa do sinal analógico e digital:
<img src="imagens\SinalAnalogicoDigital.png" width="500" height="300">

### Imagem representativa do digital:
<img src="imagens\\SinalDigital.png" width="500" height="300">

### Imagem representativa da amostragem:
<img src="imagens\SinalAnalogicoeAmostras.png" width="800" height="300">

Na imagem apresentada, é possível identificar o sinal analógico, originado de um microfone ou de outro dispositivo de captura, e as amostras correspondentes aos valores do sinal medidos em intervalos regulares. O sinal analógico mencionado pode ser aproximadamente representado pelo vetor de amostras a seguir:

```python
amostras = [0, 10, 15, 9, -5, -9, -10, 0, 10, 14, 15, 5, 0, -5, ...]
```

A **taxa de amostragem** ou **frequência de amostragem** é definida como o número de amostras do sinal analógico selecionadas por segundo.

### Imagem referente sobre a taxa de amostragem:
<img src="imagens\\RelacaoTaxa.png" width="500" height="300">