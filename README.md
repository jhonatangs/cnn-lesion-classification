# CNN Lesion Classification

Projeto desenvolvido como trabalho da disciplina BCC406 - Redes Neurais e Aprendizagem em Profundidade.

Esse trabalho propõe a aplicação de aprendizado profundo com redes neurais convolucionais na classificação automática de lesões de pele.

Foram treinadas e utilizadas para classificar as imagens da terceira fase da competição [ISIC 2017](https://challenge.isic-archive.com/landing/2017) as arquiteturas ResNet50 e ResNet152V2, a primeira treinada do zero e a segunda através de fine tuning.

Parte do código utilizado foi retirado de [cnn-libras](https://github.com/lucaaslb/cnn-libras).

## Como usar

Instalando as dependências:

```python
pip install -r requirements.txt
```

Executando a ResNet50:

```python
python main/train.py
```

Executando a ResNet152V2:

```python
python main/transfer_learning.py
```