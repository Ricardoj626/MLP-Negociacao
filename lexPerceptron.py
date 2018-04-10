'''
pip install -r requeriments.txt
'''

from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
# from sklearn.externals import joblib

def vetorizar_texto(texto, tradutor):
    vetor = [0] * len(tradutor)

    for palavra in texto:
        if len(palavra) > 0:
            raiz = palavra
            if raiz in tradutor:
                posicao = tradutor[raiz]
                vetor[posicao] += 1
    return vetor

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 5
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

classificacoes = pd.read_csv('lex-carga.csv', encoding='utf-8')
textosPuros = classificacoes['resposta']
textosQuebrados = textosPuros.str.lower().str.split(' ')
print(textosQuebrados)

dicionario = set()
for lista in textosQuebrados:
    validas = [palavra for palavra in lista if len(palavra) > 1]
    dicionario.update(validas)
print(dicionario)

totalDePalavras = len(dicionario)
tuplas = zip(dicionario, range(totalDePalavras))
tradutor = {palavra: indice for palavra, indice in tuplas}
print(totalDePalavras)

vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]
print(vetoresDeTexto[0])

marcas = classificacoes['classificacao']
X = np.array(vetoresDeTexto)
Y = np.array(marcas.tolist())
porcentagem_de_treino = 0.86
tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_do_treino
print(tamanho_do_treino)

treino_dados = X[0:tamanho_do_treino]
treino_marcacoes = Y[0:tamanho_do_treino]
validacao_dados = X[tamanho_do_treino + 1:]
print(len(validacao_dados))

validacao_marcacoes = Y[tamanho_do_treino + 1:]
resultados = {}

modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost
print(resultados)

maximo = max(resultados)
vencedor = resultados[maximo]
print("Vencedor: ")
print(vencedor)

vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)
total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
print(msg)

# joblib.dump(vencedor, 'rede_neural_decisao.pkl')


acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)
print("Total de testes: %d " % len(validacao_dados))
