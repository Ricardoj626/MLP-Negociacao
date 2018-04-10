'''
pip install -r requeriments.txt
'''

from lex_frag import frag
from sklearn.externals import joblib
import numpy as np

textoString = 'ate toparia mas na verdade quero que pague mais'

saida = []

for xx in frag:
    palavra = textoString.find(xx)
    if palavra < 0:
        saida.append(0)
    else:
        saida.append(1)


entrada = np.array(saida)
entrada = entrada.reshape(1, -1)
clf = joblib.load('rede_neural_decisao.pkl')
saida_rede = clf.predict(entrada)


print("   Entrada: \n'{}'".format(textoString))
print('----------------------------------------\n   Entrada da rede:')
print(saida)
print('----------------------------------------\n   Classificador usado:')
print(clf)
print('----------------------------------------\n[1] se SIM, [2] se NEGOCIAR, [3] se NAO\n')
print(saida_rede)