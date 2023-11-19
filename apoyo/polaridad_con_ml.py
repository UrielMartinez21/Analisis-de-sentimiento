from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os, pickle
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

class data_set_polarity:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class data_set_attraction:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

if not (os.path.exists('apoyo/corpus_attraction.pkl')):
	print ('no se ha generado el corpus lematizado')
else:
	corpus_file = open ('apoyo/corpus_attraction.pkl','rb')
	corpus_attraction = pickle.load(corpus_file)
	corpus_file.close()

if not (os.path.exists('apoyo/corpus_polarity.pkl')):
	print ('no se ha generado el corpus lematizado')
else:
	corpus_file = open ('apoyo/corpus_polarity.pkl','rb')
	corpus_polarity = pickle.load(corpus_file)
	corpus_file.close()

print (corpus_attraction.X_train[0])

# Representación vectorial binarizada
vectorizador_binario = CountVectorizer(binary=True)
vectorizador_binario_fit = vectorizador_binario.fit(corpus_attraction.X_train)
X_train = vectorizador_binario_fit.transform(corpus_attraction.X_train)
y_train = corpus_attraction.y_train
# print (vectorizador_binario.get_feature_names_out())
# print (len(vectorizador_binario.get_feature_names_out()))
# Imprimir solo los primeros 10 nombres de características
print(vectorizador_binario.get_feature_names_out()[:10])

# O evitar imprimir caracteres no ASCII
print([feature for feature in vectorizador_binario.get_feature_names_out() if feature.isascii()])
print (X_train.shape)#sparse matrix
# ~ clf = LogisticRegression()
# ~ clf = LogisticRegression(max_iter=10000)
# ~ clf.fit(X_train, y_train)


X_test = vectorizador_binario_fit.transform(corpus_attraction.X_test)
y_test = corpus_attraction.y_test
# ~ print (vectorizador_binario_fit.get_feature_names_out())
# ~ print (X_test.shape)#sparse matrix

# ~ y_pred = clf.predict(X_test)
#~ print (y_pred)
# ~ print(accuracy_score(y_test, y_pred))
# ~ print(confusion_matrix(y_test, y_pred,labels=['Restaurant','Hotel','Attractive']))
# ~ target_names = ['Restaurant','Hotel','Attractive']
# ~ print(classification_report(y_test, y_pred, target_names=target_names))

y_train_polarity = corpus_polarity.y_train
clf_polarity = LogisticRegression(max_iter=10000)
clf_polarity.fit(X_train, y_train_polarity)
y_test = corpus_polarity.y_test
y_pred = clf_polarity.predict(X_test)
print("-------------------------| Evaluar rendimiento |-------------------------")
print (y_pred)
print(accuracy_score(y_test, y_pred))

print("-------------------------| Matriz de confusión |-------------------------")
print(confusion_matrix(y_test, y_pred,labels=[1,2,3,4,5]))

print("-------------------------| Reporte |-------------------------")
target_names = ['1','2','3','4','5']
print(classification_report(y_test, y_pred, target_names=target_names))

print ('Realizando validación cruzada...')
cv_results = cross_validate(clf_polarity, X_train, y_train_polarity, cv=5, scoring='f1_macro', verbose = 3, n_jobs = -1)
print (cv_results)
