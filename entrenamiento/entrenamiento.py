# -----------------------| Bibliotecas |----------------------- #
import pickle
import statistics
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Modelos de clasificación
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# -----------------------| Importar rasgos y clases |----------------------- #
with open("../normalizacion_texto/lista_contenido_normalizado.pkl", "rb") as f:
    corpus = pickle.load(f)

data = pd.read_excel('../apoyo/Rest_Mex_2022.xlsx')
y = data['Polarity'].values

# -----------------------| Rasgos y pruebas |----------------------- #
x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=0)

# -----------------------| Representación de texto |----------------------- #
# Vector tf-idf
vectorizador_tfidf = TfidfVectorizer(token_pattern=r'(?u)\w\w+|\w\w+\n|\.')

X_train_tfidf = vectorizador_tfidf.fit_transform(x_train).toarray()
X_test_tfidf = vectorizador_tfidf.transform(x_test).toarray()

# -----------------------| KFold Cross Validation |----------------------- #
# clf = LogisticRegression(solver='lbfgs', max_iter=1000)
# clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
f1_macro = []

# -----------------------| Evaluacion de rendimiento |----------------------- #
# --> Iteraciones
for entrenamiento, prueba in skf.split(X_train_tfidf, y_train):
    # --> Validacion cruzada
    rasgos_entrenamiento = X_train_tfidf[entrenamiento]
    rasgos_prueba = X_train_tfidf[prueba]

    clases_entrenamiento = y_train[entrenamiento]
    clases_prueba = y_train[prueba]

    # --> Entrenamiento
    clf.fit(rasgos_entrenamiento, clases_entrenamiento)
    clase_predicha = clf.predict(rasgos_prueba)

    # --> Métrica de rendimiento
    f1_macro.append(f1_score(clases_prueba, clase_predicha, average='macro'))

    # --> Porcentaje avanzado
    print(f"\rProgreso: {len(f1_macro)/skf.get_n_splits():.1%}", end="")


# -----------------------| Resultados |----------------------- #
# --> Resultados
print(f"F1 macro: {round(statistics.mean(f1_macro), 2)*100}%")

# --> Guardar resultados en txt uno sobre otro
with open("resultados.txt", "a") as f:
    f.write(f"KNeighborsClassifier: {round(statistics.mean(f1_macro), 2)*100}%\n")