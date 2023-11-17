# -------------------------------| Bibliotecas |-------------------------------
#os
import os.path
import sys
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.sparse import hstack

# -------------------------------| Funciones |-------------------------------
def load_sel():
	lexicon_sel = {}
	input_file = open('apoyo/SEL_full.txt', 'r')

	# Para cada línea del archivo
	for line in input_file:
		#Las líneas del lexicon tienen el siguiente formato:
		#abundancia	0	0	50	50	0.83	Alegría
		
		palabras = line.split("\t")					# Esto hara que se separe la linea en una lista de palabras
		palabras[6]= re.sub('\n', '', palabras[6])	# Esto hara que se elimine el salto de linea
		pair = (palabras[6], palabras[5])			# Esto hara que se guarde la palabra y su valor de polaridad

		if lexicon_sel:								# Si el diccionario no esta vacio
			if palabras[0] not in lexicon_sel:		# Si la palabra no esta en el diccionario
				lista = [pair]						# Se crea una lista con la palabra y su valor de polaridad
				lexicon_sel[palabras[0]] = lista	# Se agrega la lista al diccionario
			else:
				lexicon_sel[palabras[0]].append (pair)	# Si la palabra ya esta en el diccionario, se agrega la palabra y su valor de polaridad a la lista
		else:										# Si el diccionario esta vacio
			lista = [pair]							# Se crea una lista con la palabra y su valor de polaridad
			lexicon_sel[palabras[0]] = lista		# Se agrega la lista al diccionario

	input_file.close()								# Se cierra el archivo
	del lexicon_sel['Palabra']; #Esta llave se inserta porque es parte del encabezado del diccionario, por lo que se requiere eliminar
	#Estructura resultante
		#'hastiar': [('Enojo\n', '0.629'), ('Repulsi\xf3n\n', '0.596')]
	return lexicon_sel								# Se regresa el diccionario

def getSELFeatures(cadenas, lexicon_sel):
	"""
	Esta función recibe una lista de cadenas y un diccionario con el lexicon SEL
	"""
	#'hastiar': [('Enojo\n', '0.629'), ('Repulsi\xf3n\n', '0.596')]
	features = []
	for cadena in cadenas:
		valor_alegria = 0.0
		valor_enojo = 0.0
		valor_miedo = 0.0
		valor_repulsion = 0.0
		valor_sorpresa = 0.0
		valor_tristeza = 0.0
		cadena_palabras = re.split('\s+', cadena)
		dic = {}
		for palabra in cadena_palabras:
			if palabra in lexicon_sel:
				caracteristicas = lexicon_sel[palabra]
				for emocion, valor in caracteristicas:
					if emocion == 'Alegría':
						valor_alegria = valor_alegria + float(valor)
					elif emocion == 'Tristeza':
						valor_tristeza = valor_tristeza + float(valor)
					elif emocion == 'Enojo':
						valor_enojo = valor_enojo + float(valor)
					elif emocion == 'Repulsión':
						valor_repulsion = valor_repulsion + float(valor)
					elif emocion == 'Miedo':
						valor_miedo = valor_miedo + float(valor)
					elif emocion == 'Sorpresa':
						valor_sorpresa = valor_sorpresa + float(valor)
		dic['__alegria__'] = valor_alegria
		dic['__tristeza__'] = valor_tristeza
		dic['__enojo__'] = valor_enojo
		dic['__repulsion__'] = valor_repulsion
		dic['__miedo__'] = valor_miedo
		dic['__sorpresa__'] = valor_sorpresa
		
		#Esto es para los valores acumulados del mapeo a positivo (alegría + sorpresa) y negativo (enojo + miedo + repulsión + tristeza)
		dic['acumuladopositivo'] = dic['__alegria__'] + dic['__sorpresa__']
		dic['acumuladonegative'] = dic['__enojo__'] + dic['__miedo__'] + dic['__repulsion__'] + dic['__tristeza__']
		
		features.append (dic)
	return features

if __name__=='__main__':
	# ---------------------------| Crear archivo auxiliar |-------------------------------
	if (os.path.exists('lexicon_sel.pkl')):
		lexicon_sel_file = open ('lexicon_sel.pkl','rb')
		lexicon_sel = pickle.load(lexicon_sel_file)
	else:
		lexicon_sel = load_sel()
		lexicon_sel_file = open ('lexicon_sel.pkl','wb')
		pickle.dump(lexicon_sel, lexicon_sel_file)
		lexicon_sel_file.close()
	
	# ---------------------------| Polaridad |-------------------------------
	cadena1 = 'el mejor vista de guanajuato_es uno mirador precioso y con el mejor vista de el ciudad de guanajuato . el monumento ser impresionante . frente_a el monumento ( por el parte de atrás de el pípila ) haber uno serie de local en donde vender artesanía ... si te gustar algo de ahí , comprar . a mí me pasar que ver algo y no el comprar pensar que el ver más tarde en otro lado y no ser así . te recomer que llegar hasta ahí en taxi , ser muy económico , porque como estar en uno lugar muy alto , ser muy cansar llegar caminar , aunque no estar lejos_de el centro . peroooo ... bajar caminar por el mini callejón . ¡ ser algo precioso ! te llevar directamente por uno lado de el teatro_juárez . '
	cadena2 = '¡ malo ! no gastar tu dinero ahí malo condición , deplorable . definitivamente no gastar tu dinero ahí , mejor ver a gastar en dulce en el tienda de la catrina .'
	cadenas = []
	cadenas. append(cadena1)
	cadenas. append(cadena2)
	
	polaridad = getSELFeatures(cadenas, lexicon_sel)
	print("---------------------------------------------------------")
	for x in polaridad:
		print(x)
	print("---------------------------------------------------------")
	
	# ---------------------------| Crear representación vectorial |-------------------------------
	vectorizador = CountVectorizer()
	X = vectorizador.fit_transform(cadenas)
	print ('Vectorizado')
	# print (X.toarray())
	# print(vectorizador.get_feature_names_out())
	
	# ---------------------------| Manejo de polaridad |-------------------------------
	# --> Cadena 1
	polaridad_cadena_1_pos = np.array([polaridad[0]['acumuladopositivo']])
	polaridad_cadena_1_neg = np.array([polaridad[0]['acumuladonegative']])
	polaridad_cadena_1 = np.concatenate((polaridad_cadena_1_pos, polaridad_cadena_1_neg), axis=0)
	print("-----------------| Polaridad cadena 1 |-----------------")
	print (polaridad_cadena_1)
	
	# --> Cadena 2
	polaridad_cadena_2_pos = np.array([polaridad[1]['acumuladopositivo']])
	polaridad_cadena_2_neg = np.array([polaridad[1]['acumuladonegative']])
	polaridad_cadena_2 = np.concatenate((polaridad_cadena_2_pos, polaridad_cadena_2_neg), axis=0)
	print("-----------------| Polaridad cadena 2 |-----------------")
	print (polaridad_cadena_2)
	
	# --> Concatenar cadenas
	polaridad_cadenas = np.stack((polaridad_cadena_1, polaridad_cadena_2))
	
	print ('Polaridad')
	print (polaridad_cadenas)

	# ---------------------------| Representación de texto |-------------------------------
	vectorizado_con_polaridad = hstack([X,polaridad_cadenas]).toarray()

	print ('Vectorizado + polaridad')
	print (vectorizado_con_polaridad)
