from abc import ABCMeta,abstractmethod
import numpy as np
import math
import operator
import warnings
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances

#Ignoramos los errores de cálculo de media sobre elementos 0
warnings.simplefilter("ignore")

class Clasificador:

  # Clase abstracta
  __metaclass__ = ABCMeta

  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass

  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass

  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error

    #Ponemos 1 si son iguales, 0 en caso contrario.
    #Clases reales = datos la ultima columna
    return sum(map(lambda x, y: 0 if x == y else 1, datos[:, -1], pred)) / float(len(datos[:, -1]))

  def validateROC(self, datos, pred, diccionario_clasifica):
    K = len(diccionario_clasifica[-1].items())
    return self.compute_confusion_matrix(datos[:, -1].astype(int), pred.astype(int), K)

  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion

# print("ERROR con sklearn: " + str(1 - accuracy_score(y_test, predicciones)))
  def validacion(self,particionado,dataset,clasificador,seed=None, oneHot=False):

    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.

    # estrategia = particionado.creaParticiones(dataset, seed)
    # errores = np.array()
    if oneHot:
      encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
      X = encAtributos.fit_transform(dataset.datos[:, :-1])
      particionado.creaParticiones(X, seed)
    else:
      particionado.creaParticiones(dataset.datos, seed)

    #print('Longitud Particiones: ' + str(len(particionado.particiones)))
    #Si es 2 estamos haciendo Validacion Simple
    if particionado.numeroParticiones == 1:
      particion = particionado.particiones[0]
      clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain), dataset.nominalAtributos, dataset.diccionarios)
      clasificacion = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest), dataset.nominalAtributos, dataset.diccionarios)
      return self.error(dataset.extraeDatos(particion.indicesTest), clasificacion)
    #Estamos en la validación cruzada
    else:
      errores = np.array(())
      for i, particion in enumerate(particionado.particiones):
        train = clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain), dataset.nominalAtributos, dataset.diccionarios)
        clasificacion = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest), dataset.nominalAtributos, dataset.diccionarios)
        error_clasificacion = self.error(dataset.extraeDatos(particion.indicesTest), clasificacion)
        #print('Error Particion: ' + str(i) + ' ' + str(error_clasificacion))
        errores = np.append(errores, [error_clasificacion])
      return errores


  def validacionRoc(self, particionado, dataset, clasificador, label='label', seed=None, oneHot=False):
    TPR = []
    FPR = []

    if oneHot:
      encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
      X = encAtributos.fit_transform(dataset.datos[:, :-1])
      particionado.creaParticiones(X, seed)
    else:
      particionado.creaParticiones(dataset.datos, seed)

    if particionado.numeroParticiones == 1:
      particion = particionado.particiones[0]
      clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain), dataset.nominalAtributos, dataset.diccionarios)
      clasificacion = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest), dataset.nominalAtributos, dataset.diccionarios)
      TPR_val, FPR_val = self.validateROC(dataset.extraeDatos(particion.indicesTest), clasificacion, dataset.diccionarios)
      TPR.append(TPR_val)
      FPR.append(FPR_val)
      
    #Estamos en la validación cruzada
    else:

      for i, particion in enumerate(particionado.particiones):
      
        train = clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain), dataset.nominalAtributos, dataset.diccionarios)
        clasificacion = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest), dataset.nominalAtributos, dataset.diccionarios)
        TPR_val, FPR_val = self.validateROC(dataset.extraeDatos(particion.indicesTest), clasificacion, dataset.diccionarios)
        TPR.append(TPR_val)
        FPR.append(FPR_val)

    self.plot_roc_curve(TPR, FPR, label)

  def calculate_matrix_values(self, conf_matrix):
    TP = conf_matrix[0][0]
    FN = conf_matrix[1][0]
    FP = conf_matrix[0][1]
    TN = conf_matrix[1][1]
    # Specificity or true positive rate
    TPR = self.div(TP, (TP+FN))
    # Fall out or false positive rate
    FPR = self.div(FP, (FP+TN))
    #Accuracy
    ACC = self.div((TP+TN), (TP+FP+FN+TN))

    return(TPR, FPR, ACC)

  def div(self, a, b):
    if a == 0:
        return 0
    elif b == 0:
        return 0
    else:
        return a/b

  def compute_confusion_matrix(self, true, pred, K):

    result = np.zeros((K, K))
    for i in range(len(true)):
      result[true[i]][pred[i]] += 1

    #print('*Nuestra* Matriz de confusión Particion')
    #print(result)
#    print('Matriz confusion *SKLEARN*')
#    print(confusion_matrix(true, pred))
    TPR, FPR, ACC = self.calculate_matrix_values(result)

    return (TPR, FPR)

  def plot_roc_curve(self, TPR, FPR, label):
      #plt.figure()
      plt.plot(np.amin(FPR), np.amax(TPR), 'o', label=label)
      
      #plt.plot(FPR, TPR, label='ROC curve')
      plt.plot([0, 1], [0, 1], 'k--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate (FPR)')
      plt.ylabel('True Positive Rate (TPR)')
      plt.title('Receiver Operating Characteristic')
      plt.legend(loc="lower right")
      #plt.show()


##############################################################################

class ClasificadorNaiveBayes(Clasificador):


  def __init__(self, correccionLaplace=False):
    self.correccionLaplace = correccionLaplace
    self.calculos = []

  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):

    numero_datos = len(diccionario) - 1
    #Cogemos el ultimo elemento ya que es la clase
    diccionario_clase = diccionario[-1]
    tamano_clase = len(diccionario_clase)
    numero_elementos = datostrain.shape[0]
    self.ratio_clases = []

    #Cogemos el ratio de cada clase
    for class_key, class_value in diccionario[-1].items():
      cond_clase_actual = (datostrain[:, -1] == class_value)
      self.ratio_clases.append(np.sum(cond_clase_actual) / numero_elementos)

    for i in range(numero_datos):
      #Es discreto
      if atributosDiscretos[i]:
        numero_atributos_discretos = len(diccionario[i])

        #Creamos tablas de numero de atributos * numero de clases
        result_discreto = np.zeros((numero_atributos_discretos, tamano_clase))

        for row in datostrain:
          result_discreto[int(row[i]), int(row[-1])] += 1
          if self.correccionLaplace and np.any(result_discreto == 0):
            result_discreto += 1
        self.calculos.append(result_discreto)

      #Continuo
      else:
        #print(datostrain[:, i])
        #print(datostrain[:, [i, -1]]) # Imprime cada columna de los atributos junto con la clase esperada
        #2 para clasificar mean y varianza, y tamano_clase para todas las
        #posibilidades dentro de la clase
        diccionario_clase = diccionario[-1]
        result_continuo = np.zeros((2, tamano_clase)) # Matriz bidimensional de tamaño igual a las diferentes clases posibles

        for key, value in diccionario_clase.items():

          #Iteramos datostrain y cogemos el atributo que se relacciona con la clase
          #Con [:, [i, -1]] cogemos toda la fila, y dentro de la fila cogemos el atributo i
          #con [:, [i, -1]][:, -1] cogemos todas las filas devuelta y escogemos el atributo de la clase
          #para compararlo
          atributo_clase = datostrain[:, [i, -1]][:, -1] == value # Devolvemos un array de True|False indicando cuales estan relacionados con la clase
          #cogemos los atributos que se relaccionan con esa clase
          valores = datostrain[:, [i, -1]][atributo_clase]
          result_continuo[0, int(value)] = np.mean(valores[:, 0])
          result_continuo[1, int(value)] = np.var(valores[:, 0])

        self.calculos.append(result_continuo)

  def clasifica(self,datostest,atributosDiscretos,diccionario):

    seleccion_clase_filas = []

    for row in datostest:
      #Sacamos los valores de las clases
      clases_posteriori = {}

      for class_key, class_value in diccionario[-1].items():
        prob = 1
        for i in range(len(row)-1):
          if atributosDiscretos[i] == False:
            #Ponemos un multiplicatorio para calcular la probabilidade
            #de todas las clases
            clase_examinando = row[i]
            media = self.calculos[i][0][class_value]
            varianza = self.calculos[i][1][class_value]
            prob *= self.getGaussianNB(clase_examinando, media, varianza)

          else:
            #Multiplicamos las probabilidades de cada elemento dada la clase (5/20 * 3/20 * 1/20)
            prob_parcial = self.calculos[i][int(row[i]), class_value]
            prob_total = sum(self.calculos[i][:, class_value])
            prob *= prob_parcial/prob_total

        #Ajustamos probabilidades
        prob = prob * self.ratio_clases[class_value]
        #Añadimos el posteriori para cada una de las clases dentro de la fila
        clases_posteriori[class_key] = prob

        maximo_posteriori = max(clases_posteriori.keys(), key=lambda k: clases_posteriori[k])

      #Añadimos para cada clase el máximo
      seleccion_clase_filas.append(diccionario[-1][maximo_posteriori])

    return np.array(seleccion_clase_filas)

  def getGaussianNB(self, v, u_k, sigma_k):
    if sigma_k == 0:
      return 1
    raiz = math.sqrt(2*math.pi*sigma_k**2)
    exponencial = math.exp(-((v - u_k)**2)/(2*sigma_k**2))
    return exponencial/raiz

########################################################################################################################################################################
class ClasificadorVecinosProximos(Clasificador):
    def __init__(self, k=1, normalizar=False, weights='uniform'):
        self.k = k
        self.indicesTrain = np.array(())
        self.indicesTest = np.array(())
        self.weights = weights
        
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        self.indicesTrain = datostrain

    def distanciaEuclidea(self, arrayDist):
        d = 0
        for elem in arrayDist:
            d += math.pow(elem, 2)
        return d**0.5

    def distancia_knn(self, value):
        try:
          return 1/value
        except ZeroDivisionError:
          return 0.0

    def calculate_discrete_distance(self, obj1, obj2, len_dict):
        if obj1 < obj2:
            obj1, obj2 = obj2, obj1
        return min(obj1 - obj2, math.fmod(obj2, len_dict) + obj1)

    def clasifica(self,datostest,atributosDiscretos,diccionario):
        clasificacion = clases = []
        self.indicesTest = datostest[:,:-1]
        
        longitud = len(self.indicesTest[0])
        for i in range(self.indicesTest.shape[0]):
            d = []
            for j in range(len(self.indicesTrain)):
                d_aux = []
                for l in range(longitud):
                    if (atributosDiscretos[l] == True):
                        d_aux.append(self.calculate_discrete_distance(self.indicesTest[i][l], self.indicesTrain[j][l], len(diccionario)))
                    else:
                        d_aux.append(self.indicesTest[i][l] - self.indicesTrain[j][l])
                if (self.weights == 'uniform'):
                    d.append(self.distanciaEuclidea(d_aux))
                else:
                    d.append(self.distancia_knn(self.distanciaEuclidea(d_aux)))
                    
            # EN D TENGO TODAS LAS DISTANCIAS DE DATOSTRAIN CONTRA ESE ELEMENTO DE TEST
            best = np.argsort(d)[:self.k] if (self.weights == 'uniform') else np.argsort(d)[-self.k:]
            k_vecinos = self.indicesTrain[best,-1]
            value = np.bincount(k_vecinos.tolist()).argmax()
            clases.append(value)
            
        return np.array(clases)

########################################################################################################################################################################
class ClasificadorRegresionLogistica(Clasificador):

  def __init__(self, nEpocas=4, constante=1, w=None):
    self.constante = constante
    self.nEpocas = nEpocas
    self.w = w
    
  def sigma_func(self, value):
    try:
      return (1.0/(1.0 + math.exp(-value)))
    except OverflowError:
      return 1.0 if value > 0 else 0.0

  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
    #Creamos vector inicial con valores aleatorios

    if self.w is None:
      self.w = np.random.uniform(-0.5,0.5,len(diccionario))

    for i in range(0, self.nEpocas):
      for row in datostrain:
        x = np.append([1], row[:-1])
        sigma_func = self.sigma_func(np.dot(self.w, x))
        sigma = self.constante * (sigma_func - row[-1]) * x
        self.w = np.subtract(self.w, sigma)

  def clasifica(self,datostest,atributosDiscretos,diccionario):
    pred = []
    for row in datostest:
      x = np.append([1], row[:-1])
      sigma_func = self.sigma_func(np.dot(self.w, x))
      pred.append(1.0 if sigma_func >= 0.5 else 0.0)
    return np.array(pred)
