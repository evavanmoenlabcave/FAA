from abc import ABCMeta, abstractmethod
import numpy as np
import math

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

class Particion():

  # Esta clase mantiene la lista de indices de Train y Test para cada particion del conjunto de particiones
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:

  # Clase abstracta
  __metaclass__ = ABCMeta

  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta: nombreEstrategia, numeroParticiones, listaParticiones. Se pasan en el constructor

  def __init__(self):
    self.nombreEstrategia = ""
    self.numeroParticiones = 0
    self.particiones = []


  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta
  def creaParticiones(self,datos,seed=None):
    pass

  def mixElements(self, datos, seed=None):
    np.random.seed(seed)
    #Cogemos el tamano de la fila en numpy y barajamos
    list_index = list(range(0, datos.shape[0]))
    np.random.shuffle(list_index)
    return list_index

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar

  def __init__(self, percentaje=0.6):
    self.nombreEstrategia = "Validacion Simple"
    self.percentaje = percentaje
    self.particiones = []
    self.numeroParticiones = 0

  def creaParticiones(self,datos,seed=None):
    list_index = self.mixElements(datos, seed)
    #Separamos para train y test
    partition = Particion()
    cross_point = int(math.ceil(datos.shape[0] * self.percentaje))
    #Asignamos
    partition.indicesTrain = list_index[:cross_point]
    partition.indicesTest = list_index[cross_point:]
    #Anadimos
    self.particiones.append(partition)
    self.numeroParticiones = 1

#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar

  def __init__(self, k):
    self.nombreEstrategia = "Validacion Cruzada"
    self.particiones = []
    self.numeroParticiones = k

  def creaParticiones(self,datos,seed=None):
    list_index = self.mixElements(datos, seed)
    particiones = np.array_split(list_index, self.numeroParticiones)

    for i in range(self.numeroParticiones):
        partition = Particion()
        indicesTest = particiones[i]

        #Para cada sublista en la lista de particiones
          #Si no es indicesTest
          #Añadimos cada elemento de esta en item para aplanar la lista de listas

        indicesTrain = [item
                      for elem in particiones if elem is not indicesTest
                      for item in elem]

        partition.indicesTest = indicesTest
        partition.indicesTrain = indicesTrain
        self.particiones.append(partition)

class ValidacionSKLearn:

  def __init__(self, dataset):
      self.dataset = dataset

  def validacionSimple(self, testSize, laplace, multiAtributes):
    
    encAtributos = preprocessing.OneHotEncoder(categorical_features=self.dataset.nominalAtributos[:-1], sparse=False)
    X = encAtributos.fit_transform(self.dataset.datos[:,:-1])
    Y = self.dataset.datos[:,-1]
    
    if multiAtributes == False:
      clasificador = MultinomialNB(alpha= (0 if laplace else 1))
    else:
      clasificador = GaussianNB()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = testSize)
    predicciones = clasificador.fit(x_train, y_train).predict(x_test)
    
    return (1 - accuracy_score(y_test, predicciones), accuracy_score(y_test, predicciones).std())


  def validacionCruzada(self, partitionSize, laplace, multiAtributes):
    encAtributos = preprocessing.OneHotEncoder(categorical_features=self.dataset.nominalAtributos[:-1], sparse=False)
    X = encAtributos.fit_transform(self.dataset.datos[:,:-1])
    Y = self.dataset.datos[:,-1]
    if multiAtributes == False:
      clasificador = MultinomialNB(alpha= (0 if laplace else 1))
    else:
      clasificador = GaussianNB()

    cross_score = cross_val_score(clasificador, X, Y, cv=partitionSize)

    return (1 - cross_score.mean(), cross_score.std())
