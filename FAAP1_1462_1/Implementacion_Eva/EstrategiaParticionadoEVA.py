from abc import ABCMeta,abstractmethod
from sklearn.model_selection import ShuffleSplit,KFold
import numpy as np
import random

class Particion():

  # Esta clase mantiene la lista de índices de Train y Test para cada partición del conjunto de particiones  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta: nombreEstrategia, numeroParticiones, listaParticiones. 
  # Se pasan en el constructor 

  def __init__(self, nombreEstrategia, numeroParticiones):
    self.nombreEstrategia = nombreEstrategia
    self.numeroParticiones = numeroParticiones
    self.listaParticiones = []

  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

  def __init__(self, nombreEstrategia, numeroParticiones, porcentaje):
    EstrategiaParticionado.__init__(self, nombreEstrategia, numeroParticiones)
    self.porcentaje = porcentaje

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):
    
    random.seed(seed)

    # K iteraciones = numero de particiones
    for k in range(self.numeroParticiones):

      p = Particion()

      # Obtenemos una array aleatoria de las filas de nuestros datos
      npaux = np.random.permutation(datos.shape[0])

      i = int(len(npaux.tolist())*self.porcentaje) # indice del porcentaje deseado para entrenamiento
      
      # Dividimos datos en entrenamiento y de prueba
      p.indicesTrain = npaux.tolist()[0:i]
      p.indicesTest = npaux.tolist()[i:]

      self.listaParticiones.append(p) # Insertamos la particion creada

    return self.listaParticiones


#####################################################################################################

class ValidacionSimpleSK(EstrategiaParticionado):

  def __init__(self, nombreEstrategia, numeroParticiones, porcentaje):
    EstrategiaParticionado.__init__(self, nombreEstrategia, numeroParticiones)
    self.porcentaje = porcentaje

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):

    # Dividimos datos en subconjuntos de test y train con ShuffleSplit
    ss = ShuffleSplit(n_splits=self.numeroParticiones, train_size=self.porcentaje)

    # Creamos una particion por cada split de datos
    for train, test in ss.split(datos):

      p = Particion()

      p.indicesTrain = train
      p.indicesTest = test

      self.listaParticiones.append(p) # insertamos la particion creada

    return self.listaParticiones

    
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    
    random.seed(seed)

    # Obtenemos una array aleatoria de las filas de nuestros datos
    npaux1 = np.random.permutation(datos.shape[0])

    nL = int(len(npaux1.tolist())) # numero de lineas de datos

    step = int(nL/self.numeroParticiones) # numero de datos por cada subconjunto

    # laux es una lista de la array aleatoria (npaux2) de nuestros indices de los subconjuntos
    npaux2 = np.random.permutation(range(0,nL,step))
    laux = npaux2.tolist()
    
    # Dividimos en K-1 subconjuntos para entrenamiento y el resto test
    for k in range(self.numeroParticiones):
      
      p = Particion()

      # Primero obtenemos el subconjunto para tests
      i = laux[0] # el indice inicial de nuestros datos de test
      p.indicesTest = npaux1.tolist()[i:i+step]
      laux.pop(0) # eliminamos el indice de bloque ya usado para test

      # K-1 subconjuntos para datos de entrenamiento
      for j in npaux2.tolist():
        if j != i:
          # Si tratamos con el ultimo bloque de datos
          if j == max(npaux2.tolist()):
            p.indicesTrain.extend(npaux1.tolist()[j:])
          else:
            p.indicesTrain.extend(npaux1.tolist()[j:j+step])

      # Insertamos la particion creada de cada iteracion
      self.listaParticiones.append(p)

    return self.listaParticiones

    
#####################################################################################################      
class ValidacionCruzadaSK(EstrategiaParticionado):
  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    
    # Dividimos datos en subconjuntos de test y train con KFold
    kf = KFold(n_splits=self.numeroParticiones, shuffle=True)

    # Creamos una particion por cada split de datos
    for train, test in kf.split(datos):

      p = Particion()

      p.indicesTrain = train
      p.indicesTest = test

      self.listaParticiones.append(p) # insertamos la particion creada

    return self.listaParticiones