import numpy as np
import collections
import sys
import json

class Datos:
  
  def formatDict(self, dict):
    return json.dumps(dict, indent=4)

  def dataToLists(self, data):
    return list(map(lambda line: line.rstrip().split(','), data[3:]))

  def __init__(self, nombreFichero, normalizar=False):

    self.TiposDeAtributos= ('Continuo','Nominal')
    self.tipoAtributos = [] 
    self.nombreAtributos = []
    self.nominalAtributos = []
    self.datos = np.array(())
    self.diccionarios = []
    self.medias = []
    self.desviaciones = []

    with open(nombreFichero) as f:
      sets = []
      listaAux = set()
      lines = f.readlines()
      
      numElementos = int(lines[0].rstrip())
      self.nombreAtributos = lines[1].rstrip().split(',')
      self.tipoAtributos = lines[2].rstrip().split(',')
      # print(len(self.tipoAtributos))
      # print(len(self.nombreAtributos))

      self.nominalAtributos = list(map(lambda elem: True if elem == self.TiposDeAtributos[1] else False, self.tipoAtributos))
      datosCopy = np.array(self.dataToLists(lines))
      numAtributos = len(self.nominalAtributos)

      for i in range(numAtributos):
        if self.nominalAtributos[i] == False:
          self.diccionarios.append({})
        else:
          for j in range(numElementos):
            listaAux.add(datosCopy[j][i])

          sortedSet = sorted(list(listaAux))
          if self.nombreAtributos[i] == 'Month':
            dictMonth = {'Jan':0, 'Feb':1, 'Mar':2, 'Apr': 3, 'May':4, 'June':5, 'Jul':6, 'Aug':7, 'Sep':8, 'Oct':9, 'Nov': 10, 'Dec': 11}
            d = collections.OrderedDict(dictMonth)
          else:
            d = collections.OrderedDict(map(lambda t: (t[1], t[0]), enumerate(sortedSet)))
          sets.append(sortedSet)
          self.diccionarios.append(d)
          
          listaAux = set()

      self.datos = np.zeros((numElementos, numAtributos))
      
      for i in range(numAtributos):
        if self.nominalAtributos[i] == False:
          self.datos[:, i] = datosCopy[:, i]
        else:
          for j in range(numElementos):
            self.datos[j, i] = self.diccionarios[i][datosCopy[j,i]]
            
      if normalizar:
        self.calcularMediasDesv(self.datos, self.nominalAtributos)
        self.datos = self.normalizarDatos(self.datos)
        
        
  def calcularMediasDesv(self, datos, atributosDiscretos):
      
      self.medias = [0]*(len(atributosDiscretos))
      self.desviaciones = [0]*(len(atributosDiscretos))

      for count in range(len(atributosDiscretos)):
          if atributosDiscretos[count] == True:
              self.medias[count] = None
              self.desviaciones[count] = None
          else:
              data = np.array(datos[:,count])
              self.medias[count] = np.mean(data)
              self.desviaciones[count] = np.std(data)
              
  def normalizarDatos(self,datos):
      datos_normalizados = np.zeros((datos.shape[0], datos.shape[1]))
      
      for i in range(datos.shape[0]):
          for j in range(datos.shape[1]):
              if self.medias[j] == None:
                  datos_normalizados[i, j] = datos[i, j]
              else:
                  datos_normalizados[i, j] = (datos[i, j] - self.medias[j])/self.desviaciones[j] ##MIRAR i o j
      return datos_normalizados

  def extraeDatos(self, idx):
    return np.take(self.datos,idx,axis=0)

