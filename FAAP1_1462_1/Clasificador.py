#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABCMeta,abstractmethod
import numpy as np
import math
import operator
import EstrategiaParticionado
from functools import reduce


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
    err = 0
    elems = datos.shape[0]
    
    if elems != pred.shape[0]:
        pass
    for i in range(elems):
        if datos[i][-1] != pred[i]:
            err = err + 1
            
    return (err/float(elems))
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None,laPlace=False):
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opción es repetir la validación simple un número especificado de veces, obteniendo en cada una un error. 
    # Finalmente se calcularía la media.
    particionado.creaParticiones(dataset.datos, None)
    err=[]
    data_d = dataset.diccionarios
    data_a_d = dataset.nominalAtributos
    
    # validacion simple
    if (particionado.nombreEstrategia == "ValidacionSimple"):
        clasificador.entrenamiento(datostrain=dataset.extraeDatos(particionado.listaParticiones[0].indicesTrain),
                                   atributosDiscretos=data_a_d,
                                   diccionario=data_d,
                                   laPlace=laPlace)
        prediccion = clasificador.clasifica(datostest=dataset.extraeDatos(particionado.listaParticiones[0].indicesTest),
                               atributosDiscretos=data_a_d,
                               diccionario=data_d)
        err.append(clasificador.error(datos=dataset.extraeDatos(particionado.listaParticiones[0].indicesTest),
                                    pred=prediccion))
    elif (particionado.nombreEstrategia == "ValidacionCruzada"):
        for i in range(particionado.numeroParticiones):
            clasificador.entrenamiento(datostrain=dataset.extraeDatos(particionado.listaParticiones[i].indicesTrain),
                                   atributosDiscretos=data_a_d,
                                   diccionario=data_d,
                                   laPlace=laPlace)
            prediccion = clasificador.clasifica(datostest=dataset.extraeDatos(particionado.listaParticiones[i].indicesTest),
                                   atributosDiscretos=data_a_d,
                                   diccionario=data_d)
            err.append(clasificador.error(datos=dataset.extraeDatos(particionado.listaParticiones[i].indicesTest),
                                        pred=prediccion))
    
    
    return err, prediccion
    

##############################################################################

class ClasificadorNaiveBayes(Clasificador):
    dicc_likelihood = {}
    dicc_atributos = {}
    dicc_clases = {}
 # contar la frecuencia a priori de las veces que sale A y las que sale B
 # con los atributos continuos suponemos que siguen una distribucion normal
 # calcula la media y la desviacion tipica
 # modelo de distr de probabilidad que dice si para esa clase clasifique
 # clasifica te devuelve una lista de las clases que predices
  
 # TODO: implementar
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario,laPlace=False):
      atributos = {}
      clases = {}
      likelihood = []
      indices = []
      aux = []
      n_elems = datostrain.shape[0]
      size = datostrain.shape[1] -1
      #n_clases = len(diccionario[-1])
      #
      for i in range(size): # para cada columna
          #inicializamos el diccionario
          atributos.update({i:{}})
          
          if atributosDiscretos[i]: #si son atributos discretos
              for k,v in diccionario[i].items():
                  atributos[i].update({v:{}})
                  for k2,v2 in diccionario[-1].items():
                      #para cada una de las clases
                      indices = [j for j in range(n_elems) if datostrain[j][-1] == v and datostrain[j][-1] == v2]
                      
                      #meter en tabla aux para luego hacer la correcion de laplace
                      if laPlace and not indices:
                          aux.append(i)
                      atributos[i][v].update({v2:len(indices)})

          else: # si son continuos
              atributos[i].update({'m':{}})
              atributos[i].update({'v':{}})
              for k,v in diccionario[-1].items():
                  # metemos la lista con las filas de cada clase
                  likelihood = [datostrain[j][i] for j in range(n_elems) if datostrain[j][-1] == v]
                  m = np.mean(likelihood)
                  V = np.std(np.array(likelihood))
                  atributos[i]['m'].update({v:m})
                  atributos[i]['v'].update({v:V})

      if laPlace:
          for count in aux: # para cada t a aplicar laPlace
              for k, v in atributos[count].items(): # para cada columna
                  for k2,v2 in atributos[count][k].items():
                      atributos[count][k][k2] += 1
                      
      for i in range(n_elems):
          if (datostrain[i][-1] in clases.keys()):  
              clases[datostrain[i][-1]] += 1
          else:
              clases[datostrain[i][-1]] = 1
              
      self.dicc_atributos = atributos
      self.dicc_clases = clases
      
    def clasifica(self,datostest,atributosDiscretos,diccionario):
      posterior = {}
      prior = {}
      likelihood = []
      bayes = []

      n = sum(list(self.dicc_clases.values()))
      
      for k,v in self.dicc_clases.items():
          prior.update({k:(v/n)})
          
      for count in range(len(datostest)):
          
          posterior.update({count:{}})
          for k,v in self.dicc_clases.items():
              for count2 in range(datostest.shape[1] -1):
                  if 'm' in self.dicc_atributos[count2].keys():
                      m = self.dicc_atributos[count2]['m'][k]
                      var = self.dicc_atributos[count2]['v'][k]
                      gauss = dist_normal(m,var,datostest[count][count2])
                      likelihood.append(gauss)
                  else:
                      c = datostest[count][count2]
                      likelihood.append(self.dicc_atributos[count2][c][k] / float(v))
              bayes.append(reduce(lambda x, y: x*y, likelihood)*prior[k])
              posterior[count][k] = bayes
      pred = np.zeros(datostest.shape[0])
      for i in range(datostest.shape[0]):
          pred[i] = max(posterior[i].items(), key=operator.itemgetter(1))[0]
          
      return pred
          
def dist_normal(m,v,n):
    if (v == 0):
        v += math.pow(10, -6)
        
    exp = -(math.pow((n-m), 2)/(2*v))
    base = 1/math.sqrt(2*math.pi*v)
    densidad = base*math.pow(math.e,exp)
    return densidad