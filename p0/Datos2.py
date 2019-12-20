#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class Datos:
    TiposDeAtributos=('Continuo','Nominal') # variable no modificable que guarda las dos posibles descripciones de los tipos de atributos con los que vamos a trabajar
    tipoAtributos = [] # lista con la misma longitud que el numero de atributos y que contiene el tipo de atributo de cada variable
    nombreAtributos = [] # lista con la misma longitud que el numero de atributos y que contiene el nombre de cada variable
    nominalAtributos = [] # lista de valores booleanos con la misma longitud que el numero de atributos, True = nominal, False = no nominal
    datos = np.array(()) # array bidimensional de numpy que se utilizara para almacenar los datos
    diccionarios = [] # lista de diccionarios con la misma longitud que el numero de atributos (valor nominal/categorico [k] : valor numerico [v]), actualizar en datos
  # los atributos continuos tienen una entrada en el diccionario vacia
  # TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos, nominalAtributos, datos y diccionarios
  # NOTA: No confundir TiposDeAtributos con tipoAtributos
    
    def __init__(self, nombreFichero):
        
        f = open(nombreFichero, 'r')
        l = f.readline()
        nL = int(l) # numero de lineas del fichero
        
        self.nombreAtributos = f.readline().strip().split(',')
        self.tipoAtributos = f.readline().strip().split(',')
        
        nC = len(self.tipoAtributos) # numero de columnas
        
        # clasificamos los atributos en nominales = T y continuos = F
        for atr in Datos.tipoAtributos:
            if atr == 'Continuo':
                Datos.nominalAtributos.append(False) # añadir al final false
            elif atr == 'Nominal':
                Datos.nominalAtributos.append(True) # añadir al final True
            else:
                raise (ValueError)
                
        nuevo = np.zeros(3) # para nueva entrada
        aux = np.zeros(3) # auxiliar para cada tupla
        
        self.datos = np.empty((0,nC),np.float64)
        self.diccionarios = [{} for i in range(len(self.tipoAtributos))]
        
        for i in range(nL):
            tupla = l.split('\n')[0].split(',') # [x1, x2, ...]
            for j in range(len(tupla)):
                if self.tipoAtributos[j] == 'Nominal':
                    if not tupla[j] in self.diccionarios[j]: # si no esta en el diccionario se añade
                        self.diccionarios[j].update({tupla[j]:nuevo[j]})
                        nuevo[j] += 1
                        
                    aux[j] = self.diccionarios[j].get(tupla[j]) # añadimos el valor
                else:
                    aux[j] = tupla[j] # añadimos directamente si es continua
            
            self.datos = np.vstack([self.datos, [aux]]) # actualizamos en datos
        
    def extraeDatos(self, idx):
        pass
