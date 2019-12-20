# -*- coding: utf-8 -*-
"""

@author: profesores faa
"""

from Datos import Datos

dataset=Datos('ConjuntosDatos/german.data')
#dataset=Datos('ConjuntosDatos/tic-tac-toe.data')
print (dataset.nombreAtributos)
print (dataset.tipoAtributos)
print (dataset.nominalAtributos)
print (dataset.datos)