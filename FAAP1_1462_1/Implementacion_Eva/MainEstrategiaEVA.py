"""
Practica 1

@author: Gloria del Valle
@author: Eva Gutierrez
@date: 09/10/19
"""
from Datos import Datos
from EstrategiaParticionadoEVA import ValidacionSimple,ValidacionCruzada,ValidacionSimpleSK,ValidacionCruzadaSK

dataset=Datos('ConjuntosDatos/lenses.data')

print ("-----------------------")
print ("VALIDACION SIMPLE (lenses.data): ")
e = ValidacionSimple('VS',3,0.8)
p = e.creaParticiones(dataset.datos)
for k in p:
	print ("Indices Test: ")
	print (k.indicesTest)
	print ("Indices Train: ")
	print (k.indicesTrain)
	print ("----")
print (e.nombreEstrategia)

print ("-----------------------")
print ("VALIDACION CRUZADA (lenses.data): ")
e = ValidacionCruzada('VC',5)
p = e.creaParticiones(dataset.datos)
for k in p:
	print ("Indices Test: ")
	print (k.indicesTest)
	print ("Indices Train: ")
	print (k.indicesTrain)
	print ("----")
print (e.nombreEstrategia)

print ("-----------------------")
print ("VALIDACION SIMPLE SK (lenses.data): ")
e = ValidacionSimpleSK('VS-SK',3,0.8)
p = e.creaParticiones(dataset.datos)
for k in p:
	print ("Indices Test: ")
	print (k.indicesTest)
	print ("Indices Train: ")
	print (k.indicesTrain)
	print ("----")
print (e.nombreEstrategia)

print ("-----------------------")
print ("VALIDACION CRUZADA SK (lenses.data): ")
e = ValidacionCruzadaSK('VC',5)
p = e.creaParticiones(dataset.datos)
for k in p:
	print ("Indices Test: ")
	print (k.indicesTest)
	print ("Indices Train: ")
	print (k.indicesTrain)
	print ("----")
print (e.nombreEstrategia)

"""
dataset=Datos('ConjuntosDatos/german.data')

print ("-----------------------")
print ("VALIDACION SIMPLE (german.data): ")
e1 = ValidacionSimple('VS',0)
p1 = e1.creaParticiones(dataset.datos)
print ("Indices Test: ")
print (p1[0].indicesTest)
print ("Indices Train: ")
print (p1[0].indicesTrain)
print (e1.nombreEstrategia)

print ("-----------------------")
print ("VALIDACION CRUZADA (german.data): ")
e2 = ValidacionCruzada('VC',5)
p2 = e2.creaParticiones(dataset.datos)

for k in p2:
	print ("Indices Test: ")
	print (k.indicesTest)
	print ("Indices Train: ")
	print (k.indicesTrain)
	print ("----")
print (e2.nombreEstrategia)

dataset=Datos('ConjuntosDatos/tic-tac-toe.data')

print ("-----------------------")
print ("VALIDACION SIMPLE (tic-tac-toe.data): ")
e1 = ValidacionSimple('VS',0)
p1 = e1.creaParticiones(dataset.datos)
print ("Indices Test: ")
print (p1[0].indicesTest)
print ("Indices Train: ")
print (p1[0].indicesTrain)
print (e1.nombreEstrategia)

print ("-----------------------")
print ("VALIDACION CRUZADA (tic-tac-toe.data): ")
e2 = ValidacionCruzada('VC',5)
p2 = e2.creaParticiones(dataset.datos)

for k in p2:
	print ("Indices Test: ")
	print (k.indicesTest)
	print ("Indices Train: ")
	print (k.indicesTrain)
	print ("----")
print (e2.nombreEstrategia)
"""