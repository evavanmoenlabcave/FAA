import numpy as np
import random
import operator
import copy
from Clasificador import Clasificador
from math import ceil, floor
from operator import itemgetter

class Clasificador_AG(Clasificador):

	def __init__(self, dataset, estrategia, mostrar_proceso=True, plots=False, option=False, n_reglas=10, poblacion=200, generacion=300, cruce=0.85, mutacion=0.1, elitismo=0.05):

		assert(n_reglas>0)
		self.dataset = dataset
		self.datostrain = []
		if option == True:
			self.n_reglas = random.randint(1, n_reglas)
		else:
			self.n_reglas = n_reglas
		self.estrategia = estrategia
		self.mostrar_proceso = mostrar_proceso
		self.plots = plots
		self.poblacion = poblacion
		self.generacion = generacion
		self.cruce = cruce
		self.mutacion = mutacion
		self.elitismo = elitismo
		self.best = None
		self.estrategia.creaParticiones(dataset.datos)
		self.datostrain = dataset.extraeDatos(self.estrategia.particiones[-1].indicesTrain)
		self.datostest = dataset.extraeDatos(self.estrategia.particiones[-1].indicesTest)

	def randomizer(self, progenitor):
		for p in progenitor[:-1]:
			if (all(v == 0 for v in p)):
				variacion = random.randint(0,len(p)-1)
				p[variacion] = 1
		return progenitor

	def reglas_ini(self):
		l_r = []
		for _ in range(self.n_reglas):
			sub_r = []
			for d in self.dataset.diccionarios[:-1]:
				sub_d = []
				if len(d) == 0:
					longitud = 1
				else:
					longitud = len(d)
				for _ in range(longitud):
					sub_d.append(np.random.randint(0,2))
				sub_r.append(sub_d)
			clase = np.random.randint(0,2)
			sub_r.append([clase])
			aux = self.randomizer(sub_r)
			l_r.append(aux)
		return l_r

	def poblacion_ini(self):
		l_p = []
		for _ in range(self.poblacion):
			l_p.append(self.reglas_ini())
		return l_p

	def conteo_aciertos(self, lista, regla, fila):
		visto = 1
		for n, ind in enumerate(fila[:-1]):
			if regla[n][int(ind)] == 0:
				visto = 0
				break
			if visto == 0:
				break
		if visto:
			lista.append(regla[-1][0])
		return lista

	def fitness(self, individuo):
		acierto = 0
		copia_individuo = copy.deepcopy(individuo)
		for fila in self.datostrain:
			conjunto = []
			for r in copia_individuo:
				conjunto = self.conteo_aciertos(conjunto, r, fila)
			if conjunto:
				if np.argmax(np.bincount(conjunto)) == int(fila[-1]):
					acierto += 1
		return acierto/len(self.datostrain)

	def fitness_poblacion(self, poblacion, fitness_indv):
		for n, indv in enumerate(poblacion):
			fitness_indv[n] = self.fitness(indv)
		return fitness_indv, np.mean(fitness_indv)

	def seleccion_proporcional(self, poblacion, fitness):
		total_fit = np.sum(fitness)
		acc = 0
		punto = random.uniform(0, total_fit)
		for n, valor in enumerate(fitness):
			acc += valor
			if punto < acc:
				return poblacion[n]
		if acc == 0:
			return poblacion[0]

	def progenitor(self, poblacion, fitness, tam):
		seleccion_progenitores = []
		for i in range(tam):
			copia = copy.deepcopy(self.seleccion_proporcional(poblacion, fitness))
			seleccion_progenitores.append(copia)
		return seleccion_progenitores

	def cruce_punto(self, padre1, padre2):
		l1, l2 = [], []
		pos = random.randint(1, min(len(padre1), len(padre2))) -1
		if random.uniform(0,1) <= self.cruce:
			l1 += list(padre1[:pos])
			l1 += list(padre2[pos:])
			l2 += list(padre2[:pos])
			l2 += list(padre1[pos:])
			return l1, l2
		return padre1, padre2

	def cruzar_ag(self, padres_t):
		l_cruce = []
		mitad = int(len(padres_t)/2)
		for i in range(mitad):
			padre1, padre2 = [], []
			ant = -(i+1)
			long1, long2 = len(padres_t[i]), len(padres_t[ant])
			longitud = min(long1, long2)
			for j in range(longitud):
				p1, p2 = self.cruce_punto(padres_t[i][j], padres_t[ant][j])
				padre1.append(p1)
				padre2.append(p2)
			l_cruce.append(padre1)
			l_cruce.append(padre2)
		if (len(padres_t) % 2 != 0):
			l_cruce.append(padres_t[-1])
		return l_cruce

	def mutar_individuo(self, individuo):
		for r in individuo:
			m = random.randrange(len(r))
			n = random.randrange(len(r[m]))
			r[m][n] = 0 if r[m][n] == 1 else 1
		return individuo

	def mutar(self, poblacion_t):
		l_mutacion = []
		for i in poblacion_t:
			copia = copy.deepcopy(i)
			l_mutacion.append(self.mutar_individuo(copia))
		return l_mutacion

	def elite(self, poblacion_t):
		n = int(ceil(self.poblacion*self.elitismo))
		elite = []
		if len(poblacion_t) < n:
			n = len(poblacion_t)
		for i in range(n):
			elite.append(poblacion_t[i][0])
		return elite

	def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
		poblacion = self.poblacion_ini()
		fit = [0 for i in range(len(poblacion))]

		if self.plots:
			mejor_l_fit, mejor_l_pob = [], []
		for g in range(self.generacion):
			tupla_poblacion = []
			if self.mostrar_proceso == True:
				print("\nGeneracion: ", g+1)
			copia_poblacion, copia_fit = copy.deepcopy(poblacion), copy.deepcopy(fit)
			
			fit, fit_p = self.fitness_poblacion(copia_poblacion, copia_fit)
			
			if self.mostrar_proceso == True:
				print("Mejor fitness: ", max(fit), " | Fitness medio poblacion: ", fit_p)
			if self.plots:
				mejor_l_fit.append(max(fit))
				mejor_l_pob.append(fit_p)

			tupla_poblacion = list(zip(copia_poblacion, copia_fit))
			# if self.mostrar_proceso == True:	
			# 	best = max(tupla_poblacion, key=itemgetter(1))
			# 	print("Regla: ", best[0])
			elite = self.elite(tupla_poblacion)

			l_c = self.cruzar_ag(self.progenitor(poblacion, fit, floor(self.poblacion*self.cruce)))
			l_m = self.mutar(self.progenitor(poblacion, fit, floor(self.poblacion*self.mutacion)))
			l_c.extend(l_m)
			l_c.extend(elite)
			poblacion = copy.deepcopy(l_c)

		tupla_poblacion = list(zip(copia_poblacion, copia_fit))
		best = max(tupla_poblacion, key=itemgetter(1))
		self.best = best[0]
		print("Best: ", self.best, "| fit:", best[1])
		if self.plots:
			return mejor_l_fit, mejor_l_pob

	def clasifica(self, datostest, atributosDiscretos, diccionario):
		clasificacion = []
		cont = 0
		for f in self.datostest:
			prediccion = []
			for r in self.best:
				prediccion = self.conteo_aciertos(prediccion, r, f)
			if prediccion:
				clasificacion.append(np.argmax(np.bincount(prediccion)))
		return clasificacion

# from EstrategiaParticionado import ValidacionSimple, ValidacionCruzada
# from Datos import Datos
# dataset = Datos('tic-tac-toe.data')
# estrategia = ValidacionSimple(0.7)
# #estrategia.creaParticiones(dataset.datos, 45)
# ag = Clasificador_AG(dataset=dataset, estrategia=estrategia)
# #ag.entrenamiento(datostrain, dataset.nominalAtributos, dataset.diccionarios)
# #prediccion = ag.clasifica(datostest, dataset.nominalAtributos, dataset.diccionarios)
# #seed = np.random.seed()
# error = ag.validacion(estrategia, dataset, ag)
# print("Error: ", error)
# #print("Prediccion", prediccion)