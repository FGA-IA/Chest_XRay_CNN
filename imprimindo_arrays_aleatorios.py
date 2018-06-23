import numpy as np
import random
import os
import asyncio
import sys

def inserePesosConv2d(lista, j, k, h):
	qtdPesos = 32
	g = 0
	while(g < qtdPesos):
		peso = random.uniform(-1.0, 1.0)
		lista[j][k][h].append(peso)
		g += 1
	return lista

def montaConv2d_1():
	camada = list()
	lista = list() ## no primeiro caso deve ter 3 posições, no segundo caso deve ter 32 posições

	descendente = 2 ## quantidade de elementos na camada

	i = 0
	while(i < descendente):
		lista = []
		if( i == 0):
			tamanhoSubCamada = 3
			j = 0
			while(j < tamanhoSubCamada):
				subLista = list()
				lista.append(subLista)

				tamanhoSubSubCamada = 3
				k = 0
				while(k < tamanhoSubSubCamada):
					subsubLista = list()
					lista[j].append(subsubLista)

					tamanhoSubSubSubcamada = 3
					h = 0
					while(h < tamanhoSubSubSubcamada):
						subsubsubLista = list()
						lista[j][k].append(subsubsubLista)

						lista = inserePesosConv2d(lista, j, k, h)

						h += 1
					k += 1
				j += 1
						
		elif(i == 1):
			qtdPesos = 32
			j = 0
			while(j < qtdPesos):
				lista.append(0.0)
				j += 1
		else:
			print("Não pode ter mais que duas posições")
		
		camada.append(np.array(lista, dtype='float32'))
		i += 1
	return camada



def montaCamada(arraysNaCamada, qtdPorArray):
	camada = list()
	return camada 

def montaBatch_normalization(arraysNaCamada, qtdPorArray):
	camada = list()
	if(not arraysNaCamada == 0):
		i = 0
		lista = None
		
		while(i < arraysNaCamada):
			lista = list()
			
			pesoAtual = 0
			while(pesoAtual < qtdPorArray[i]):
				if(i == 0 or i == 3):
					numero = 1.0
				else:
					numero = 0.0

				lista.append(numero)
				pesoAtual += 1

			camada.append(np.array(lista, dtype='float32'))
			i = i + 1

	return camada 

def montaPesoNaDensa(lista):
	qtdPesos = 32768
	g = 0
	while(g < qtdPesos):
		sub = []
		peso = random.uniform(-1.0, 1.0)
		sub.append(peso)
		lista.append(sub)
		g += 1
	return lista

def montaDensa(arraysNaCamada, qtdPorArray):
	camada = list()
	if(not arraysNaCamada == 0):
		i = 0
		lista = None
		
		while(i < arraysNaCamada):
			lista = []	
			DEFAULT = 0.0
			if(i == 0):
				result = montaPesoNaDensa(lista)
			elif(i == 1):
				peso = DEFAULT
				lista.append(peso)
			else:
				print("Nao era pra cair aqui")

			i = i + 1
			camada.append(np.array(lista, dtype='float32'))
	
	return camada 



arraysNaCamada = None
qtdPorArray = None


array0 = montaConv2d_1() 

array1 = montaCamada(0, []) # é vazio

array2 = montaBatch_normalization(4, [32,32,32,32])

array3 = montaCamada(0, []) # é vazio

array4 = montaCamada(0, []) # é vazio

array5 = montaDensa(2, [32768, 1])

array6 = montaBatch_normalization(4, [1,1,1,1])


print("conv2d_1 (Conv2D): ")

print(array0)

print("activation_1 (Activation): ")
print(array1)

print("batch_normalization_1 (Batch: ")
print(array2)

print("max_pooling2d_1 (MaxPooling2: " + ''.join(array3))
print(array3)

print("flatten_1 (Flatten): "+ ''.join(array4))
print(array4)

print("dense_1 (Dense): ")
print(array5)

print("batch_normalization_2 (Batch: ")
print(array6)
