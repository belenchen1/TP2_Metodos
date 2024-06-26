import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import math

def abrirImagenesEscaladas(carpeta, escala=64):
   # abre todas las imagenes de la carpeta, y las escala de tal forma que midan (escala x escala)px
   # devuelve las imagenes aplanadas -> vectores de tamano escala^2 con valores entre 0 y 1
   imagenes = []
   for dirpath, dirnames, filenames in os.walk(carpeta):
      
      for file in filenames:
         img = Image.open( os.path.join(carpeta, file) )
         img = img.resize((escala, escala))
         img.convert('1')
         img = np.asarray(img)
         if len(img.shape)==3:
               img = img[:,:,0].reshape((escala**2 )) / 255
         else:
               img = img.reshape((escala**2 )) / 255
         
         imagenes.append( img )
      
   return imagenes

def balancear_datos(normal, pneumonia):
   n = min(len(normal), len(pneumonia))
   pneumonia = pneumonia[:n]
   normal = normal[:n]
   d = ([1] * n) + ([0] * n)
   
   imagenes_entrenamiento_balanceadas = (pneumonia+normal)
   
   return imagenes_entrenamiento_balanceadas, d

#!###########################################################################################################

# def f(w:list, b:int, i:list[list]): # funcion normal. f 
#    return (math.tanh(w@i + b) + 1)/2


# lagrangiano
def f(x_t, i, d):
   # funcion a minimizar
   w = x_t[0]
   b = x_t[1]
   sum = 0
   idx = 0
   for imagen in i:
      t_0:float = math.tanh(w@imagen+b) 
      sum += ((t_0 + 1) / 2 - d[idx])**2
      idx += 1
   return sum

def dfw(w,b, i:list[list], d:list):
   # derivada del lagrangiano contra w
   idx = 0
   sum = 0
   for imagen in i:
      # t_0:float = math.tanh(b+w@imagen) 
      # sum += (1-t_0**2)*(((t_0+1)/2-d[idx]))*imagen
      t_0 = np.tanh((b + (w).dot(imagen)))
      t_1 = (((1 + t_0) / 2) - d[idx])
      sum += (((1 - (t_0 ** 2)) * t_1) * imagen)
      idx += 1 
   return sum

def dfb(w,b, i:list[list], d:list):
   # derivada del lagrangiano contra b
   sum = 0
   idx = 0
   for imagen in i:
      t_0:float = math.tanh(b+w@imagen) 
      sum += (1-t_0**2)*(((t_0+1)/2-d[idx]))
      idx += 1 
   return sum

def gradiente(x_t,i:list[list],d:list):
   # x es un punto de i_j (imagen aplanada = vector de tamaño escala*escala) (i: conjunto de imagenes)(i_j:imagen j del conjunto)
   # gradiente tiene una tupla: derivada de f contra b, derivada de f contra w.
   
   w = x_t[0]
   b = x_t[1]
   res = [dfw(w,b,i,d), dfb(w,b,i,d)]
   return res

# x_0 es un punto aleatorio en Rn
# x_0 va a tener un valor aleatorio de b y un valor aleatorio de w
def descenso_por_gradiente(i, d):
   # función a optimizar
   K = len(i[0])
   # ? Multiplicar por 0.01 para reducir la escala de los valores (???)
   w_0 = np.random.randn(K) # w es un vector de R^K tq K=cant de píxeles en una imagen
   b_0 = 0.0
   x_t = (w_0, b_0)
   alpha = 0.0001
   TOLERANCIA = 0.0001
   MAX_ITER = 1000
   iter = 0
   while iter < MAX_ITER: # necesitamos esto porque no tenemos x_tsig al pcpio
      x_tsig = [x_t[0] - (alpha * gradiente(x_t, i, d)[0]), x_t[1] - (alpha * gradiente(x_t, i, d)[1])]
      loss = [] 
      loss.append(f(x_tsig, i, d))
      if abs(f(x_tsig, i, d) - f(x_t, i, d)) < TOLERANCIA:
         break 
      x_t = x_tsig
      iter = iter + 1
   return x_tsig, loss

def error_cuadratico(conjunto:str):
   
   if conjunto != "test" and conjunto != "train":
      raise Exception ("'conjunto' tiene que ser 'test' o 'train'")
   
   path_normal = './chest_xray/' + conjunto + '/NORMAL'
   path_pneumonia = './chest_xray/' + conjunto + '/PNEUMONIA'
   normal_test = abrirImagenesEscaladas(path_normal, 64)
   pneumonia_test = abrirImagenesEscaladas(path_pneumonia, 64)
   img, d = balancear_datos(normal_test, pneumonia_test)
   opt, loss = descenso_por_gradiente(img, d)
   error = f(opt, img, d)
   
   return loss