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

# lagrangiano
def L(x_t, i, d):
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

def dl_w(w,b, i:list[list], d:list):
   # derivada del lagrangiano contra w
   idx = 0
   sum = 0
   for imagen in i:
      t_0:float = math.tanh(b+w@imagen) 
      sum += (1-t_0**2)*(((t_0+1)/2-d[idx]))*imagen
      idx += 1 
   return sum

def dl_b(w,b, i:list[list], d:list):
   # derivada del lagrangiano contra b
   sum = 0
   idx = 0
   for imagen in i:
      t_0:float = math.tanh(b+w@imagen) 
      sum += (1-t_0**2)*(((t_0+1)/2-d[idx]))
      idx += 1 
   return sum

def gradiente(x_t,i:list[list],d:list):
   # gradiente tiene una tupla: derivada del error contra b, derivada del error contra w.
   w = x_t[0]
   b = x_t[1]
   res = [dl_w(w,b,i,d), dl_b(w,b,i,d)]
   return res

def descenso_por_gradiente(i, d, alpha=0.001, MAX_ITER=1000):
   # función a optimizar
   np.random.seed(42)
   K = len(i[0])
   w_0 = np.random.randn(K) # w es un vector de R^K tq K=cant de píxeles en una imagen. Inicialmente es un vector aleatorio.
   b_0 = 0.0
   x_t = (w_0, b_0)
   loss = [] 
   iter = 0
   while iter < MAX_ITER:
      x_tsig = [x_t[0] - (alpha * gradiente(x_t, i, d)[0]), x_t[1] - (alpha * gradiente(x_t, i, d)[1])]
      loss.append(L(x_tsig, i, d))
      # if abs(L(x_tsig, i, d) - L(x_t, i, d)) < TOLERANCIA and L(x_tsig, i, d)<=0.05:
      #    break 
      x_t = x_tsig
      iter = iter + 1
   return x_tsig, loss

def error_cuadratico(conjunto:str, escala):
   
   if conjunto != "test" and conjunto != "train":
      raise Exception ("'conjunto' tiene que ser 'test' o 'train'")
   
   path_normal = './chest_xray/' + conjunto + '/NORMAL'
   path_pneumonia = './chest_xray/' + conjunto + '/PNEUMONIA'
   normal_test = abrirImagenesEscaladas(path_normal, escala)
   pneumonia_test = abrirImagenesEscaladas(path_pneumonia, escala)
   img, d = balancear_datos(normal_test, pneumonia_test)
   optimo, loss = descenso_por_gradiente(img, d)
   
   return optimo, loss

def normalizar_vector(vector):
   m = 0
   vector_normalizado = []
   for elem in vector:
      m += elem**2
   for elem in vector:
      vector_normalizado.append(elem/(math.sqrt(m)))
   return vector_normalizado

def diagnostico(imagen, w, b):
   t_0: float = math.tanh(w @ imagen + b)
   res = (t_0 + 1) / 2
   if res >= 0.5:
      return 1
   else:
      return 0

def matriz_confusion(conjunto, escala, w ,b):
   if conjunto != "test" and conjunto != "train":
      raise Exception ("'conjunto' tiene que ser 'test' o 'train'")
   path_normal = './chest_xray/' + conjunto + '/NORMAL'
   path_pneumonia = './chest_xray/' + conjunto + '/PNEUMONIA'
   normal_test = abrirImagenesEscaladas(path_normal, escala)
   pneumonia_test = abrirImagenesEscaladas(path_pneumonia, escala)
   n = min(len(normal_test), len(pneumonia_test))
   pneumonia_test = pneumonia_test[:n]
   normal_test = normal_test[:n]

   imagenes_con = [] 
   imagenes_sin = []
   
   for imagen in normal_test:
      imagenes_sin.append(diagnostico(imagen,w,b))
   for imagen in pneumonia_test:
      imagenes_con.append(diagnostico(imagen,w,b))
   
   matriz =[[0,0],[0,0]]   # [tiene_enferemedad][diagnostico] 
                           # [0][0]=true positive, [0][1]=false positive, [1][0]=false negative, [1][1]=true negative
                           
   matriz[0][0]=imagenes_con.count(1)/n
   matriz[0][1]=imagenes_sin.count(1)/n
   matriz[1][0]=imagenes_con.count(0)/n
   matriz[1][1]=imagenes_sin.count(0)/n
   
   return matriz
