import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os



def f(imagenes, b , w):
    res = 0
    ## chequear si estamos trayendo bien las columnas/filas de la imagen
    for i in imagenes:
        t_0 = np.tanh((b + (w).dot(i[0])))
        t_1 = (((1 + t_0) / 2) - i[1])
        res += (t_1 ** 2)
    return res

def fTest(imagen, b , w):
    res = (np.tanh(b + (w).dot(imagen)) + 1) / 2
    return res


def devParcialW(b, d, i, w):
    if isinstance(b, np.ndarray):
        dim = b.shape
        assert dim == (1, )
    if isinstance(d, np.ndarray):
        dim = d.shape
        assert dim == (1, )
    assert isinstance(i, np.ndarray)
    dim = i.shape
    assert len(dim) == 1
    i_rows = dim[0]
    assert isinstance(w, np.ndarray)
    dim = w.shape
    assert len(dim) == 1
    w_rows = dim[0]
    assert w_rows == i_rows

    t_0 = np.tanh((b + (w).dot(i)))
    t_1 = (((1 + t_0) / 2) - d)
    gradient = (((1 - (t_0 ** 2)) * t_1) * i)

    return gradient


def devParcialB(b, d, i, w):
    if isinstance(b, np.ndarray):
        dim = b.shape
        assert dim == (1, )
    if isinstance(d, np.ndarray):
        dim = d.shape
        assert dim == (1, )
    assert isinstance(i, np.ndarray)
    dim = i.shape
    assert len(dim) == 1
    i_rows = dim[0]
    assert isinstance(w, np.ndarray)
    dim = w.shape
    assert len(dim) == 1
    w_rows = dim[0]
    assert w_rows == i_rows

    t_0 = np.tanh((b + (w).dot(i)))
    t_1 = (((1 + t_0) / 2) - d)
    gradient = ((1 - (t_0 ** 2)) * t_1)

    return gradient




def descensoXgradiente(w ,b ,imagenes , alpha, tolerancia, max_iter):
    xtW = w
    xtB = b
    i = 0
    flagB = 0
    flagW = 0
    while i <= max_iter:
        #print("Iteración: ", i, "- Mínimo alcanzado hasta el momento: ", fx)

        # Calculamos el gradiente y actualizamos los valores de w y b
        gradienteW = 0
        gradienteB = 0
        for imagen in imagenes:
            # imagen[0] tiene la imagen en si
            # imagen[1] tiene el diagnostico conocido
            gradienteW += devParcialW(b, imagen[1], imagen[0], xtW)
            gradienteB += devParcialB(b, imagen[1], imagen[0], xtB)
        xtW_sig = xtW - alpha * gradienteW
        xtB_sig = xtB - alpha * gradienteB

        # Chequeamos si ya alcanzamos la convergencia en ambos parametros 
        ### experimentacion: fijarse que es mejor si cada parametro por separado o juntos
        if abs(f(imagenes, b ,xtW_sig) - f(imagenes, b, w)) < tolerancia:
            flagW = 1
        if abs(f(imagenes, xtB_sig, w)-  f(imagenes, b, w)) < tolerancia:
            flagB = 1
          
        if flagW == 1 and flagB == 1:
            break

        xtW = xtW_sig
        xtB = xtB_sig
        i += 1
    return xtW, xtB


def train(imagenes, tolerancia, max_iter, alpha):
    # Inicialización aleatoria de w
    w = np.random.randn(imagenes[0][0].shape[0])  
    # Inicialización de b, puede ser cero
    b = 0.0  
    ## ver de donde sacamos el diagnostico
    w,b = descensoXgradiente(w, b,imagenes,alpha, tolerancia, max_iter)
    return w, b
    