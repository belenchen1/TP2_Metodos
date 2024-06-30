from funciones import *
import matplotlib.pyplot as plt
import math
import time

# loss_train = normalizar_vector(error_cuadratico('train'))
inicio = time.time()
optimo, loss_test = error_cuadratico('test', 512)
loss_test = normalizar_vector(loss_test)
fin = time.time()
print(f"{fin - inicio} seg")
print(loss_test[-1])

# plt.plot(loss_train, label='Train Loss')
plt.plot(loss_test, label='Test Loss')
plt.xlabel('iteraciones')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
true_negative, false_negative, true_positive, false_positive = matriz_confusion("test", 64, optimo[0], optimo[1])
print("porcentaje true-negative =",true_negative)
print("porcentaje false-negative =",false_negative)
print("porcentaje true-positive =",true_positive)
print("porcentaje false-positive =",false_positive)
'''