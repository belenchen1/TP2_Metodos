from funciones import *
import matplotlib.pyplot as plt
import math
import time

# loss_train = normalizar_vector(error_cuadratico('train'))
inicio = time.time()
optimo, loss_test = error_cuadratico('test', 4)
print("no-normalizado: ",loss_test[-1])
loss_test = normalizar_vector(loss_test)
fin = time.time()
print(f"{fin - inicio} seg")
print('normalizado', loss_test[-1])


# plt.plot(loss_train, label='Train Loss')
plt.plot(loss_test, label='Test Loss')
plt.xlabel('iteraciones')
plt.ylabel('Loss')
plt.legend()
plt.show()

matriz = matriz_confusion("test", 4, optimo[0], optimo[1])
# [0][0]=true positive, [0][1]=false positive, [1][0]=false negative, [1][1]=true negative
print('true-positive: ', matriz[0][0])
print('false-positive: ', matriz[0][1])
print('false-negative: ', matriz[1][0])
print('true-negative: ', matriz[1][1])
