from funciones import *
import matplotlib.pyplot as plt
import math

loss_train = normalizar_vector(error_cuadratico('train'))
loss_test = normalizar_vector(error_cuadratico('test'))

plt.plot(loss_train, label='Train Loss')
plt.plot(loss_test, label='Test Loss')
plt.xlabel('iteraciones')
plt.ylabel('Loss')
plt.legend()
plt.show()

