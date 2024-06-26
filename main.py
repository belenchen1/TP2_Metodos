from funciones import *

loss_train = error_cuadratico('train')
loss_test = error_cuadratico('test')

import matplotlib.pyplot as plt

plt.plot(loss_train, label='Train Loss')
plt.plot(loss_test, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# print(f'loss train {loss_train}')
# print(f'loss test {loss_test}')