from funciones import *

normal = abrirImagenesEscaladas('./chest_xray/train/NORMAL', 64)
print('ejecuté normal')
pneumonia = abrirImagenesEscaladas('./chest_xray/train/PNEUMONIA', 64)
print('ejecuté pneumonía')

imagenes_entrenamiento_balanceadas, d = balancear_datos(normal, pneumonia)

diagnost = descenso_por_gradiente(imagenes_entrenamiento_balanceadas, d)

print (diagnost)






