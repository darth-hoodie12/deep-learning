import numpy as np

A=np.array([[1, 2], [3, 4]])
B=np.array([[4, 3], [2, 1]])
C=np.array([1, 2])

print(A)
print(B)
print(C)

print(A*B) #원소별 곱셉

print(A.dot(B)) #행렬 곱셈

print(A @ B)

print(A*C)

print(A.dot(C))

print(A @ C)