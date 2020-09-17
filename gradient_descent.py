import numpy as np

def get_matrices(x, y):
    x=np.array(x, dtype=float) # 만약 numpy array가 아닌 경우, numpy array로 변환
    y=np.array(y, dtype=float)
    
    N=x.shape[0] # Data point의 수 N=4
    X=np.column_stack([x, np.ones(N)]) # 전부 1의 값을 갖는 column을 덧붙인다.
    #print(X)
    
    XTX=X.T.dot(X)/N # Matrix 곱은 dot 연산 또는 @ 연산 사용!
    XTy=X.T.dot(y)/N
    
    return XTX, XTy
    
def update(w, XTX, XTy, lam):
    grad=XTX.dot(w)-XTy
    w=w-lam*grad
    return w

def get_mse(x, y, w):
    x=np.array(x, dtype=float) # 만약 numpy array가 아닌 경우, numpy array로 변환
    y=np.array(y, dtype=float)
    w=np.array(w, dtype=float)
    
    N=x.shape[0] # Data point의 수 N=4
    X=np.column_stack([x, np.ones(N)]) # 전부 1의 값을 갖는 column을 덧붙인다.
    Xw=X.dot(w)
    v=y-Xw
    e=v.dot(v)/N
    
    return e

x1=np.array([2, 4, 6, 8], dtype=float) # 공부한 시간
x2=np.array([0, 4, 2, 3], dtype=float) # 과외 수업 횟수
y=np.array([81, 93, 91, 97], dtype=float) # 점수

x=np.vstack([x1, x2]) # x1, x2를 수직으로 쌓아서 하나로 만든다.
x=x.T # Transpose를 해서 Nx2 행렬로 만든다.        

MAX_ITER=5000 # Epoch의 수
lam=0.01 # 학습률
w0=np.random.randn(2)
print(w0)
XTX, XTy = get_matrices(x1, y)
w=w0
e=get_mse(x1, y, w)
print("Initial")
print("Parameters={}, MSE={}".format(w, e))
for i in range(MAX_ITER):
    w=update(w, XTX, XTy, lam)
    if i%1000==999:
        e=get_mse(x1, y, w)
        print("{}th epoch.".format(i+1))
        print("Parameters={}, MSE={}".format(w, e))
        
w0=np.random.randn(3)
print(w0)
XTX, XTy = get_matrices(x, y)
w=w0
e=get_mse(x, y, w)
print("Initial")
print("Parameters={}, MSE={}".format(w, e))
for i in range(MAX_ITER):
    w=update(w, XTX, XTy, lam)
    if i%1000==999:
        e=get_mse(x, y, w)
        print("{}th epoch.".format(i+1))
        print("Parameters={}, MSE={}".format(w, e))

