import numpy as np

def least_squares_method(x, y):
    x=np.array(x, dtype=float) # 만약 numpy array가 아닌 경우, numpy array로 변환
    y=np.array(y, dtype=float)
    
    N=x.shape[0] # Data point의 수 N=4
    X=np.column_stack([x, np.ones(N)]) # 전부 1의 값을 갖는 column을 덧붙인다.
    #print(X)
    
    # w=inv(XTX)XTy pseudo inverse 연산 실행
    XTX=X.T.dot(X) # Matrix 곱은 dot 연산 또는 @ 연산 사용!
    XTy=X.T.dot(y)
    XTXi=np.linalg.inv(XTX) # 역행렬 계산
    w=XTXi.dot(XTy)
    
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
print("High-dimensional input={}".format(x)) # 입력이 잘 만들어졌는지 확인

w1=least_squares_method(x1, y) # 공부한 시간과 점수의 관계
print("Weights={}".format(w1))
e=get_mse(x1, y, w1)
print("MSE={}".format(e))

w2=least_squares_method(x, y) # 공부한 시간, 과외 수업 횟수, 점수의 관계
print("Weights={}".format(w2))
e=get_mse(x, y, w2)
print("MSE={}".format(e))


