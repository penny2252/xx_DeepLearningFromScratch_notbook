import numpy as np
import matplotlib.pyplot as plt
x=np.array([2,-2,1,0])
y=x>0
print(y)
y=y.astype(np.int)
print(y)
y=y.astype(np.bool)
print(y)
x=x.astype(np.bool)
print(x)
x=x.astype(np.int)
print(x)

print('-'*10)
def step_functon(x):
    return np.array(x>0,dtype=np.int)
x=np.arange(-5,5,0.1)
y1=step_functon(x)
plt.plot(x,y1)
# plt.ylim(-0.1,1.1)
# plt.show()

print('-'*10)
def sigmoid(x):
    return 1/(1+np.exp(-x))
y2=sigmoid(x)
plt.plot(x,y2)
# plt.ylim(-0.1,1.1)
# plt.show()
print('-'*10)

def relu(x):
    return np.maximum(0,x)
y3=relu(x)/5
plt.plot(x,y3)
plt.ylim(-0.1,1.1)
plt.show()


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

