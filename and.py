# let's try to train the and operator

import numpy as np

# Data of and function
x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = [1, 0, 0, 1]

n = 4


# Since the activation function will need positive results at both ends of (1, 1) and (0, 0) we square it.


f = lambda z: np.tanh(z) ** 2
df = lambda z: 2 * np.tanh(z) * (1/(np.cosh(z)**2))

'''
since we are working with multiple inputs coming into one neuron consider the following formula for neuron activation:

a1 = f(w0a00 + w1a01 + b)
z1 = w0a00 + w1a01 + b
represent w as a vector and a as a vector:
z1 = w @ a0 + b

C = (a1 - y) ^ 2
dC/da1 = 2 * (a1 - y)
da1/dz1 = df(z1)
dz1/dw = a0 

'''

w = np.random.randn(2)
b = np.random.random()

def Jw(x, y):
    z1 = w @ x + b # should be scalar
    a1 = f(z1) # still scalar
    return (((2 * a1 - 2 * y) * df(z1)) * x) / len(x)

def Jb(x, y):
    z1 = w @ x + b
    a1 = f(z1)
    return (2 * a1 - 2 * y) * df(z1) / len(x)

aggression = 3.5

for i in range(100):
    c_b, c_w = 0, 0
    for j in range(n):
        c_b += Jb(x[j], y[j])
        c_w += Jw(x[j], y[j])
    
    w = w - ((c_w / n) * (1 + np.random.random()) * aggression)
    b = b - ((c_b / n) * (1 + np.random.random()) * aggression)

def neuron(x):
    return f(w @ x + b)

print('Weight:', w)
print('Bias:', b)
print('\n\n')
for i in range(n):
    print(x[i], neuron(x[i]))
