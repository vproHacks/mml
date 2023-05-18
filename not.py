# let's try to train a singular neuron to emulate the not function using backpropagation

from math import cosh, exp
import random

# activation function and derivative of activation function
f = lambda z: 1 / (1 + exp(-z))
df = lambda z: cosh(z/2) ** (-2) / 4

# train for the values now, start at some arbritary weights and biases

w = random.random()
b = random.random()

# represent a jacobian that shows how changing the weight will affect the cost (dC / dw)
def jacobian_weight(x, y):
    z1 = w * x + b
    a1 = f(z1)
    # dC/dw = dC/da1 * da1/dz1 * dz1/dw
    return (((2 * a1 - 2 * y) * df(z1)) * x)

# represent a jacobian that shows how changing the bias will affect cost (dC / db)
def jacobian_bias(x, y):
    z1 = w * x + b
    a1 = f(z1)
    # dC/db = dC/da1 * da1/dz1 * dz1/db
    return (2 * a1 - 2 * y) * df(z1) # * 1

'''
TO approach the optimal solution consider that dC/dw and dC/db want to be minimized as much as possible
that is that the derivative approaches 0 towards the minima of the cost parabola:

C = (untrained_output - desired_output) ^ 2

consider that the value of dC/db over all trained data will represent an average of how much difference / cost
there is between the current bias and the desired output. As such this value will approach 0 when the solution is close,
implying less variance in the bias (b) variable.

The same can be said for dC/dw: c_w -> 0 which means less impact will be taken from the weight

Since the jacobian will return the direction towards the positive slope we will target gradient descent and invert it

by this extension the weight/bias can be calculated at each point as:

weight = weight - (average of all dC/dw in dataset) * variance, where variance must be >= 1 to not approach 0
bias = bias - (average of all dC/db in dataset) * variance
'''

n = 2
aggression = 1
x, y = [[1, 0], [0, 1]]

for i in range(100):
    c_b, c_w = 0, 0
    for j in range(n):
        c_b += jacobian_bias(x[j], y[j])
        c_w += jacobian_weight(x[j], y[j])
    w = w - (((c_w / n) * (1 + random.random())) * aggression)
    b = b - (((c_b / n) * (1 + random.random())) * aggression)

print('Weight:', w)
print('Bias:', b)

# calculate with the W and B at hand
def neuron(x):
    return f(w * x + b)

print('Desired Output:\nnot(0) = 1\nnot(1) = 0')
print(f'Current Output:\nneuron(0) = {neuron(0)}\nneuron(1) = {neuron(1)}')