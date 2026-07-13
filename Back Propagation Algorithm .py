import numpy as np

# Inputs and target
X = np.array([0, 1])   # X1=0, X2=1
target = 1
lr = 3

# Weights
w13, w14 = 0.5, -0.3
w24, w25 = 0.2, 0.5
w35, w45 = 0.1, 0.3

# Biases
b3, b4, b5 = 0.6, -0.4, 0.8

# Activation (sigmoid)
def sigmoid(x): return 1/(1+np.exp(-x))
def d_sigmoid(y): return y*(1-y)

# Forward pass
net3 = X[0]*w13 + b3
net4 = X[0]*w14 + X[1]*w24 + b4
out3, out4 = sigmoid(net3), sigmoid(net4)

net5 = out3*w35 + out4*w45 + X[1]*w25 + b5
out5 = sigmoid(net5)

# Error
error = target - out5

# Backpropagation
delta5 = error * d_sigmoid(out5)
delta3 = d_sigmoid(out3) * delta5 * w35
delta4 = d_sigmoid(out4) * delta5 * w45

# Update weights
w35 += lr * delta5 * out3
w45 += lr * delta5 * out4
w25 += lr * delta5 * X[1]

w13 += lr * delta3 * X[0]
w14 += lr * delta4 * X[0]
w24 += lr * delta4 * X[1]

# Update biases
b3 += lr * delta3
b4 += lr * delta4
b5 += lr * delta5

print("Updated weights:", w13, w14, w24, w25, w35, w45)
print("Updated biases:", b3, b4, b5)