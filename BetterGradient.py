import numpy as np
import utils as u

#points = [(np.array([2]), 4), (np.array([4]), 2)]
#d = 1 ## FUNCKING NEEDS TO BE ONE. dont know why (I think it has to be with dimentions of something)
# oh ok, cause its a liear multiplication like in like 14 comment (fuck you)

#creatingArtificial data
true_weights = np.array([1, 2, 3, 4, 5])
d = len(true_weights)
points = []

for i in range(10000):
    x = np.random.randn(d) # Creating a (random) d long array
    y = true_weights.dot(x) + np.random.randn() #x1*w1 + x2*w2 .... + noise
    points.append((x, y))
    

# Models: what do we compute


def F(w):#TrainLoss (i guess)
    return sum((w.dot(x) - y)**2 for x, y in points) / len(points)


def dF(w):
    return sum(2*(w.dot(x) - y) * x for x, y in points) / len(points)


def sF(w, i):#Sthocastic
    x, y = points[i]
    return u.loss(x, y, w) 


def sdF(w, i):
    x, y = points[i]
    return 2 * (w.dot(x) - y) * x

 
# ------------------------------------------------------------------------------------------------------

# Algorithm: How do we compute


# Gradient descent
def gradientDecent(F, dF, d):
    w = np.zeros(d)
    step_size = 0.001
    for t in range(5000):
        value = F(w)
        gradient = dF(w) # inclinig of the traniLoss Function
        w = w - step_size * gradient
        print("iteration {}: w = {}, F(w) = {}, gradient = {}".format(t, w, value, gradient ))


# Gradient descent
def sthocasticGradientDecent(sF, sdF, d, n):
    w = np.zeros(d)
    step_size = 1
    numUpd = 0
    for t in range(1000):
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i) 
            numUpd += 1
            step_size = 1.0 / numUpd
            w = w - step_size * gradient
        print("iteration {}: w = {}, F(w) = {}, gradient = {}".format(t, w, value, gradient ))

#gradientDecent(F, dF, d)
sthocasticGradientDecent(sF, sdF, d, len(points))