import numpy as np

points = [(np.array([2]), 4), (np.array([4]), 2)]
d = 2

# Models: what do we compute

def F(w):
    return sum((w.dot(x) * x - y)**2 for x, y in points)


def dF(w):
    return sum(2*(w.dot(x) * x - y) * x for x, y in points)



# ------------------------------------------------------------------------------------------------------

# Algorithm: How do we compute

# Gradient descent
def gradientDecent(F, dF, d):
    w = np.zeros(d)
    step_size = 0.01
    for t in range(100):
        value = F(w)
        gradient = dF(w)
        w = w - step_size * gradient
        print("iteration {}: w = {}, F(w) = {}, gradient = {}".format(t, w, value, gradient ))

gradientDecent(F, dF, d)