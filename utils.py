import numpy as np # np like not paying my taxes, kidding


# How confident are we predicting
def score_total(w, fi):
    return w * fi


def score(feature_vec, weigth_vec):
    return sum (x * y for x, y in zip(feature_vec, weigth_vec))


def sign(x, y):
    return (x * y > 0) - (x * y < 0) # Yes of course, why not? You can't understand it????? Me neither, it just work


# x output
# y correct output 
# w weight
def loss(x, y, w):
    return (w.dot(x) - y)**2


# How correct we are
def margin(w, fi, y):
    return score_total(w, fi) * y


def loss01(x, y, w):
    return 1 if (sign(x, y) <= y) else 0