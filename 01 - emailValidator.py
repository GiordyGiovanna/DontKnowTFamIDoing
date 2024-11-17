import utils

## features
# length > 10
# fracOfAlpha
# contains_@
# endswith_.com
# endswith_.org

feature_vector = [1, 0.85, 1, 1, 0]

## weights
# length > 10
# fracOfAlpha
# contains_@
# endswith_.com
# endswith_.org
weight_vector = [-1.2, 0.6, 3, 2.2, 1.4]

print(utils.score(feature_vec = feature_vector, weigth_vec = weight_vector))

print("Printing sign:", [utils.sign(x, y) for x, y in zip(feature_vector, weight_vector)])

