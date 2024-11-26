import submission, utils ## OKKKKKKKKKKK buddy, not strage at all.....
from collections import defaultdict

# ok, thats what I understood: we are creating a function that defins feature vector phi, not sure how it fucking works, on it

#readExamples
trainExp = utils.readExamples("../Resources/names.train") # I FUCKING DO HAVE IT, not pushing it 
devExp = utils.readExamples("../Resources/names.dev")

def featureExtractor(x): #phi(x)
    phi = defaultdict(float)
    tokens = x.split()
    left, entity, right = tokens[0], tokens[1:-1], tokens[-1]
    phi['entity is ' + ' '.join(entity)] = 1
    phi['entity left is ' + left] = 1
    phi['entity right is ' + right] = 1
    for word in entity:
        phi['word is ' + word] = 1
    return phi
# yeah bro, im fucking scared

weights = submission.learnPredictor(trainExp, devExp, featureExtractor)
utils.outputWeigths(weights, "weights")
utils.outputErrorAnalysis(devExp, featureExtractor, weights, 'Error-analysis')