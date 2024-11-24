import submission, util ## OKKKKKKKKKKK buddy, not strage.....
from collections import defaultdict

#readExamples
trainExp = util.readExamples("yeah.ImNotPushingThisFile") # I FUCKING DO, not pushing it 
devExp = util.readExamples("yeah.ImNotPushingThisFile")

def featureExtractor(x): #phi(x)
    phi = defaultdict(float)
    tokens = x.split()
    left, entity, reght = tokens[0], tokens[1:-1], tokens[-1]
    phi['entity is ' + ' '.join(entity)] = 1
    return phi
# yeah bro, im fucking scared

weights = submission.learnPredictor(trainExp, devExp, featureExtractor)
util.outputWeigths(weights, "weights")
util.outputErrorAnalysis(devExp, featureExtractor, weights, 'Error-analysis')