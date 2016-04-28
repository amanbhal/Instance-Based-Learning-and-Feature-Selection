import heapq
import math
class DW:
    training = []
    feature_ignore = []
    prop = []
    def __init__(self,training,feature_ignore,prop):
        self.training = training
        self.feature_ignore = feature_ignore
        self.prop = prop

    def performDW(self,input):
        distTup = []
        for line in self.training:
            dist = self.calcDist(line,input)
            distTup.append((dist,line))
        return distTup

    def calcDist(self,line,input):
        dist = 0
        feature_selected = [x for x in range(len(line)) if x not in self.feature_ignore]
        discreteIndexes = self.prop[1]
        for f in feature_selected:
            if f not in discreteIndexes:    #EUCLEDIAN DISTANCE
                dist += (float(line[f])-float(input[f]))**2
            else:   #HAMMING DISTANCE
                if line[f]!=input[f]:       #ERROR SCOPE
                    dist += 1
        dist = math.sqrt(dist)
        return dist