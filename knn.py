import heapq
import math
class Knn:
    training = []
    feature_ignore = []
    k = None
    prop = []
    def __init__(self,training,feature_ignore,k,prop):
        self.training = training
        self.feature_ignore = feature_ignore
        self.k = k
        self.prop = prop

    def performKnn(self,input):
        maxHeap = []
        for line in self.training:
            dist = self.calcDist(line,input)
            if len(maxHeap)>=self.k:
                top = -maxHeap[0][0]
                if top>dist:
                    heapq.heappop(maxHeap)
                    heapq.heappush(maxHeap,(-dist,line))
            else:
                heapq.heappush(maxHeap,(-dist,line))
        return maxHeap

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