import random
import heapq
from sys import *
from knn import *
from collections import defaultdict
import thread
from distanceWeighting import *

def randomizeData(data):
    randomData = []
    while len(data)!=0:
        randomLine = data.pop(random.randrange(len(data)))
        randomData.append(randomLine)
    return randomData

def calcAccuracy(knn,validationData):
    calculated = 0
    expected = len(validationData)
    for line in validationData:
        maxHeap = knn.performKnn(line)
        classifier = defaultdict(float)
        for i in range(len(maxHeap)):
            if maxHeap[i][0]==0:
               classifier[maxHeap[i][1][targetIndex]] += maxint
            else:
                classifier[maxHeap[i][1][targetIndex]] += 1/(-maxHeap[i][0])
        classifierValue = sorted(classifier,key=classifier.__getitem__,reverse=True)[0]
        if line[targetIndex]==classifierValue:
            calculated += 1
    accuracy = float(calculated)/expected
    return accuracy

def calcAccuracyDW(dw,validationData):
    calculated = 0
    expected = len(validationData)
    for line in validationData:
        list = dw.performDW(line)
        classifier = defaultdict(float)
        for i in range(len(list)):
            if list[i][0]==0:
               classifier[list[i][1][targetIndex]] += maxint
            else:
                classifier[list[i][1][targetIndex]] += 1/((list[i][0])**2)
        classifierValue = sorted(classifier,key=classifier.__getitem__,reverse=True)[0]
        if line[targetIndex]==classifierValue:
            calculated += 1
    accuracy = float(calculated)/expected
    return accuracy

def makeBuckets(data):
    tenPercent = len(data)/10
    bucket = []
    while len(data)>=tenPercent:
        temp = []
        for i in range(tenPercent):
            temp.append(data.pop())
        bucket.append(temp)
    return bucket

def normalize(data,prop):
    #print prop
    minmax = [[maxint,-maxint] for i in range(len(data[0]))]
    continuous = [i for i in range(len(data[0])) if (i not in prop[0] and i not in prop[1])]
    for j in continuous:
        for i in range(len(data)):
            if data[i][j]<minmax[j][0]:
                minmax[j][0] = data[i][j]
            if data[i][j]>minmax[j][1]:
                minmax[j][1] = data[i][j]
    for j in continuous:
        for i in range(len(data)):
            #print data[i][j], minmax[j]
            data[i][j] = int(data[i][j])
            data[i][j] = (float)(data[i][j]-minmax[j][0])/(minmax[j][1]-minmax[j][0])
    return data

def convertToNumeric(data,prop):
    numericIndexes = []
    if prop[2]=='none':
        for i in range(len(data[0])):
            numericIndexes.append(i)
    else:
        for i in range(len(data[0])):
            if i not in prop[2] and i not in prop[0]:
                numericIndexes.append(i)
    for i in range(len(data)):
        for j in numericIndexes:
            data[i][j] = float(data[i][j])
    return data

def calcDataCIupper(classValues):
    dataCIupper = {}
    denominator = 0
    for key in classValues.keys():
        denominator += classValues[key]
    for key in classValues.keys():
        error = float(classValues[key])/denominator
        dataCIupper[key] = error+math.sqrt(((error)*(1-error))/denominator)
    return dataCIupper

def findAllAcceptable(reducedData,classValues,prop):
    targetIndex = prop[0][0]
    acceptable = []
    dataCIupper = calcDataCIupper(classValues)
    for line in reducedData:
        if line[1]+line[2]>0:
            lineAccuracy = float(line[1])/(line[1]+line[2])
            lineCIlower = lineAccuracy-math.sqrt(((lineAccuracy)*(1-lineAccuracy))/(line[1]+line[2]))
            if lineCIlower>dataCIupper[line[0][targetIndex]]:
                acceptable.append(line)
    return acceptable

def calcNearestAcceptableNeighbour(reducedData,line,prop,classValues,feature_ignore):
    acceptable = findAllAcceptable(reducedData,classValues,prop)
    if len(acceptable)==0:
        return None
    x = nn(acceptable,line,feature_ignore,prop)
    nearestAcceptable = x[0][1]
    return nearestAcceptable

def calcClassValues(data,prop):
    classValues = defaultdict(int)
    targetIndex = prop[0][0]
    for line in data:
        classValues[line[targetIndex]] += 1
    return classValues

def nn(data,input,feature_ignore,prop):
    maxHeap = []
    for line in data:
        dist = calcDist(line[0],input,feature_ignore,prop)
        if len(maxHeap)>=1:
            top = -maxHeap[0][0]
            if top>dist:
                heapq.heappop(maxHeap)
                heapq.heappush(maxHeap,(-dist,line))
        else:
            heapq.heappush(maxHeap,(-dist,line))
    return maxHeap

def calcDist(line,input,feature_ignore,prop):
    dist = 0
    feature_selected = [x for x in range(len(line)) if x not in feature_ignore]
    discreteIndexes = prop[1]
    for f in feature_selected:
        if f not in discreteIndexes:    #EUCLEDIAN DISTANCE
            dist += (float(line[f])-float(input[f]))**2
        else:   #HAMMING DISTANCE
            if line[f]!=input[f]:       #ERROR SCOPE
                dist += 1
    dist = math.sqrt(dist)
    return dist

def nnWithRadius(reducedData,point,radius,feature_ignore,prop):
    targetIndex = prop[0][0]
    for line in reducedData:
        if calcDist(line[0],point,feature_ignore,prop)<=radius:
            if line[0][targetIndex]==point[targetIndex]:
                line[1] += 1
            else:
                line[2] += 1
    return reducedData

def updateReducedData(reducedData,point,nearestAcceptableNeighbour,feature_ignore,prop):
    radius = calcDist(point,nearestAcceptableNeighbour[0],feature_ignore,prop)
    reducedData = nnWithRadius(reducedData,point,radius,feature_ignore,prop)
    return reducedData

def calcDataCIlower(classValues):
    dataCIlower = {}
    denominator = 0
    for key in classValues.keys():
        denominator += classValues[key]
    for key in classValues.keys():
        error = float(classValues[key])/denominator
        dataCIlower[key] = error-math.sqrt(((error)*(1-error))/denominator)
    return dataCIlower

def dropNoisyData(reducedData,classValues,prop):
    targetIndex = prop[0][0]
    nonNoisyData = []
    dataCIlower = calcDataCIlower(classValues)
    for line in reducedData:
        accuracy = float(line[1])/(line[1]+line[2])
        ciUpper = accuracy+math.sqrt(((accuracy)*(1-accuracy))/(line[1]+line[2]))
        if ciUpper>=dataCIlower[line[0][targetIndex]]:
            nonNoisyData.append(line)
    return nonNoisyData

def ntGrowth(data,prop,feature_ignore):
    reducedData = [[data[0],0,0]]   #data,trueClassified,falseClassified
    classValues = calcClassValues(data,prop)
    for line in range(1,len(data)-1):
        nearestAcceptableNeighbour = calcNearestAcceptableNeighbour(reducedData,data[line],prop,classValues,feature_ignore)
        if nearestAcceptableNeighbour==None:
            x = nn(reducedData,data[line],feature_ignore,prop)
            nearestAcceptableNeighbour = x[0][1]
        if data[line][prop[0][0]]==nearestAcceptableNeighbour[0][prop[0][0]]:
            reducedData.append(nearestAcceptableNeighbour)
        reducedData = updateReducedData(reducedData,data[line],nearestAcceptableNeighbour,feature_ignore,prop)
        # if line>(20*(len(data))/100):
        #     reducedData = dropNoisyData(reducedData,classValues,prop)
    result = []
    for line in reducedData:
        result.append(line[0])
    return result

def runCVwithDiffK(databuckets,targetIndex,prop,k):
    a = []
    for i in range(10):
        testingData = databuckets[i]
        restOfData = databuckets[:i]+databuckets[i+1:]
        validationData = []
        trainingData = []
        for j in range(9):
            if j<3:
                for line in restOfData[j]:
                    validationData.append(line)
            else:
                for line in restOfData[j]:
                    trainingData.append(line)
        knn = Knn(trainingData+validationData,[targetIndex],k,prop)
        accuracy = calcAccuracy(knn,testingData)
        a.append(accuracy)
    print "Mean Accuracy for K:",k,"is:", float(sum(a))/10
    return float(sum(a))/10

def runCVwithDW(databuckets,targetIndex,prop):
    a = []
    for i in range(10):
        testingData = databuckets[i]
        restOfData = databuckets[:i]+databuckets[i+1:]
        validationData = []
        trainingData = []
        for j in range(9):
            if j<3:
                for line in restOfData[j]:
                    validationData.append(line)
            else:
                for line in restOfData[j]:
                    trainingData.append(line)
        dw = DW(trainingData+validationData,[targetIndex],prop)
        accuracy = calcAccuracyDW(dw,testingData)
        a.append(accuracy)
    print "Mean Accuracy for Distance Weighting is:", float(sum(a))/10

def runCVwithSBE(databuckets,targetIndex,prop):
    o = []
    a = []
    for i in range(10):
        print "******** ITERATION "+str(i+1)+" *************"
        testingData = databuckets[i]
        restOfData = databuckets[:i]+databuckets[i+1:]
        validationData = []
        trainingData = []
        for j in range(9):
            if j<3:
                for line in restOfData[j]:
                    validationData.append(line)
            else:
                for line in restOfData[j]:
                    trainingData.append(line)
        knn = Knn(trainingData,[targetIndex],8,prop)
        originalAccuracy = calcAccuracy(knn,validationData)
        o.append(originalAccuracy)
        #STEPWISE BACKWARD ELEMINATION
        maxAccuracy = -maxint
        features_removed = [targetIndex]
        attributesIndex = [x for x in range(len(validationData[0])) if (x!=targetIndex and x not in features_removed)]
        run = True
        while run:
            localAccuracy = []
            for index in attributesIndex:
                print ".",
                knn1 = Knn(trainingData,[index,targetIndex],8,prop)
                localAccuracy.append((calcAccuracy(knn1,validationData),index))
            print
            localAccuracy = sorted(localAccuracy,key=lambda x: x[0],reverse=True)
            maxAccuracy = localAccuracy[0][0]
            if maxAccuracy<originalAccuracy:
                run = False
            else:
                features_removed.append(localAccuracy[0][1])
                attributesIndex = [x for x in range(len(validationData[0])) if (x!=targetIndex and x not in features_removed)]
        knn2 = Knn(trainingData,features_removed,8,prop)
        accuracy = calcAccuracy(knn,testingData)
        a.append(accuracy)
        print "Original: ", originalAccuracy
        #print "Features Choosen:", [i for i in range(len(data[0])) if i not in features_removed]
        print "After: ", accuracy
    print "********** SUMMARY ***********"
    print "Mean Accuracy Before Feature Selection:", float(sum(o))/10
    print "Mean Accuracy After Feature Selection:", float(sum(a))/10

def runCVwithNT(databuckets,targetIndex,prop):
    o = []
    a = []
    for i in range(10):
        print "******** ITERATION "+str(i+1)+" *************"
        testingData = databuckets[i]
        restOfData = databuckets[:i]+databuckets[i+1:]
        validationData = []
        trainingData = []
        for j in range(9):
            if j<3:
                for line in restOfData[j]:
                    validationData.append(line)
            else:
                for line in restOfData[j]:
                    trainingData.append(line)
        knn = Knn(trainingData,[targetIndex],8,prop)
        originalAccuracy = calcAccuracy(knn,validationData)
        o.append(originalAccuracy)
        #NT Growth
        trainingData = ntGrowth(trainingData+validationData,prop,[prop[0][0]])
        knn2 = Knn(trainingData,[targetIndex],8,prop)
        accuracy = calcAccuracy(knn,testingData)
        a.append(accuracy)
        print "Original: ", originalAccuracy
        print "After: ", accuracy
    print "********** SUMMARY ***********"
    print "Mean Accuracy Before NT Growth:", float(sum(o))/10
    print "Mean Accuracy After NT Growth:", float(sum(a))/10

if __name__ == '__main__':

    # propfile = "iris-prop.txt"
    # datafile = "iris-data.txt"

    propfile = "heart-prop.txt"
    datafile = "HeartDataSet.txt"

    # propfile = "tic_tac_toe_prop.txt"
    # datafile = "tic-tac-toe-data.txt"

    # propfile = "credit_screening-prop.txt"
    # datafile = "credit-screening-data.txt"

    # propfile = "voting-prop.txt"
    # datafile = "voting-data.txt"

    # propfile = "wine-prop.txt"
    # datafile = "wine-data.txt"

    # propfile = "pima-prop.txt"
    # datafile = "pima-indians-diabetes.data.txt"

    with open(propfile) as f:
    #with open("iris-prop.txt") as f:
        prop = f.readlines()
    #prop[0] => target Index
    #prop[1] => discrete attribute index
    #prop[2] => non numeric attribute index
    prop = [line.rstrip('\n').split(',') for line in prop]
    prop[0][0] = int(prop[0][0])
    if prop[1][0]!='none':
        for i in range(len(prop[1])):
            prop[1][i] = int(prop[1][i])
    if prop[2][0]!='none':
        for i in range(len(prop[2])):
            prop[2][i] = int(prop[2][i])
    with open(datafile) as f:
    #with open("iris-data.txt") as f:
        data = f.readlines()
    data = [line.rstrip('\n').split(',') for line in data if "?" not in line]
    data = randomizeData(data)
    data = convertToNumeric(data,prop)
    data = normalize(data,prop)
    classValues = calcClassValues(data,prop)
    print classValues
    databuckets = makeBuckets(data)
    targetIndex = prop[0][0]
    #10 cross validation
    runCVwithNT(databuckets,targetIndex,prop)
    runCVwithDW(databuckets,targetIndex,prop)
    runCVwithSBE(databuckets,targetIndex,prop)
    features = [i for i in range(len(databuckets[0][0])) if i not in prop[0]]
    accuracy = []
    for k in range(1,len(features)+1):
        accuracy.append((runCVwithDiffK(databuckets,targetIndex,prop,k),k))
    accuracy.sort(key=lambda x: x[0])
    print "********** SUMMARY ***********"
    print "Max Accuracy obtained with k=",accuracy[-1][1]