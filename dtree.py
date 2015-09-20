"""
The Decision Tree Classifier
"""
import numpy as np
import scipy

class DecisionTree(object):


    def __init__(self, depth=None):
        """
        Constructs a Decision Tree Classifier

        @param depth=None : maximum depth of the tree,
                            or None for no maximum depth
        """
        self.treeHead = TreeNode(None)
        self.maxDepth = depth
        self.depth = 0
        self.size = 1
        pass

    def fit(self, X, y, sample_weight=None):
        """ Build a decision tree classifier trained on data (X, y) """
        

        self.ID3(self.treeHead, X, y, range(len(y)), range(len(X[0])), self.maxDepth)

        return

    def ID3(self, root, X, y, indexes, attributes, maxDepth):
        if(root.depth > self.depth):
            self.depth = root.depth
        numTotal = len(indexes)
        numPositive = len(filter(lambda l: y[l] == 1, indexes))
        if numPositive == numTotal:
            root.classLabelConfidence = 1.0
            return
        elif numPositive == 0:
            root.classLabelConfidence = 0.0
            return
        elif len(attributes) == 0 or root.depth >= maxDepth:
            root.classLabelConfidence = float(numPositive)/numTotal
            #print "Ending tree here"
            return

        bestAttr = DecisionTree.getBestAttr(X, y, indexes, attributes)
        root.attribute = bestAttr
        for val in DecisionTree.getValuesOf(bestAttr, X):
            childNodeWithVal = TreeNode(root)
            self.size += 1
            root.children[val] = childNodeWithVal
            indexesWithVal = filter(lambda l: X[l][bestAttr] == val, indexes)
            if len(indexesWithVal) == 0:
                childNodeWithVal.classLabelConfidence = float(numPositive)/numTotal
            else:
                #print str(bestAttr) + "\n"
                #print attributes
                nextAttr = list(attributes)
                nextAttr.remove(bestAttr)
                self.ID3(childNodeWithVal, X, y, indexesWithVal, nextAttr, maxDepth)

        return
        
    @staticmethod
    def getBestAttr(X, y, indexes, attributes):
        numTotal = len(indexes)
        bestAttr = -1
        lowestEntropy = float("inf")
        for attr in attributes:
            entropy = 0.0
            possibleValues = DecisionTree.getValuesOf(attr, X)
            for val in possibleValues:
                indexesWithVal = filter(lambda l: X[l][attr] == val, indexes)
                numYPositive = len(filter(lambda l: y[l] == 1, indexesWithVal))
                entropy += DecisionTree.calcEntropy(numYPositive, len(indexesWithVal), numTotal)
            #print "Entropy is %f for %d" % (entropy, attr)
            if entropy < lowestEntropy:
                lowestEntropy = entropy
                bestAttr = attr

        return bestAttr

    @staticmethod
    def calcEntropy(numPositive, numWithVal, numTotal):
         if 0 in [numPositive, numWithVal, numTotal]:
            return 0
         if numPositive in [numWithVal, numTotal]:
            return 0
         if numWithVal in [numPositive, numTotal]:
            return 0
         if numTotal in [numPositive, numWithVal]:
            return 0

         #print "Numpos %d numWithVal %d numTotal %d" % (numPositive, numWithVal, numTotal)
         probVal = float(numWithVal) / numTotal
         probPos = float(numPositive) / numWithVal
         probNeg = 1 - probPos
         #print "probVal %f probPos %f probNeg %f" % (- probVal*(probPos*np.log2(probPos) + probNeg*np.log2(probNeg)), probPos*np.log2(probPos), probNeg*np.log2(probNeg))
         return - probVal*(probPos*np.log2(probPos) + probNeg*np.log2(probNeg))

    @staticmethod
    def getValuesOf(attr, X):
        vals = set()
        for example in X:
            vals.add(example[attr])
        return vals

    def predict(self, X):
        """ Return the -1/1 predictions of the decision tree """
        predictions = []
        for example in X:
            prob = self.predict_proba_example(example)
            if prob >= 0.5:
                predictions.append(1)
            else:
                predictions.append(-1)
        return predictions

    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        predictions = []
        for example in X:
            prob = self.predict_proba_example(example)
            predictions.append(prob)
        return predictions

    def predict_proba_example(self, example):
        curNode = self.treeHead
        while curNode.classLabelConfidence is None:
            curAttr = curNode.attribute
            exampleVal = example[curAttr]
            curNode = curNode.children[exampleVal]
        return curNode.classLabelConfidence

    def size(self):
        """
        Return the number of nodes in the tree
        """
        return self.size

    def depth(self):
        """
        Returns the maximum depth of the tree
        (A tree with a single root node has depth 0)
        """
        return self.depth


class TreeNode(object):

    def __init__(self, parent):
        self.children = {}
        self.parent = parent
        if parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        self.classLabelConfidence = None

    def makeLeafNode(self, confidenceLevel):
        self.classLabelConfidence = confidenceLevel
        self.trueNode = None
        self.falseNode = None

