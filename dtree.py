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
            return

        (bestAttr, split) = self.getBestAttr(X, y, indexes, attributes)
        root.attribute = bestAttr
        if split is None:
            for val in DecisionTree.getValuesOf(bestAttr, X):
                childNodeWithVal = TreeNode(root)
                self.size += 1
                root.children[val] = childNodeWithVal
                indexesWithVal = filter(lambda l: X[l][bestAttr] == val, indexes)
                if len(indexesWithVal) == 0:
                    childNodeWithVal.classLabelConfidence = float(numPositive)/numTotal
                else:
                    nextAttr = list(attributes)
                    nextAttr.remove(bestAttr)
                    self.ID3(childNodeWithVal, X, y, indexesWithVal, nextAttr, maxDepth)
        else:
            root.splitVal = split
            childGreaterOrEqual = TreeNode(root)
            root.children[">="] = childGreaterOrEqual
            indexesGreaterOrEqual = filter(lambda l: X[l][attribute] >= split, indexes)
            if(len(indexesGreaterOrEqual) == 0):
                childNodeWithVal.classLabelConfidence = float(numPositive)/numTotal
            else:
                nextAttr = list(attributes)
                nextAttr.remove(bestAttr)
                self.ID3(childNodeWithVal, X, y, indexesGreaterOrEqual, nextAttr, maxDepth)

            childLessThan = TreeNode(root)
            root.children["<"] = childLessThan
            indexesLessThan = filter(lambda l: X[l][attribute] < split, indexes)
            if(len(indexesLessThan) == 0):
                childNodeWithVal.classLabelConfidence = float(numPositive)/numTotal
            else:
                nextAttr = list(attributes)
                nextAttr.remove(bestAttr)
                self.ID3(childNodeWithVal, X, y, indexesLessThan, nextAttr, maxDepth)

        return
        
    def attrIsNominal(self, attr):
        if self.schema is None:
            return True
        else:
            return self.schema.is_nominal(attr)

    def getBestAttr(self, X, y, indexes, attributes):
        numTotal = len(indexes)
        bestAttr = -1
        lowestEntropy = float("inf")
        bestSplit = None
        for attr in attributes:
            entropy = 0.0
            split = None
            if(self.attrIsNominal(attr)):
                possibleValues = DecisionTree.getValuesOf(attr, X)
                for val in possibleValues:
                    indexesWithVal = filter(lambda l: X[l][attr] == val, indexes)
                    numYPositive = len(filter(lambda l: y[l] == 1, indexesWithVal))
                    entropy += DecisionTree.calcEntropy(numYPositive, len(indexesWithVal), numTotal)
            else:
                (split, entropy) = DecisionTree.getBestSplit(X, y, indexes, attr)

            if entropy < lowestEntropy:
                lowestEntropy = entropy
                bestAttr = attr
                bestSplit = split

        return bestAttr, bestSplit

    @staticmethod
    def getBestSplit(X, y, indexes, attribute):
        numTotal = len(indexes)
        lowestEntropy = float("inf")
        bestSplit = None
        splits = DecisionTree.getSplits(X, y, attribute)
        for split in splits:
            entropy = 0.0
            indexesGreaterOrEqual = filter(lambda l: X[l][attribute] >= split, indexes)
            numYPositiveGOE = len(filter(lambda l: y[l] == 1, indexesGreaterOrEqual))
            entropy += DecisionTree.calcEntropy(numYPositiveGOE, len(indexesGreaterOrEqual), numTotal)

            indexesLessThan = filter(lambda l: X[l][attribute] < split, indexes)
            numYPositiveLT = len(filter(lambda l: y[l] == 1, indexesLessThan))
            entropy += DecisionTree.calcEntropy(numYPositiveLT, len(indexesLessThan), numTotal)

            if entropy < lowestEntropy:
                lowestEntropy = entropy
                bestSplit = split
        return bestSplit, lowestEntropy


    @staticmethod
    def getSplits(X, y, attribute):
        attr = X[:][attribute]
        attrWithLabel = zip(attr, y)
        attrWithLabel = DecisionTree.removeDuplicates(attrWithLabel)
        attrWithLabel.sort(key=lambda l: l[0])

        attrWithLabel.insert(0, (float("-inf"), None))
        attrWithLabel.append((float("inf"), None))
        #print attrWithLabel
        lowestEntropy = 0.0
        splits = set()
        for i in range(1, len(attrWithLabel)-1):
            curVal = attrWithLabel[i][0]
            prevVal = attrWithLabel[i-1][0]
            nextVal = attrWithLabel[i+1][0]
            curLabel = attrWithLabel[i][1]
            prevLabel = attrWithLabel[i-1][1]
            nextLabel = attrWithLabel[i+1][1]

            if curVal != prevVal and curLabel != prevLabel:
                splits.add((curVal + prevVal)/2)
            if curVal != nextVal and curLabel != nextLabel:
                splits.add((curVal + nextVal)/2)
            if curVal == prevVal:
                splits.add((curVal + nextVal)/2)
            if curVal == nextVal:
                splits.add((curVal + prevVal)/2)
        return splits

    @staticmethod
    def removeDuplicates(l):
        toReturn = []
        for element in l:
            if element not in toReturn:
                toReturn.append(l)
        return toReturn[0]

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

         probVal = float(numWithVal) / numTotal
         probPos = float(numPositive) / numWithVal
         probNeg = 1 - probPos
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
            if curNode.splitVal is None:
                curNode = curNode.children[exampleVal]
            elif exampleVal >= curNode.splitVal:
                curNode = curNode.children[">="]
            elif exampleVal < curNode.splitVal:
                curNode = curNode.children["<"]

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
        self.splitVal = None

