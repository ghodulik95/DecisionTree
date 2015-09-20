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
        self.treeHead = TreeNode(None, None, None)
        self.maxDepth = depth
        pass

    def fit(self, X, y, sample_weight=None):
        """ Build a decision tree classifier trained on data (X, y) """
        

        ID3(self.treeHead, X, y, range(len(y)), range(len(X[0])))

        return

    def ID3(root, X, y, indexes, attributes):
        numTotal = len(indexes)
        numPositive = len(filter(lambda l: l == 1, y))
        if numPositive == numTotal:
            root.classLabelConfidence = 1.0
            return
        else if numPositive == 0:
            root.classLabelConfidence = 0.0
            return
        else if len(indexes) == 0 or len(attributes) == 0
            root.classLabelConfidence = numPositive/numTotal
            return

        bestAttr = getBestAttr(X, y, indexes, attributes)
        root.attribute = bestAttr
        for val in getValuesOf(bestAttr, X)
            childNodeWithVal = TreeNode(root)
            root.children.append(childNodeWithVal)
            indexesWithVal = filter(lambda l: X[l][attr] == val, indexes)
            if len(indexesWithVal) == 0:
                childNodeWithVal.classLabelConfidence = numPositive/numTotal
            else:
                ID3(childNodeWithVal, X, y, indexesWithVal, attributes.remove(bestAttr))

        return
        
    def getBestAttriute(X, y, indexes, attributes):
        numTotal = len(indexes)
        bestAttr = -1
        lowestEntropy = float("inf")
        for attr in attributes:
            entropy = 0.0
            possibleValues = getValuesOf(attr, X)
            for val in possibleValues:
                indexesWithVal = filter(lambda l: X[l][attr] == val, indexes)
                numYPositive = filter(lambda l: y[l] == 1, indexesWithVal)
                entropy += calcEntropy(numYPositive, len(indexesWithVal), numTotal)
            if entropy < lowestEntropy:
                lowestEntropy = entropy
                bestAttr = attr

        return bestAttr

    def calcEntropy(numPositive, numWithVal, numTotal):
         if(numPositive == 0 || numPositive == numTotal):
            return 0

         probVal = numWithVal / numTotal
         probPos = numPositive / numWithVal
         probNeg = 1 - probPos
         return - (probPos*np.log2(probPos) + probNeg*np.log2(probNeg))

    def getValuesOf(attr, X):
        vals = set()
        for example in X:
            vals.add(example[attr])
        return vals

    def predict(self, X):
        """ Return the -1/1 predictions of the decision tree """
        pass

    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        pass

    def size(self):
        """
        Return the number of nodes in the tree
        """
        pass

    def depth(self):
        """
        Returns the maximum depth of the tree
        (A tree with a single root node has depth 0)
        """
        pass

    def entropyY(y):
        totalNum = len(y)
        numPositive = len(filter(lambda l: l == 1, y))
        numNegative = len(filter(lambda l: l != 1, y))

        pp = numPositive/totalNum
        pn = numNegative/totalNum

        return - (pp*np.log2(pp) + pn*np.log2(pn)) 

    def entropyYgivenX(X, y):
        pass


class TreeNode(object)

    def __init__(self, parent):
        self.children = []
        self.parent = parent
        self.classLabelConfidence = None

    def makeLeafNode(self, confidenceLevel)
        self.classLabelConfidence = confidenceLevel
        self.trueNode = None
        self.falseNode = None

