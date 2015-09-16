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
        numTotal = len(y)
        numPositive = len(filter(lambda l: l == 1, y))
        if numPositive == numTotal:
            self.treeHead.classLabelConfidence = 1.0
            return
        else if numPositive == 0:
            self.treeHead.classLabelConfidence = 0.0
            return
        else if len(X) == 0 or len(X[0]) == 0
            self.treeHead.classLabelConfidence = numPositive/numTotal
            return

        ID3(self.treeHead, X, y, range(numTotal), range(len(X[0])))

        return

    def ID3(root, X, y, indexes, attributes):
        

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

    def __init__(self, parent, trueNode=None, falseNode=None):
        self.trueNode = trueNode
        self.falseNode = falseNode
        self.parent = parent
        self.classLabelConfidence = None

    def makeLeafNode(self, confidenceLevel)
        self.classLabelConfidence = confidenceLevel
        self.trueNode = None
        self.falseNode = None

