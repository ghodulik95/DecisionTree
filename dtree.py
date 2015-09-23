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
        #The maximum depth withh be a non-zero positive number, or None
        if depth <= 0:
            self.maxDepth = None
        else:
            self.maxDepth = depth
        #Initialize the head of our decisiton tree
        self.treeHead = TreeNode(None)
        #The initial depth is 0
        self.depth = 0
        #The initial size is 1 - the head is a node of the tree
        self.size = 1
        pass

    def fit(self, X, y, schema, sample_weight=None):
        """ Build a decision tree classifier trained on data (X, y) """

        #We will use the schema to get the possible values of attributes
        self.schema = schema
        #We just call the ID3 recursive algorithm
        self.ID3(self.treeHead, X, y, range(len(y)), range(len(X[0])))

        return

    #The parameters are the root of the tree - this is recursive, so root will not always be the head of the tree
    #X is the examples, y is the labels
    #indexes is the list of indexes which apply to this subtree
    #attributes is the list of attributes which have not already been used somewhere up the tree
    def ID3(self, root, X, y, indexes, attributes):
        #Some maintenance - we just set the depth of the tree if this node is deeper than the recorded max depth
        if(root.depth > self.depth):
            self.depth = root.depth

        #Get the number of examples in this subtree, and the number of thos which are positive
        numTotal = len(indexes)
        numPositive = len(filter(lambda l: y[l] == 1, indexes))
        #If all the examples are positive, we can make this a leaf node with confidence 1
        if numPositive == numTotal:
            root.classLabelConfidence = 1.0
            return
        #If all the examples are not positive, we can make this a leaf node with a confidence of 0
        elif numPositive == 0:
            root.classLabelConfidence = 0.0
            return
        #If we are out of attributes to try, or we are at max depth, make this a leaf node 
        elif len(attributes) == 0 or (self.maxDepth is not None and root.depth >= self.maxDepth):
            #The class label confidence is the percentage which are positive
            root.classLabelConfidence = float(numPositive)/numTotal
            return

        #If none of the above cases apply, we find the best attribute to apply to this node
        (bestAttr, split) = self.getBestAttr(X, y, indexes, attributes)
        root.attribute = bestAttr
        #If the split of bestAttr is None, meaning this is a discrete attribute
        if split is None:
            #For every value of this attribute, defined in the schema
            for val_str in self.schema.nominal_values[bestAttr]:
                val = int(val_str)
                #Make a child node
                childNodeWithVal = TreeNode(root)
                root.children[val] = childNodeWithVal
                #increment size of tree
                self.size += 1
                #Get the indexes which have this value
                indexesWithVal = filter(lambda l: X[l][bestAttr] == val, indexes)
                #If none have this value, make the child have the confidence its parent would have if it were a leaf
                if len(indexesWithVal) == 0:
                    childNodeWithVal.classLabelConfidence = float(numPositive)/numTotal
                #Otherwise, remove this attribute and recurse
                else:
                    nextAttr = list(attributes)
                    nextAttr.remove(bestAttr)
                    self.ID3(childNodeWithVal, X, y, indexesWithVal, nextAttr)
        #If this is a continuous attribute
        else:
            #We will make a child for having a value >= to the split, and one for <
            #Process is otherwise the same as in the discrete case, just done twice
            root.splitVal = split
            childGreaterOrEqual = TreeNode(root)
            root.children[">="] = childGreaterOrEqual
            indexesGreaterOrEqual = filter(lambda l: X[l][bestAttr] >= split, indexes)
            if(len(indexesGreaterOrEqual) == 0):
                childGreaterOrEqual.classLabelConfidence = float(numPositive)/numTotal
            else:
                nextAttr = list(attributes)
                nextAttr.remove(bestAttr)
                self.ID3(childGreaterOrEqual, X, y, indexesGreaterOrEqual, nextAttr)

            childLessThan = TreeNode(root)
            root.children["<"] = childLessThan
            indexesLessThan = filter(lambda l: X[l][bestAttr] < split, indexes)
            if(len(indexesLessThan) == 0):
                childLessThan.classLabelConfidence = float(numPositive)/numTotal
                #print "Ending with %d/%d conf second cont lt" % (numPositive, numTotal)
            else:
                nextAttr = list(attributes)
                nextAttr.remove(bestAttr)
                self.ID3(childLessThan, X, y, indexesLessThan, nextAttr)

        return
        
    #This just returns whether an attribute is nominal or not based on the schema
    def attrIsNominal(self, attr):
        if self.schema is None:
            return True
        else:
            return self.schema.is_nominal(attr)

    #We will find the best attribute by finding which one results in the lowest entropy, thus the highest information gain
    def getBestAttr(self, X, y, indexes, attributes):
        #The number of examples applicable to the current subtree
        numTotal = len(indexes)
        #The lowest entorpy is initially infinite so any other entropy will be lower at first
        lowestEntropy = float("inf")
        #bestAttr will be overwritten to be an attribute index
        bestAttr = -1
        #In the continous case, bestSplit will be the split value of bestAttr which results in the lowest entropy
        bestSplit = None
        for attr in attributes:
            #initializing entropy and split
            entropy = 0.0
            split = None
            #If this is a discrete attribute
            if(self.attrIsNominal(attr)):
                #We iterate over eaach possible value
                possibleValues = self.schema.nominal_values[attr]
                for val_string in possibleValues:
                    val = int(val_string)
                    #Find the number of examples which have this value
                    indexesWithVal = filter(lambda l: X[l][attr] == val, indexes)
                    #And the number of those which have a positive class label
                    numYPositive = len(filter(lambda l: y[l] == 1, indexesWithVal))
                    #And calculate the entropy
                    entropy += DecisionTree.calcWeightedEntropy(numYPositive, len(indexesWithVal), numTotal)
            else:
                #In the continous case, find the split value which has the lowest entropy for this attr
                #And the resulting entropy
                (split, entropy) = DecisionTree.getBestSplit(X, y, indexes, attr)

            #If this is the lowest entropy encountered so far, remember it
            if entropy < lowestEntropy:
                lowestEntropy = entropy
                bestAttr = attr
                bestSplit = split

        return bestAttr, bestSplit

    #We will find the split value which resuls in the lowest entropy for the given attribute
    #Returns (bestSplit,lowestEntropy)
    @staticmethod
    def getBestSplit(X, y, indexes, attribute):
        #First, we get all the possible split values for this attribute
        #Note that the first split will always be -inf and the last +inf
        #splits = DecisionTree.getSplits(X, y, indexes, attribute)

        #Project the attribute onto it's corresponding class label
        attrWithLabel = DecisionTree.projectAttributeWithClassLabel(X, y, indexes, attribute)
        #Sort over the attribute value
        attrWithLabel.sort(key=lambda l: l[0])

        #We will iterate over all the splits
        #To have a O(N) runtime, I keep track of the number of positive class labels in the >= split and < split categories
        # as we iterate
        #The number of total examples will always be the length of the list of indexes
        numTotal = len(indexes)
        #Intially, all example are >= the split (first split is -inf), so the number of positve example GOE
        # is just the number of all positive examples
        numPositiveGOE = len(filter(lambda l: y[l] == 1, indexes))
        #Similarly, no exaples are < -inf, so there are no LT positive examples
        numPositiveLT = 0
        #Inital values to find the best split
        lowestEntropy = float("inf")
        bestSplit = None
        splits = DecisionTree.getSplits(list(attrWithLabel))
        #i is the index of the first value to be >= the current split
        #So, when the first split is -inf, i is 0
        i = 0
        for split in splits:
            #Increase i until we find an attribute that is >= the split
            while i < len(attrWithLabel) and attrWithLabel[i][0] < split:
                #If the ith example is positive, then, by increasing i, we are moving a positive GOE example to the LT examples
                #So, we increment accordingly
                if attrWithLabel[i][1] == 1:
                    numPositiveGOE -= 1
                    numPositiveLT += 1
                i += 1

            #Calculae the entropy of this split
            entropy = 0.0
            #The number of >= examples (regarless of class label) is numTotal - i
            numGOE = numTotal - i
            entropy += DecisionTree.calcWeightedEntropy(numPositiveGOE, numGOE, numTotal)

            #The number of < examples (regardless of class label) is i
            numLT = i
            entropy += DecisionTree.calcWeightedEntropy(numPositiveLT, numLT, numTotal)

            #If the entropy is the lowest recorded so far, remember it
            if entropy < lowestEntropy:
                lowestEntropy = entropy
                bestSplit = split
        return bestSplit, lowestEntropy

    #Produce a list of lists which inner lists are of the form [attributeValue, classLabel] for an example
    #Order of the elements in the outer list does not matter
    #This is nice so that we can ignore all the other attributes when finding the best split
    @staticmethod
    def projectAttributeWithClassLabel(X, y, indexes, attribute):
        #Make a list that is just the attribute values we want from X
        attr = [row[attribute] for row in X]
        #Zip this with y
        attrWithLabel = zip(attr, y)
        #Currently, attrWithLabel contains all examples.
        #We only care about the examples that have an index in indexes, so remove the rest
        #Note that doing this on X and y before zipping was slower in my tests
        attrWithLabel = DecisionTree.removeNotInIndexes(attrWithLabel, indexes)
        return attrWithLabel

    #Returns a sorted list of all the splits we should check for lowest entropy
    #We want any split for which the two examples (one on each side of split) with attribute values closest to the split have different class labels
    @staticmethod
    def getSplits(attrWithLabel):
        #We want to remove duplicates so that, if two examples have the same attribute value but different class labels
        # it will be very apparent since the same attribute value will appear multiple times
        attrWithLabel = DecisionTree.removeDuplicates(attrWithLabel)

        #We add +-inf to the beginning and end of the list
        attrWithLabel.insert(0, [float("-inf"), None])
        attrWithLabel.append([float("inf"), None])

        lowestEntropy = 0.0
        #We may add values multiple times, so we should store splits as a set to not have duplicates
        splits = set()
        #We will iterate all examples - since the beginning and end are just -+ inf and not examples, we do not iterate over them
        for i in range(1, len(attrWithLabel)-1):
            #We get the values and labels of the example at i, i-1, and i+
            curVal = attrWithLabel[i][0]
            prevVal = attrWithLabel[i-1][0]
            nextVal = attrWithLabel[i+1][0]
            curLabel = attrWithLabel[i][1]
            prevLabel = attrWithLabel[i-1][1]
            nextLabel = attrWithLabel[i+1][1]

            #If the values of the attributes of current and prev/next are different AND class labels are different
            # We add  the average of the two attribute values to our set of split candidates
            if curVal != prevVal and curLabel != prevLabel:
                splits.add(curVal/2 + prevVal/2)
            if curVal != nextVal and curLabel != nextLabel:
                splits.add(curVal/2 + nextVal/2)
            #If the current value == the previous value, then that means the current value is both positive and non-positive
            if curVal == prevVal:
                #So, we have to add the average of current val with the next
                splits.add(curVal/2 + nextVal/2)
            #If the current value == the next value, that means the curent value is both positive and non-positive
            if curVal == nextVal:
                #So, we have to add the average of the curent val with the previous
                splits.add(curVal/2 + prevVal/2)
        #Finally, we convert the set of candidate split values into a sorted list and return it
        listOfSplits = list(splits)
        listOfSplits.sort()
        return listOfSplits

    #Simply returns a list of example which were in the given examples
    @staticmethod
    def removeNotInIndexes(l, indexes):
        #Make a new list
        toReturn = []
        #Iterate over each index
        for i in range(len(l)):
            #If is in in our list of indexes, add the element at i to our return list
            if i in indexes:
                toReturn.append(l[i])
        return toReturn

    #Removes duplicates from a list of lists/tuples
    # Note that this returns as a list of lists, whether the input was a list of tuples, or a dictionary, etc
    @staticmethod
    def removeDuplicates(l):
        #Make a new list
        toReturn = []
        #For every value in the given list/set/dicstionary
        for val, label in l:
            #If our new list doesn't have this element, add it
            if [val, label] not in toReturn:
                toReturn.append([val,label])
        return toReturn

    #Calculates the entropy of the class labels of examples with certain value of an attribute
    # The entropy is weighted by the probability that any example (in the total count for this subtree) has that attribute value
    # So, this is P(X = x)*(P(Y = + | X = x)log(P(Y = + | X = x)) + P(Y = - | X = x)log(P(Y = - | X = x)))
    # I am calling it weighted since, technically, the entropy is not multiplied by P(X = x)
    @staticmethod
    def calcWeightedEntropy(numPositive, numWithVal, numTotal):
         #Firstly, if any parameter is 0, the entropy is 0
         # This means that there were no positive class labels or no examples with this attribute value
         # Note that if numTotal == 0, this is probably an error, but I return 0 so a divide by zero doesn't hapen
        if 0 in [numPositive, numWithVal, numTotal]:
           return 0
        #If all the examples had a positive class labelm the entropy is also 0
        if numPositive in [numWithVal, numTotal]:
           return 0

        #P(X = x)
        probVal = float(numWithVal) / numTotal
        #P(Y = + | X = x)
        probPos = float(numPositive) / numWithVal
        #P(Y != + | X = x)
        probNeg = 1 - probPos

        #Return the weighted entropy
        return - probVal*(probPos*np.log2(probPos) + probNeg*np.log2(probNeg))

    #Return a list of prediction corresponding to the given examples
    def predict(self, X):
        """ Return the -1/1 predictions of the decision tree """
        #The list that will hold the predictions
        predictions = []

        for example in X:
            #Get the confidence prediction for the example
            # Note that confidence ranges from 0 to 1, and is the number of positive labels / number of examples at the leaf
            prob = self.predict_proba_example(example)
            #If the confidence is above or equal to 0.5, we will predict positive
            if prob >= 0.5:
                predictions.append(1)
            #Otherwise, predict negative
            else:
                predictions.append(-1)

        return predictions

    #Return a list of confidences corresponding to the given examples
    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        predictions = []
        for example in X:
            prob = self.predict_proba_example(example)
            predictions.append(prob)
        return predictions

    #Predicts the class label for one example
    def predict_proba_example(self, example):
        #Start at the head node
        curNode = self.treeHead
        #While we are note at a leaf
        while not curNode.isLeaf():
            #Get the attribute of the current node
            curAttr = curNode.attribute
            #Get the examples value of that attribute
            exampleVal = example[curAttr]
            #In the discrete case, just go to the corresponding child node
            if curNode.isNominal():
                curNode = curNode.children[exampleVal]
            #In the continues case, compare with the splitVal to determine the corresponding child
            elif exampleVal >= curNode.splitVal:
                curNode = curNode.children[">="]
            elif exampleVal < curNode.splitVal:
                curNode = curNode.children["<"]

        return curNode.classLabelConfidence

    #Return the size of the tree
    # This value is kept track of in the fit function
    def size(self):
        """
        Return the number of nodes in the tree
        """
        return self.size

    #Return the depth of the tree - ie the longest path in the tree
    # The depth variable is kept track of in the fit function
    def depth(self):
        """
        Returns the maximum depth of the tree
        (A tree with a single root node has depth 0)
        """
        return self.depth


#A TreeNode class for storing our decision tree
class TreeNode(object):

    #Initialize
    def __init__(self, parent):
        #A dictionary of children
        self.children = {}
        #The parent is the given parent
        self.parent = parent
        #No parent indicates it is the tree head, so depth is 0
        if parent is None:
            self.depth = 0
        #Otherwise, the node depth is 1 more than its parent
        else:
            self.depth = self.parent.depth + 1

        #At initialization, the classLabelConfidence and splitVal are None
        # If this node turns out to be a leaf node, classConfidence will be set to a value between 0 and 1
        # splitVal will be set to a float value if the attribute is continuous
        self.classLabelConfidence = None
        self.splitVal = None

    #A node is a leaf node if it has a non-none confidence
    def isLeaf(self):
        return self.classLabelConfidence != None

    #Returns whether or not the attribute value assosciated with this node is nominal
    def isNominal(self):
        return self.splitVal is None

