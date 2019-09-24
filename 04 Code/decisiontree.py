import sys
import os
import math
import pandas

# -------------------------------------------------------------------------

train_file = sys.argv[1]
test_file = sys.argv[2]
dirpath = os.getcwd()
train_path = os.path.basename(train_file)
test_path = os.path.basename(test_file)


# -------------------------------------------------------------------------
class DecisionTree:

    def __init__(self):
        """
        Making an object to refer to the classify class
        Object used twice, once in Training the tree, and then in Predicting final value
        """
        self.classify = Classifier()

    def inputdata(self):

        """
        Reads the data from path file
        :return: Tree
        """
        global train_file
        indep_vars = []
        with open(train_file, "r") as input_csv:
            # with open("blackbox13_train.csv", "r") as input_csv:
            data = input_csv.readlines()

            for row in range(len(data)):
                val = list(map(int, data[row].rstrip("\n").split(",")))
                indep_vars.append(val)

        # Calling make_tree function in classify class by passing the training data
        # Returns a tree object
        tree = self.classify.make_dtree(indep_vars)
        return tree

    # -------------------------------------------------------------------------

    def testData(self, tree):

        """
        Reads the test data and creates submission file
        :param tree: The tree trained by training data is used to traverse the test data
        :return: call predict now function to start traversal
        """
        global test_file
        test_indep_vars = []

        with open(test_file, "r") as test_csv:
            # with open("blackbox13_test.csv", "r") as test_csv:
            test_data = test_csv.readlines()
            num_rows_test = len(test_data)

            for row in range(num_rows_test):
                val = list(map(int, test_data[row].rstrip("\n").split(",")))
                test_indep_vars.append(val)

        self.predict_now(test_indep_vars)

    # -------------------------------------------------------------------------

    def predict_now(self, test_indep_vars):

        """
        Store predicted value for each row of input testing data
        Iteratively call predict function
        :param test_indep_vars: Testing data
        :return: Final solution; csv dump
        """
        res = []
        for i in range(0, len(test_indep_vars)):
            if self.classify.predict(test_indep_vars[i], tree) == -1:
                res.append(0)
            else:
                res.append(1)

        submission = pandas.DataFrame(res)
        # print(submission)
        # submission.to_csv("blackbox13_predictions.csv", header=False, index=False)
        global test_path
        global dirpath

        submission.to_csv(test_path[:10] + "_predictions.csv", header=False, index=False)


# -------------------------------------------------------------------------


class Classifier:

    def infogain(self, rows, c, value):

        """
        Splits tree in two branches based on less than or greater than one of the unique values in that column
        Then calculates the information gain on the node with its two branches
        :param rows: total num of rows
        :param c: id of column under consideration
        :param value: one of the unique values in that column
        :return: left subtree, right subtree, and calculated information gain
        """

        # left subtree consists of numbers greater than value, and right is smaller
        left, right = [], []
        for i in rows:
            if i[c] >= value:
                left.append(i)
            else:
                right.append(i)

        # Entropy of the root node - including all left and right tree values
        root_node_ent = self.calc_entropy(rows)

        # Info Gain = Root Node Entropy - Summation of Entropy of Children
        tot_obs = len(rows)
        info_gain = root_node_ent - ((len(left) / tot_obs) * self.calc_entropy(left)) - \
                    ((len(right) / tot_obs) * self.calc_entropy(right))

        return left, right, info_gain

    # -------------------------------------------------------------------------

    def make_dtree(self, rows):

        """
        The main function which makes the decision tree:
            1. Checks if data is empty
            2. Cycles through all the columns to find one with max info gain
            3. Info gain calculated on each unique value in a column
            4. Column with max info gain chosen for splitting
            5. If resulting gain is more than 0.05, then tree is again spilt
        :param rows: All data
        :return: make further trees else if unsplittable return label
        """

        if len(rows) == 0:
            return Node()

        attrib_idx, attrib_val = 0, 0
        max_gain = float('-inf')
        lhs, rhs = [], []

        # Cycle through all the columns
        for i in range(len(rows[0]) - 1):

            # Create list of unique values in that column
            uniq_col_val = set()
            for r in rows:
                uniq_col_val.add(r[i])

            # Calculate entropy for each unique value
            for value in list(uniq_col_val):

                # returns infor gain, and the left and right branch with >= value and < value respectively
                left, right, info_gain = self.infogain(rows, i, value)

                # Store column, column_var, left & right subtree with max info gain
                if info_gain > max_gain:
                    max_gain = info_gain
                    attrib_idx = i
                    attrib_val = value
                    lhs = left
                    rhs = right

        # Check if tree can still be slit, if yes split again, else return max frequency label

        """
        - Limiting the max gain to a 5% threshold to as to avoid the scenarios of a possible split at the primary key level
        - Consistent with the industry practices
        - A form of pre-pruning
        - Ex: The one Dr. Shen gave in class if when john plays the DT tries to split on Days (D1, D2...) we might 
          achieve a higher training accuracy but at the cost of testing accuracy        
        """

        if max_gain > 0:
            left_tree = self.make_dtree(lhs)
            right_tree = self.make_dtree(rhs)
            # Split tree gain on selected column attributes
            return Node(indx=attrib_idx, value=attrib_val, left_tree=left_tree, right_tree=right_tree)

        else:
            # Calculate the count of each of the labels, so we can assign the overall label to the one with max count
            neg_lbl, pos_lbl = 0, 0
            for i in rows:
                if i[-1] == 1:
                    pos_lbl += 1
                else:
                    neg_lbl += 1

            # If node has more +ive labels, make it +ive, else -ive
            if pos_lbl > neg_lbl:
                return Node(label=1)
            else:
                return Node(label=-1)

    # -------------------------------------------------------------------------

    def calc_entropy(self, rows):

        """
        Calculate the Entropy or measure the amount of randomness of a dataset
        :param rows: data under consideration
        :return: entropy
        """

        # Calculate the +ive and -ive lables in that data
        neg_lbl, pos_lbl = 0, 0
        for i in rows:
            if i[-1] == 1:
                pos_lbl += 1
            else:
                neg_lbl += 1

        prob, ent = 0.0, 0.0
        tot_lbls = len(rows)

        # Entropy = - probability * log2(probability)
        # Sum all the entropy's of similar labels
        for lbl in pos_lbl, neg_lbl:
            if lbl == 0:
                ent += 0
            else:
                prob = lbl / tot_lbls
                ent += -1 * prob * math.log(prob, 2)

        return ent

    # -------------------------------------------------------------------------

    def predict(self, features, tree):

        """
        Recursive Tree traversal to search and return label on testing data
        :param features: Test data
        :param tree: Decision tree made using the training data
        :return: Final predicted value/row
        """

        # Return Final label if you reach the end of node
        if tree.label:
            return tree.label
        # Else make branches of of all the features and traverse the tree
        else:
            obs = features[tree.indx]
            if obs >= tree.value:
                branch = tree.left_branch
            else:
                branch = tree.right_branch

            # recursive call same function of either selected branch
            return self.predict(features, branch)


# -------------------------------------------------------------------------

# As each value of tree is stored in a node, made a common node class
class Node:

    def __init__(self, indx=0, value=None, label=None, left_tree=None, right_tree=None):
        """
        Set initial values for each node
        :param col: Index of the column under consideration
        :param value: Value of that column under consideration
        :param label: Final label allotted
        :param left_tree: Subtree with values >= selected val
        :param right_tree: Subtree with values < selected val
        """
        self.indx = indx
        self.value = value
        self.label = label
        self.left_branch = left_tree
        self.right_branch = right_tree


# -------------------------------------------------------------------------


if __name__ == '__main__':
    obj = DecisionTree()
    tree = obj.inputdata()
    obj.testData(tree)