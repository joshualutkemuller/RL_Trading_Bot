""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
"""
import numpy as np

class RTLearner(object):
    """
    Regression Random Tree learner. Splits on a randomly chosen feature at each node,
    using the median of that feature as the split value. Stops splitting when
    - number of samples ≤ leaf_size, OR
    - all target values are identical.
    """

    def __init__(self,leaf_size,verbose=False):
        self.leaf_size=leaf_size
        self.verbose=verbose
        self.learner=[]

    def author(self):
        return "jlutkemuller3"  # replace with your Georgia Tech username

    def gtid(self):
        """
        :return: The GT ID of the student
        :rtype: int
        """
        return 904051695  # replace with your GT ID number

    def study_group(self):
        """
        Returns
            A comma separated string of GT_Name of each member of your study group
            # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

        Return type
            str
        """
        return "jlutkemuller3"

    def build_tree(self, data):
        tree = np.array([])
        flag = 0
        if (data.shape[0] <= self.leaf_size):
            tree = np.array([['leaf', data[0][-1], '-1', '-1']])
            return tree

        X_attr = int(np.random.randint(data.shape[1] - 1))

        # if values of Xattribute are the same
        if (np.all(data[:, X_attr] == data[0][X_attr])):
            return np.array([['leaf', np.mean(data[:, -1]), '-1', '-1']])

        data = data[np.argsort(data[:, X_attr])]
        splitVal = np.median(data[0:, X_attr])
        if max(data[:, X_attr]) == splitVal:
            return np.array([['leaf', np.mean(data[:, -1]), '-1', '-1']])

        # building left and right sub-trees
        left_tree = self.build_tree(data[data[:, X_attr] <= splitVal])
        right_tree = self.build_tree(data[data[:, X_attr] > splitVal])
        root_values = [X_attr, splitVal, 1, left_tree.shape[0] + 1]
        tree = np.vstack((root_values, left_tree, right_tree))
        return tree

    def add_evidence(self, Xtrain, Ytrain):
        data = np.concatenate(([Xtrain, Ytrain[:, None]]), axis=1)
        tree = self.build_tree(data)
        self.learner = np.array(tree)

    def query(self, trainX):
        row = 0
        predictionY = np.array([])
        for data in trainX:
            while (self.learner[row][0] != 'leaf'):
                X_attr = self.learner[row][0]
                X_attr = int(float(X_attr))
                if (float(data[X_attr]) <= float(self.learner[row][1])):
                    row = row + int(float(self.learner[row][2]))
                else:
                    row = row + int(float(self.learner[row][3]))
                row = int(float(row))
            if (self.learner[row][0] == 'leaf'):
                predictionY = np.append(predictionY, float(self.learner[row][1]))
                row = 0
        return predictionY
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    pass
