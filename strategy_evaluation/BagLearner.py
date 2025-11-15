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

class BagLearner(object):
    """
    BagLearner implements Bootstrap Aggregation (bagging). It can wrap any learner
    that follows the add_evidence/query API. Each bag is trained on a bootstrap
    sample of the data, where predictions are the average of the bagged learners.

    For experimental & performance purposes, the default N bags is set to 20.
    """

    def __init__(self, learner, kwargs=None, bags=20, boost=False, verbose=False):
        """
        @param learner: the class (constructor) of the base learner to bag (e.g. DTLearner)
        @param kwargs:  dictionary of keyword arguments to pass to each base-learner's constructor
        @param bags:    number of bootstrap samples / learners to create
        @param boost:   unused (boosting not implemented)
        @param verbose: if True, print training progress
        """
        self.learner_cls = learner
        self.kwargs = {} if kwargs is None else kwargs.copy()
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []

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

    def add_evidence(self, Xtrain, Ytrain):
        """
        Train `bags` learners, each on a bootstrap sample of (Xtrain, Ytrain).
        """
        # Ensure numpy arrays
        X = np.array(Xtrain)
        Y = np.array(Ytrain).reshape(-1, )

        n = X.shape[0]
        self.learners = []

        for i in range(self.bags):
            # Bootstrap sample indices (with replacement)
            idxs = np.random.choice(n, size=n, replace=True)
            X_sample = X[idxs, :]
            Y_sample = Y[idxs]

            # Instantiate a new base learner and train it
            learner = self.learner_cls(**self.kwargs)
            learner.add_evidence(X_sample, Y_sample)
            self.learners.append(learner)

            if self.verbose:
                print(f"BagLearner: trained bag {i + 1}/{self.bags}")

    def query(self, Xtest):
        """
        Query each bagged learner, then return the average of their predictions.
        """
        X = np.array(Xtest)

        # Collect predictions from each learner
        preds_array = []
        for learner in self.learners:
            pred = learner.query(X)
            preds_array.append(pred.reshape(-1, ))

        # Stack to shape (bags, M), then average over axis=0 → (M,)
        all_predictions_array = np.vstack(preds_array)
        return np.mean(all_predictions_array, axis=0)


if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    pass
    #print("the secret clue is 'zzyzx'")
