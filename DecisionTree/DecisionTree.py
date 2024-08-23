import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, predicted_value=None):
        '''
        Args: 
            feature (scalar): Index of feature. On basis of this feature the node should split data.
            threshold (scalar): Value of the feature on from which other datapoint is compared for spliting 
            predicted_value (scalar): If node is leaf node in decesion then it will return prediction value. Otherwise it will be null.
            left (Node)
            right (Node)
        '''

        self.feature = feature
        self.threshold = threshold
        self.predicted_value = predicted_value

        self.left = left
        self.right = right
        
    def is_leaf_node(self):
        return self.predicted_value != None

class DecisionTree:
    def __init__(self, max_depth=100):
        self.max_depth = 100
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y, curr_depth=0, max_depth=self.max_depth)

    def build_tree(self, X, y, curr_depth ,max_depth):
        '''
        It creates decesion tree based on given training data

        Args:
            X (ndarray(m, n)): m examples and n features
            y (ndarray(m, )): target value of m examples
            max_depth (scalar)
        Return:
            root (Node): return parent node
        '''
        m,n = X.shape
    
        # base case
        if curr_depth == max_depth or len(np.unique(y)) == 1 or m < 2:
            # find most common label
            unique_labels, cnt = np.unique(y, return_counts=True)
            max_cnt_idx = np.argmax(cnt)
            most_common_label =  unique_labels[max_cnt_idx]

            return Node(predicted_value=most_common_label)
            
        # select best feature and threshold 
        feature, threshold = self.best_split(X, y)
    
        # split data based on best feature and threshold
        left_idxs, right_idxs = self.split(X, feature, threshold)
        
        # call child node
        left_node = self.build_tree(X[left_idxs], y[left_idxs], curr_depth+1, max_depth)
        right_node = self.build_tree(X[right_idxs], y[right_idxs], curr_depth+1, max_depth)

        return Node(feature, threshold, left_node, right_node)

    def best_split(self, X, y):
        '''
        Returns the feature and threshold that is hightest information w.r.t given datapoints
    
        Args:
            X (ndarray(m,n))
            y (ndarray(m,))
        Returns:
            best_feature (scalar): index of the feature
            best_threshold (scalar): one of the category or number from the feature
        '''
        
        n_features = X.shape[1]
    
        best_feature = -1
        best_threshold = -1
        best_info_gain = -1
        
        for feature in range(n_features):
            categories = np.unique(X[:, feature])
        
            for threshold in categories:
                info_gain = self.information_gain(X, y, feature, threshold)
    
                if best_info_gain < info_gain:
                    best_info_gain = info_gain
                    best_threshold = threshold
                    best_feature = feature

        return best_feature, best_threshold



    def information_gain(self, X, y, feature, threshold):
        '''
        Args:
            X (ndarray(m, n))
            y (ndarray(m,))
            feature (scalar)
            threshold (scalar)
    
        Returns:
            info_gain (scalar)
        '''
        # split data into left and right idices
        left_idxs, right_idxs = self.split(X, feature, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
            
        # Calculate entropy of parent and children node
        w_left = len(left_idxs) / len(y)
        w_right = len(right_idxs) / len(y)
        entropy_left = self.entropy(y[left_idxs])
        entropy_right = self.entropy(y[right_idxs])
        
        entropy_parent = self.entropy(y)
        entropy_children = w_left * entropy_left + w_right * entropy_right
        
        info_gain =  entropy_parent - entropy_children
    
        return info_gain

    def split(self, X, feature, threshold):
        '''
        Returns indices of left split and right split. This is done based on feature and threshold given.
        
        Args:
            X (ndarray(m, n))
            feature (scalar)
            threshold (scalar)
            
        Returns:
            left_idxs (ndarray(m2))
            right_idxs (ndarray(m2,))
        '''
        left_idxs = np.argwhere(X[:, feature] <= threshold).flatten()
        right_idxs = np.argwhere(X[:, feature] > threshold).flatten()
    
        return left_idxs, right_idxs

    def entropy(self, y):
        '''
        Calculates impurities in the target value 
    
        Args:
            y ndarray(m, )
    
        Returns:
            entropy 
        '''
        categories, cnt = np.unique(y, return_counts = True)
        p = cnt / len(y)
    
        entropy = -np.sum(p * np.log(p))
        return entropy
    
    def score(self, X, y):
        y_pred = self.predict(X)

        return np.sum(y_pred == y) / len(y)
        
    def predict(self, X):
        if not self.root:
            raise ValueError("The tree has not been trained yet.")
            
        ans = np.array([self._traverse_tree(x, self.root) for x in X])
        return ans

    def _traverse_tree(self, x, node):
        '''Predicts target value for single datapoint'''
        if node.is_leaf_node():
            return node.predicted_value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
            
        return self._traverse_tree(x, node.right)