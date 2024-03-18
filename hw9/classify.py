import numpy as np
import pandas as pd
from scipy.stats import mode

def count_syllables(word: str) -> int:
    vowels = 'aeiouy'
    count = 0
    word = word.lower()
    for i in range(len(word)):
        if word[i] in vowels and (i == 0 or word[i-1] not in vowels):
            count += 1
    return count

def generate_features(word: str) -> pd.Series:
    """
    Generates features given a word.
    """
    vowels = ['a', 'e', 'i', 'o', 'u']
    conditions = dict()
    
    # Letter Counts
    conditions['e_count'] = sum(1 for letter in word if letter.lower() in 'e')
    conditions['a_count'] = sum(1 for letter in word if letter.lower() in 'a')
    conditions['u_count'] = sum(1 for letter in word if letter.lower() in 'u')
    conditions['o_count'] = sum(1 for letter in word if letter.lower() in 'o')
    
    # Presence
    conditions['ch_presence'] = 'ch' in word.lower()
    conditions['contains_eu'] = 'eu' in word
    
    # Word Meta
    conditions['syllable_count'] = count_syllables(word)
    conditions['word_length'] = len(word)
    conditions['consonant_vowel_ratio'] = (len(word) - sum(word.lower().count(v) for v in vowels)) /\
                                        max(1, sum(word.lower().count(v) for v in vowels))
    
    # Prefix/Suffix Analysis
    conditions['starts_with_pre'] = word.startswith('pre')
    conditions['starts_with_re'] = word.startswith('re') 
    conditions['ends_with_cion'] = word.endswith('cion') 
    conditions['ends_in_vowel'] = word[-1] in vowels
    conditions['ends_in_two_vowels'] = word[-1] in vowels and word[-2] in vowels
    conditions['ends_in_r'] = word[-1] in 'r'
    
    # Letter Combinations
    conditions['ll_presence'] = 'll' in word
    conditions['qu_presence'] = 'qu' in word
    conditions['ch_presence_fr'] = 'ch' in word
    conditions['ou_presence'] = 'ou' in word
    
    
    return pd.Series(conditions)

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < 2:
            return {'label': mode(y)[0][0]}

        # Find best split
        best_gini = np.inf
        best_feature, best_threshold = None, None
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_idx] <= threshold)[0]
                right_indices = np.where(X[:, feature_idx] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gini = self._gini_impurity(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_feature is None:
            return {'label': 0}
            return {'label': mode(y)[0][0]}  # If no valid split found, return the majority class
        
        left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature] > best_threshold)[0]

        # Grow left and right subtrees
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature_idx': best_feature,
                'threshold': best_threshold,
                'left': left_subtree,
                'right': right_subtree}

    def _gini_impurity(self, left_y, right_y):
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = len(right_y) / (len(left_y) + len(right_y))
        return p_left * (1 - np.sum(np.square(np.bincount(left_y) / len(left_y)))) + \
               p_right * (1 - np.sum(np.square(np.bincount(right_y) / len(right_y))))

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if 'label' in tree:
            return tree['label']
        else:
            if x[tree['feature_idx']] <= tree['threshold']:
                return self._predict_tree(x, tree['left'])
            else:
                return self._predict_tree(x, tree['right'])

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if not self.max_features:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return mode(predictions)[0][0]



def classify(train_words, train_labels, test_words):
    # Generate Features
    proccess_y = lambda y_set: np.array([word == 'spanish' for word in y_set])
    X_train = np.array([generate_features(word).to_numpy() for word in train_words])
    y_train = proccess_y(train_labels)
    X_test = np.array([generate_features(word).to_numpy() for word in test_words])

    rf = RandomForest(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)
    return ['spanish' if item == 1 else 'french' for item in rf.predict(X_test)]