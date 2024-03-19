import numpy as np
import pandas as pd
from scipy.stats import mode
import numpy as np
from scipy.linalg import cholesky, solve_triangular

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

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, impurity_measure='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.impurity_measure = impurity_measure
        self.root = None

    def gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def impurity(self, y):
        if self.impurity_measure == 'gini':
            return self.gini_impurity(y)
        elif self.impurity_measure == 'entropy':
            return self.entropy(y)

    def information_gain(self, X, y, feature_index):
        left_indices = X.iloc[:, feature_index] <= X.iloc[:, feature_index].median()
        right_indices = ~left_indices
        left_y = y[left_indices]
        right_y = y[right_indices]
        parent_impurity = self.impurity(y)
        left_impurity = self.impurity(left_y)
        right_impurity = self.impurity(right_y)
        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        info_gain = parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)
        return info_gain

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        if self.max_features is not None:
            feature_indices = np.random.choice(num_features, self.max_features, replace=False)
        else:
            feature_indices = range(num_features)

        best_gain = 0
        best_feature = None

        if (np.all(y == y[0]) or
            depth == self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return Node(value=np.bincount(y).argmax())

        for feature_index in feature_indices:
            gain = self.information_gain(X, y, feature_index)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index

        if best_feature is None:
            return Node(value=np.bincount(y).argmax())

        threshold = X.iloc[:, best_feature].median()
        left_indices = np.where(X.iloc[:, best_feature] <= threshold)[0]
        right_indices = np.where(X.iloc[:, best_feature] > threshold)[0]

        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            return Node(value=np.bincount(y).argmax())

        left_X = X.iloc[left_indices, :]
        right_X = X.iloc[right_indices, :]
        left_y = y[left_indices]
        right_y = y[right_indices]
        left_node = self.build_tree(left_X, left_y, depth + 1)
        right_node = self.build_tree(right_X, right_y, depth + 1)
        return Node(best_feature, threshold, left_node, right_node)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i, row in X.iterrows():
            node = self.root
            while node.left and node.right:
                if row[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions[i] = node.value
        return predictions

def classify(train_words, train_labels, test_words):
    # Generate Features
    proccess_y = lambda y_set: np.array([word == 'spanish' for word in y_set])
    X_train = pd.DataFrame([generate_features(word) for word in train_words])
    y_train = proccess_y(train_labels)
    X_test = pd.DataFrame([generate_features(word) for word in test_words])

    rf = DecisionTreeClassifier(
        max_depth=7,
        min_samples_split=2,
        max_features=8,
        impurity_measure='entropy'
    )
    rf.fit(X_train, y_train)
    return ['spanish' if item == 1 else 'french' for item in rf.predict(X_test)]