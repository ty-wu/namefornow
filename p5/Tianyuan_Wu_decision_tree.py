import numpy as np
import util

from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


'''DO NOT modify this'''
def q1_train_test_split(X,y):
    '''
    Return a dictionary that contains 50 random train/test splits
    For the i-th split:
        splits[i][0]: X_train
        splits[i][1]: y_train
        splits[i][2]: X_test
        splits[i][3]: y_test
    '''
    random_seeds = [i for i in range(50)]
    splits = {}
    for i in range(50):
        # print random_seeds
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=random_seeds[i])
        splits[i] = []
        splits[i].append(X_train)
        splits[i].append(y_train)
        splits[i].append(X_test)
        splits[i].append(y_test)
    return splits


def decision_tree(X_train, y_train, X_test, y_test, max_depth=20):
    clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=1,max_depth = max_depth)
    clf = clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)

def cross_val(X_train,y_train):
    '''
    Returns the best max_depth parameter after 5-fold cross-validation
    Input:
        X_train : np.array (n_train, d) - array of training feature vectors
        y_train : np.array (n_train) - array of labels corresponding to X_train samples
    Return:
        best_max_depth: int - the best maximum depth of the tree on this training sample

    Read sklearn documention about how to use KFold
    You cannot use sklearn.model_selection.cross_val_score
    '''
    cv = KFold(n_splits = 5, shuffle=True, random_state=1)
    numOfFs = np.shape(X_train)[1]
    res = 0
    bestScore = 0

    for i in range(numOfFs):
        score = 0
        for train_idx, test_idx in cv.split(X_train):
            X_t, X_v = X_train[train_idx], X[test_idx]
            y_t, y_v = y_train[train_idx], y_train[test_idx]
            score += decision_tree(X_t, y_t, X_v, y_v, max_depth = (i+1))
        score /= cv.get_n_splits(X_train)
        print 'score when depth = ', i+1, 'is', score
        if score >= bestScore:
            bestScore = score
            res = i+1

    return res

if __name__ == '__main__':
    X,y = util.load_data('dataset/Automobile_data.csv')
    '''----------------------q1----------------------'''
    splits = q1_train_test_split(X,y)
    avg_acc = 0.0
    # print splits[0][0].shape, splits[0][1].shape, splits[0][2].shape, splits[0][3].shape
    for i in range(50):
        avg_acc += decision_tree(splits[i][0], splits[i][1], splits[i][2], splits[i][3])
    avg_acc /= 50
    print 'Average accuracy is:', avg_acc

    '''----------------------q2----------------------'''
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=1)
    best_depth = cross_val(X_train,y_train)
    print 'Best max_depth found is:', best_depth
