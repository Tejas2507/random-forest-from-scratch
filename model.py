"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np 
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    n = len(labels) 
    if n == 0:
        return 0
    p = np.unique(labels,return_counts = True)

    p = np.sum((p[1]/n)**2)
    return 1-p

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):

    col = features[:,feature_index]
    mask = col <= threshold
    left_features = features[mask]
    left_lables = labels[mask]

    right_features = features[~mask]
    right_labels = labels[~mask]
    return (left_features,left_lables,right_features,right_labels)

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    ip = impurity(parent_labels)
    il = impurity(left_labels)
    ir = impurity(right_labels)
    n = len(parent_labels)
    sr = len(right_labels)/n
    sl = len(left_labels)/n

    return ip - (sr*ir + sl*il)

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    dih = {'feature_index': None,'threshold': None, 'score': 0.0 }

    for fi in feature_indices:
        u = np.unique(features[:,fi])
        thresholds = (u[:-1] + u[1:]) / 2
        for t in thresholds:
            lf, ll, rf, rl = split_dataset(features, labels, fi, t)
            if len(ll) == 0 or len(rl) == 0:
                continue
            s = split_score(labels, ll, rl)
            if s>dih['score']:
                dih['feature_index'] = fi
                dih['threshold'] = t
                dih['score'] = s
    
    return dih

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    if depth >= max_depth:
        return True 
    elif len(labels) < min_samples_split:
        return True 
    elif len(np.unique(labels)) == 1:
        return True 
    else:
        return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    classes, counts = np.unique(labels,return_counts = True)

    idx = np.argmax(counts)
    return int(classes[idx])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    if should_stop(labels,depth,max_depth,min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    if feature_subset is None:
        candidate_features = list(range(features.shape[1]))
    else:
        candidate_features = list(feature_subset)
    
    split = best_split(features,labels,candidate_features)
    if split['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    fi = split['feature_index']
    t = split['threshold']

    lf, ll, rf, rl = split_dataset(features, labels, fi, t)

    if len(ll) == 0 or len(rl) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    left_subtree = build_tree(
        lf, ll, max_depth, min_samples_split, feature_subset, depth + 1
    )

    right_subtree = build_tree(
        rf, rl, max_depth, min_samples_split, feature_subset, depth + 1
    )

    return {
        'leaf': False,
        'feature_index': fi,
        'threshold': t,
        'left': left_subtree,
        'right': right_subtree
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    if tree['leaf']:
        return int(tree['prediction'])
    
    j = tree['feature_index']
    t = tree['threshold']

    if example[j] <= t:
        return predict_example_tree(tree['left'],example)
    else:
        return predict_example_tree(tree['right'],example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.

    ans = [predict_example_tree(tree,x) for x in features]
    return np.array(ans)

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # TODO: draw a bootstrap sample of rows (with replacement) using rng.
    
    n = features.shape[0]
    idx = rng.integers(0,n,size=n)
    ans1 = features[idx]
    ans2 = labels[idx]

    return (ans1,ans2)

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # TODO: return num_to_pick distinct random feature indices from range(num_features) using rng.
    idx= rng.choice(num_features,size=num_to_pick,replace=False)
    return idx

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    rng = np.random.default_rng(random_state)
    
    d = features.shape[1]
    if num_features_per_split is None:
        num_features_per_split = max(1, int(round(np.sqrt(d))))
    forest = []
    for _ in range(num_trees):
        x_boot,y_boot = bootstrap_sample(features,labels,rng)
        selected_features = feature_subset(d,num_features_per_split,rng)
        tree = build_tree(x_boot,y_boot,max_depth=max_depth,min_samples_split=min_samples_split,feature_subset=selected_features)
        forest.append({'tree':tree,'feature_indices': selected_features})
    return forest

# Step 13 - combine_predictions
def combine_predictions(tree_predictions):
    ans = []

    for t in tree_predictions.T:
        classes, counts = np.unique(t, return_counts=True)
        ans.append(classes[np.argmax(counts)])

    return np.asarray(ans)

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

