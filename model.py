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
    sl = len(right_labels)/n

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

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

