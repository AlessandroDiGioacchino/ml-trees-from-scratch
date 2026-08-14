
import itertools
import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from ml_trees import TreePredictor


def manual_gridsearch(
  X: np.ndarray,
  y: np.ndarray,
  parameters: dict,
  categorical_features: list[ int ] ):

  ns = 3
  rs = 42
  nuv = 2**6

  max_depth = parameters[ 'max_depth' ]
  min_samples_split = parameters[ 'min_samples_split' ]
  min_impurity_decrease = parameters[ 'min_impurity_decrease' ]

  outer_cv = StratifiedKFold( n_splits=ns, shuffle=True, random_state=rs )

  parameters_star = []
  scores = []
  out = { 'Best depth, split, impurity triplets': parameters_star,
          'Respective scores': scores }

  for outer_train_idx, test_idx in outer_cv.split( X, y ):
    outer_X_train = X[ outer_train_idx ]
    outer_y_train = y[ outer_train_idx ]
    X_test, y_test = X[ test_idx ], y[ test_idx ]

    score_star = -1
    depth_star = None
    split_star = None
    impurity_star = None

    for depth, split, impurity in itertools.product( max_depth,
      min_samples_split, min_impurity_decrease ):

      ix = 0
      print( f'-- Current parameters: max_depth = {depth},',
             f'min_samples_split = {split},',
             f'min_impurity_decrease = {impurity}' )

      inner_cv = StratifiedKFold( n_splits=ns, shuffle=True, random_state=rs )

      for inner_train_idx, val_idx in inner_cv.split( outer_X_train,
        outer_y_train ):

        inner_X_train = outer_X_train[ inner_train_idx ]
        inner_y_train = outer_y_train[ inner_train_idx ]
        X_val, y_val = outer_X_train[ val_idx ], outer_y_train[ val_idx ]

        clf = TreePredictor(
          categorical_features, max_depth=depth, min_samples_split=split,
          min_impurity_decrease=impurity, n_unique_values=nuv )

        start = time.time()
        clf.fit( inner_X_train, inner_y_train )
        end = time.time()

        score = accuracy_score( y_val, clf.predict( X_val ) )

        print( f'-- Score on fold {ix}: {score}' )
        print( '-- Time required to train', round( end - start, 2 ) )
        if score_star < score:
          score_star = score
          depth_star = depth
          split_star = split
          impurity_star = impurity

        ix = ix + 1

    clf = TreePredictor(
      categorical_features, max_depth=depth_star, min_samples_split=split_star,
      min_impurity_decrease=impurity_star, n_unique_values=nuv )

    clf.fit( outer_X_train, outer_y_train )

    parameters_star.append(( depth_star, split_star, impurity_star ))

    score = accuracy_score( y_test, clf.predict( X_test ))
    scores.append( score )

  return out
