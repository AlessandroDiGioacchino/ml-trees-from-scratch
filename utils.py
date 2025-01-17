
import itertools
import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import tree_predictor


# Function from SO (slightly modified): https://stackoverflow.com/a/12583436
def iter_sample_fast( iterator, samplesize ):
  results = []
  # iterator = iter(iterable)
  # Fill in the first samplesize elements:
  try:
    for _ in range( samplesize ):
      results.append( next( iterator ) )
  except StopIteration:
    raise ValueError( 'Sample larger than population.' )

  np.random.shuffle( results )  # Randomize their positions
  for i, v in enumerate( iterator, samplesize ):
    r = np.random.randint( 0, i )
    if r < samplesize:
      results[ r ] = v  # at a decreasing rate, replace random items

  return results


def load_mushroom_dataset() -> tuple[ np.ndarray, np.ndarray, list[ int ] ]:
  mushroom = pd.read_csv(
    './secondary+mushroom+dataset/MushroomDataset/secondary_data.csv', sep=';'
  )

  categorical_columns = [
    'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
    'gill-attachment', 'gill-spacing', 'gill-color', 'stem-root',
    'stem-surface', 'stem-color', 'veil-type', 'veil-color', 'has-ring',
    'ring-type', 'spore-print-color', 'habitat', 'season' ]

  # Removing columns with majority of NaN values
  nan_percentages = mushroom.isna().mean()
  columns_to_drop = nan_percentages[ nan_percentages > 0.5 ].index
  columns_to_drop = columns_to_drop.tolist()

  mushroom = mushroom.drop( columns_to_drop, axis=1 )

  mushroom = mushroom.dropna()  # Removing rows with NaN values

  y = (
    mushroom[ 'class' ]
    .replace({ 'e': '0', 'p': '1' })
    .astype( int )
    .values )

  mushroom = mushroom.drop( 'class', axis=1 )  # Dropping target

  # Offsets for categorical features
  categorical_features = mushroom.columns.get_indexer( categorical_columns )
  categorical_features = [ int( x ) for x in categorical_features if x != -1 ]

  X = mushroom.values

  return ( X, y, categorical_features )


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

        clf = tree_predictor.TreePredictor(
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

    clf = tree_predictor.TreePredictor(
      categorical_features, max_depth=depth_star, min_samples_split=split_star,
      min_impurity_decrease=impurity_star, n_unique_values=nuv )

    clf.fit( outer_X_train, outer_y_train )

    parameters_star.append(( depth_star, split_star, impurity_star ))

    score = accuracy_score( y_test, clf.predict( X_test ))
    scores.append( score )

  return out


#   Recipe of itertools:
# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset( iterable ):
  "Subsequences of the iterable from shortest to longest."
  # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
  s = list( iterable )
  return itertools.chain.from_iterable(
    itertools.combinations( s, r ) for r in range( len( s ) + 1 ) )
