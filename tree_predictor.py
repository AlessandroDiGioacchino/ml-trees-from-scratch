
import numpy as np
import pandas as pd
import time

from typing import Callable, List

from node import Node
from splitting_criterion import Gini, SplittingCriterion

class TreePredictor:
  def __init__(
    self, categorical_features: List[ str ], root: Node = None,
    splitting_criterion: SplittingCriterion = Gini(), max_depth: int = 100,
    min_samples_split: int = 2, min_impurity_decrease: float = 0.0
  ):

    self.root = root
    self.categorical_features = categorical_features
    self.splitting_criterion = splitting_criterion
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_impurity_decrease = min_impurity_decrease

  def fit( self, Xs: np.ndarray, ys: np.ndarray ):
    if isinstance( Xs, pd.DataFrame ):
      print( 'DataFrame detected, casting..' )
      Xs = Xs.values

    if isinstance( ys, pd.Series ):
      print( 'Series detected, casting..' )
      ys = ys.values

    self.root = self._build_tree( Xs, ys, 0 )

  def predict( self, Xs: np.ndarray ):
    res = [ self._predict_sample( xs, self.root ) for xs in Xs ]
    return res

  def _predict_sample( self, xs: np.ndarray, node: Node ) -> int:
    if node.leaf:
      return node.class_label

    # print( xs )
    if node.decision_function( xs ):
      print( 'Going left' )
      return self._predict_sample( xs, node.left )
    else:
      print( 'Going right' )
      return self._predict_sample( xs, node.right )

  def _build_tree( self, Xs: np.ndarray, ys: np.ndarray, depth: int ) -> Node:
    most_freq_label = self._most_freq_label( ys )
    root = Node( leaf=True, class_label=most_freq_label )

    n_samples = len( Xs )
    n_labels = len( np.unique( ys ) )
    print( 'Amount of samples:', n_samples )
    print( 'Amount of labels:', n_labels )

    # Only one unique label, or not enough samples to split, or tree too deep
    if ( n_labels == 1 or n_samples <= self.min_samples_split or
         self.max_depth <= depth ):

      return root

    attribute, test = self._pick_attribute_and_test( Xs, ys )
    if attribute is None:
      return root

    # xs = Xs.iloc[ :, attribute ]
    left_indices, right_indices = self._partition( Xs, test )
    print(
      'Amount of Trues in left mask:', np.sum( left_indices ),
      'amount of Trues in right mask:', np.sum( right_indices ) )

    if len( left_indices ) == 0 or len( right_indices ) == 0:
      return root

    # Xs = Xs.drop( Xs.columns[ attribute ], axis=1 )
    # Xs = np.delete( Xs, attribute, axis=1 )
    #
    # new_categorical_features = []
    # for column in self.categorical_features:
    #   if column < attribute:
    #     new_categorical_features.append( column )
    #   if attribute < column:
    #     new_categorical_features.append( column - 1 )
    #
    # self.categorical_features = new_categorical_features

    left_subset, left_labels = Xs[ left_indices ], ys[ left_indices ]
    right_subset, right_labels = Xs[ right_indices ], ys[ right_indices ]
    depth = depth + 1

    print( 'Length of left subset:', len( left_subset ) )
    print( 'Length of right subset:', len( right_subset ) )

    print( 'Building left subtree..' )
    left_child = self._build_tree( left_subset, left_labels, depth )

    print( 'Building right subtree' )
    right_child = self._build_tree( right_subset, right_labels, depth )
    root = Node( decision_function=test, left=left_child, right=right_child )
    return root

  def _pick_attribute_and_test( self, Xs: np.ndarray, ys: np.ndarray ):
    best_gain = None
    attribute = None
    test = None
    threshold = None

    for column in range( Xs.shape[ 1 ] ):
      # if columns_to_ignore[ column ]:
      #   continue

      xs = Xs[ :, column ]
      # if the attribute is categorical, test is the membership function
      # if it is continuous, test is the threshold function
      #   Overall, we want test to be a function returning indices of elements
      # satisfying some condition

      # print( xs.dtype )
      # nan_mask = np.isnan( xs, casting='no' )
      # ts = np.unique( xs[ ~nan_mask ] )  # issues with NaN values
      # if np.any( nan_mask ):
      #   ts = np.concatenate([ ts, [ np.nan ] ])
      # ts = xs.unique()
      ts = np.unique( xs )
      print( 'Evaluating attribute', column, 'with', len( ts ), 'distinct values' )

      for t in ts:  # Possibly the most expensive operation
        if column in self.categorical_features:
          print( 'Attribute', column, 'is categorical' )
          test_ = lambda xs: xs[ column ] == t
        else:
          np.random.seed( 42 )
          num_samples = int( len( ts ) * 0.1 )
          ts = np.random.choice( ts, size=num_samples, replace=False )
          test_ = lambda xs: xs[ column ] <= t

      # For now, we just select the median..
        # t = np.median( ts )
        # test_ = lambda xs: xs.iloc[ column ] <= t

        start_time = time.time()
        gain = self._compute_gain( Xs, ys, test_ )
        end_time = time.time()
        # print( f'Function self._compute_gain took {end_time - start_time:.2f} seconds to execute' )

        if best_gain is None or best_gain < gain:
          print( gain )
          best_gain = gain
          attribute = column
          test = test_
          threshold = t

    if attribute is not None:
      print( '-- Attribute selected:', attribute, 'with threshold', threshold, 'and gain', best_gain )
    return attribute, test

  def _compute_gain( self, Xs: np.ndarray, ys: np.ndarray, test: Callable[ [ np.ndarray ], bool ] ):
    # For now, only continuous attributes
    #   Threshold should be replaced by some function to work for both
    # categorical and continuous
    start_time = time.time()
    left_indices, right_indices = self._partition( Xs, test )
    end_time = time.time()
    # print( f'Function self._partition took {end_time - start_time:.2f} seconds to execute' )

    # left_subset, right_subset = Xs[ left_indices ], Xs[ right_indices ]
    return self.splitting_criterion.evaluate( ys, left_indices, right_indices )

  # xs is only a column of the dataset
  #   test is a function that takes a numpy vector x as input, and returns True
  # if x passes the test
  def _partition( self, Xs: np.ndarray, test: Callable[ [ np.ndarray ], bool ] ):
    #   Doesn't work: result doesn't contain False values, so it is a much
    # shorter list than Xs -> can't be used to access Xs
    # left_indices = np.array([ test( xs ) for xs in Xs ])
    # left_mask = np.array( list( map( test, Xs ) ) )
    # left_mask = np.array([ True if test( xs ) else False for xs in Xs ])
    # left_mask = Xs.apply( test, axis=1 )
    left_mask = np.apply_along_axis( test, axis=1, arr=Xs )
    amnt = np.sum( left_mask )
    # print( len( Xs ) )
    # print( left_mask.any() )
    right_mask = ~left_mask
    amnt = np.sum( right_mask )
    # print( 'Does the right mask contain any True?', right_mask.any() )
    # left_indices = [ i for i, x in enumerate( xs ) if x <= threshold ]
    # right_indices = [ i for i, x in enumerate( xs ) if threshold < x ]
    # first_subset = xs[[ test( x ) == 1 for x in xs[ xs.columns[ attribute ] ] ]]
    # second_subset = xs[[ test( x ) == 2 for x in xs[ xs.columns[ attribute ] ] ]]

    return ( left_mask, right_mask )

  #   When working with Numpy Arrays, columns can only be accessed by their
  # offset. After splitting over a column, it is deleted from the dataset so
  # that it is no longer considered for future splits. Since categorical
  # features are stored in terms of their offset, we need to update them after
  # dropping a column
  #   A more elegant solution could be to also pass a list of columns to
  # ignore (i.e. those that have already been used for a split) to function
  # _pick_attribute_and_test
  def _update_categorical_features( self, deleted_attribute: int ):
    return

  def _most_freq_label( self, ys: np.ndarray ) -> int:
    n_zeroes = len( ys[ ys == 0 ] )
    n_ones = len( ys[ ys == 1 ] )
    if n_zeroes < n_ones:
      return 1

    return 0
