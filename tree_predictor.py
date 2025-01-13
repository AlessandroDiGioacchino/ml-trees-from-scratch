
import numpy as np
# import pandas as pd
# import time

from dataclasses import asdict, dataclass
from typing import Callable, Optional, Union

import utils
from node import Node
from splitting_criterion import Gini, SplittingCriterion


class TreePredictor:
  '''
  Class for building and using a binary tree predictor.

  Attributes
  ----------
  categorical_features : list[ int ]
    The list of offsets for categorical features in the dataset.
  root : Node, default=None
    The root of this tree predictor.
  splitting_criterion : SplittingCriterion, default=Gini()
    The criterion used to split a leaf.
  max_depth : int, default=100
    The maximum depth of the tree predictor.
  min_samples_split : int, default=2
    The minimum amount of samples required to perform a split.
  min_impurity_decrease : float, default=0.0

  n_unique_values : int, default=None
    The sample size extracted from all unique values of a feature.
  '''

  def __init__( self,
    categorical_features: list[ int ],
    # root: Optional[ Node ] = None,
    splitting_criterion: SplittingCriterion = Gini(),
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_impurity_decrease: float = 0.0,
    n_unique_values: int = None
  ) -> None:

    self.categorical_features = categorical_features

    # self.root = root
    self.splitting_criterion = splitting_criterion
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_impurity_decrease = min_impurity_decrease
    self.n_unique_values = n_unique_values

  def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
    '''
    Builds the tree from the training set ( X, y ).

    Parameters
    ----------
    X : np.ndarray of shape ( n_samples, n_features )
      The training input samples.
    y : np.ndarray of shape ( n_samples, )
      The target values (class labels) as integers (zero or one).
    '''
    self.root = self._build_tree( X, y )

  def predict( self, X: np.ndarray ) -> list[ int ]:
    '''
    Predicts the label of the input samples X.

    Parameters
    ----------
    X : np.ndarray of shape ( n_samples, n_features )
      The input samples.

    Returns
    -------
    preds : list[ int ] of shape ( n_samples, )
        The predicted labels of input samples X. The label of sample X[ i ] is
      preds[ i ].
    '''
    preds = [ self._predict_sample( x, self.root ) for x in X ]
    return preds

  def _predict_sample( self, xs: np.ndarray, root: Node ) -> int:
    '''
    Predicts the label of the input sample xs.

    Parameters
    ----------
    xs : np.ndarray of shape ( 1, n_features )
      The input sample.
    root : Node
      The root of the binary tree predictor.

    Returns
    -------
    y : int
      The predicted label of input sample xs.
    '''
    if root.is_leaf():
      return root.label

    # print( xs )
    if root.decision_criterion( xs ):
      # print( 'Going left' )
      return self._predict_sample( xs, root.left )
    else:
      # print( 'Going right' )
      return self._predict_sample( xs, root.right )

  def _build_tree( self,
    Xs: np.ndarray,
    ys: np.ndarray,
    depth: int = 0
  ) -> Node:
    '''
    Recursively builds this binary tree predictor from training set ( Xs, ys ).

    Parameters
    ----------
    Xs : np.ndarray of shape ( n_samples, n_features )
      The input samples.
    ys : np.ndarray of shape ( n_samples, )
      The target values (class labels) as integers (zero or one).
    depth : int, default=0
      The depth of this binary tree predictor.

    Returns
    -------
    root : Node
      The root of the resulting binary tree predictor.
    '''
    most_freq_label = TreePredictor._most_freq_label( ys )
    root = Node( label=most_freq_label )

    n_samples = len( Xs )
    n_unique_labels = len( np.unique( ys ) )
    # print( 'Amount of samples:', n_samples )
    # print( 'Amount of labels:', n_unique_labels )

    # Only one unique label, or not enough samples to split, or tree too deep
    if n_unique_labels == 1:
      return root

    if n_samples <= self.min_samples_split:
      print( f'Only {n_samples} samples, not splitting' )
      return root

    if self.max_depth <= depth:
      print( f'Current depth {depth} greater than or equal to maximum depth, '
              'not splitting' )

      return root

    # Only needed for debug purposes
    feature_names = [
      'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
      'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
      'stem-height', 'stem-width', 'stem-color', 'has-ring', 'ring-type',
      'habitat', 'season'
    ]

    # feature, test, gain, set_threshold = self._pick_feature_and_test( Xs, ys )
    result = asdict( self._pick_feature_and_test( Xs, ys ) )

    if ( gain := result[ 'best_gain' ] ) <= self.min_impurity_decrease:
      print( f'information gain {gain} too small, not splitting' )
      return root

    best_feature = result[ 'best_feature' ]
    if best_feature is None:
      return root

    # print( f'information gain {gain} OK, splitting on feature {feature_names[ feature ]}' )

    best_test = result[ 'best_test' ]
    left_mask, right_mask = TreePredictor._partition( Xs, best_test )
    # print(
    #   f'Amount of Trues in left mask: {np.sum( left_mask )}, '
    #   f'amount of Trues in right mask: {np.sum( right_mask )}' )

    if left_mask.size == 0 or right_mask.size == 0:
      return root

    description = ( f"Feature {feature_names[ best_feature ]}"
      f"{'in' if best_feature in self.categorical_features else '<='}"
      f"{result[ 'best_set_threshold' ] }" )

    left_subset, left_labels = Xs[ left_mask ], ys[ left_mask ]
    right_subset, right_labels = Xs[ right_mask ], ys[ right_mask ]
    depth += 1

    # print( 'Size of left subset:', len( left_subset ) )
    # print( 'Size of right subset:', len( right_subset ) )

    # print( '---- Building left subtree..' )
    left_child = self._build_tree( left_subset, left_labels, depth )
    # print( '---- Building right subtree..' )
    right_child = self._build_tree( right_subset, right_labels, depth )
    root = Node(
      decision_criterion=best_test,
      left=left_child,
      right=right_child,
      decision_description=description )

    return root


  def _pick_feature_and_test( self, Xs: np.ndarray, ys: np.ndarray
  ) -> ( Optional[ int ], Optional[ Callable[ [ np.ndarray ], bool ] ] ):

    '''
    Picks a feature of Xs and a test according to the splitting criterion.

    Parameters
    ----------
    Xs : np.ndarray of shape ( n_samples, n_features )
      The input samples.
    ys : np.ndarray of shape ( n_samples, )
      The target values (class labels) as integers (zero or one).

    Returns
    -------
    best_feature : int
      The offset of the feature chosen to perform the split.
    best_test : Callable[ [ np.ndarray ], bool ]
      The function to perform the split.
    '''
    best_feature = None
    best_gain = None
    best_set_threshold = None
    best_test = None

    # Only needed for debug purposes
    feature_names = [
      'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
      'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
      'stem-height', 'stem-width', 'stem-color', 'has-ring', 'ring-type',
      'habitat', 'season'
    ]

    np.random.seed( 42 )

    def _compute_gain_for_feature(
      feature: np.ndarray,
      test: Callable[ [ np.ndarray ], bool ],
      set_threshold: Union[ float, set ]
    ):

      nonlocal best_feature, best_gain, best_test, best_set_threshold

      gain = self._compute_gain( Xs, ys, test )
      if gain is None:
        return

      if best_gain is None or best_gain < gain:
        # print( 'New best gain:', best_gain, end='\x1b[1K\n' )
        best_feature = feature
        best_gain = gain
        best_set_threshold = set_threshold
        best_test = test

    for feature in range( Xs.shape[ 1 ] ):
      xs = Xs[ :, feature ]

      # if the attribute is categorical, test is the membership function
      # if it is continuous, test is the threshold function
      #   Overall, we want test to be a function returning indices of elements
      # satisfying some condition

      # We can try something here to properly manage NaN values
      ts = np.unique( xs )
      # print( f'Evaluating attribute {feature_names[ feature ]} with {len( ts )} distinct values' )

      if feature in self.categorical_features:
        # print( f'Attribute {feature_names[ feature ]} is categorical' )
        def categorical_test( col: int, st: set ):
          return lambda xs: xs[ col ] in st

        powerset = utils.powerset( ts )
        if 2**( len( ts ) ) > self.n_unique_values:
          powerset = utils.iter_sample_fast( powerset, self.n_unique_values )

        for st in powerset:
          test = categorical_test( feature, st )
          _compute_gain_for_feature( feature, test, st )
      else:
        def numerical_test( col: int, val: float ):
          return lambda xs: xs[ col ] <= val

        if len( ts ) > self.n_unique_values:
          num_samples = self.n_unique_values  # int( len( ts ) * 0.1 )
          ts = np.random.choice( ts, size=num_samples, replace=False )

        for t in ts:
          test = numerical_test( feature, t )
          _compute_gain_for_feature( feature, test, t )

    if best_feature is not None:
      print( f'-- Attribute selected: {feature_names[ best_feature ]}, '
             f'with set/threshold {best_set_threshold} and '
             f'gain {best_gain}' )

    # Definition of BestSplit class at the bottom
    return BestSplit(
      best_feature=best_feature,
      best_gain=best_gain,
      best_set_threshold=best_set_threshold,
      best_test=best_test )


  def _compute_gain( self,
    Xs: np.ndarray,
    ys: np.ndarray,
    test: Callable[ [ np.ndarray ], bool ]
  ) -> float:
    '''
      Computes the information gain achieved by splitting input samples Xs
    with function test.

    Parameters
    ----------
    Xs : np.ndarray of shape ( n_samples, n_features )
      The input samples.
    ys : np.ndarray of shape ( n_samples, )
      The target values (class labels) as integers (zero or one).
    test : Callable[ [ np.ndarray ], bool ]
      The function to perform the split.

    Returns
    -------
    info_gain : float
        The information gain achieved by splitting input samples Xs with
      function test.
    '''
    # start_time = time.time()
    left_mask, right_mask = TreePredictor._partition( Xs, test )
    # end_time = time.time()
    # print( f'Function self._partition took {end_time - start_time:.2f} seconds to execute' )
    # print( 'Splitting criterion is', self.splitting_criterion )

    return (
      self.splitting_criterion.information_gain( ys, left_mask, right_mask ) )


  def __str__( self ):
    """
    Returns a string representation of the tree.
    """
    if self.root is None:
      return "Tree is empty."

    return self._print_tree( self.root, 0 )


  def _print_tree( self, node, depth ):
    """
    Helper method to recursively print the tree.

    Parameters
    ----------
    node : Node
        The current node of the tree.
    depth : int
        The current depth for indentation.

    Returns
    -------
    tree_str : str
        The string representation of the subtree rooted at `node`.
    """
    if node.is_leaf():
      return f"{'  ' * depth}Leaf: Label={node.label}\n"

    result = f"{'  ' * depth}Node:\n"
    result += f"{'  ' * ( depth + 1 )}Decision: {node.decision_description}\n"
    result += f"{'  ' * ( depth + 1 )}Left:\n"
    result += self._print_tree( node.left, depth + 2 )
    result += f"{'  ' * ( depth + 1 )}Right:\n"
    result += self._print_tree( node.right, depth + 2 )
    return result


  #   test is a function that takes a numpy vector x as input, and returns True
  # if x passes the test
  @staticmethod
  def _partition(
    Xs: np.ndarray,
    test: Callable[ [ np.ndarray ], bool ]
  ) -> tuple[ np.ndarray, np.ndarray ]:
    '''
      Computes the partition of indices obtained by splitting input samples Xs
    with function test.

    Parameters
    ----------
    Xs : np.ndarray of shape ( n_samples, n_features )
      The input samples.
    test : Callable[ [ np.ndarray ], bool ]
      The function to perform the split.

    Returns
    -------
    mask : Tuple[ np.ndarray, np.ndarray ] of shape ( n_samples, n_samples )
        The pair of boolean vectors to access input samples Xs and achieve the
      partition resulting from applying function test to Xs. For each index i,
      left_mask[ i ] != right_mask[ i ].
    '''

    #   Doesn't work: result doesn't contain False values, so it is a much
    # shorter list than Xs -> can't be used to access Xs
    # left_indices = np.array([ test( xs ) for xs in Xs ])
    # left_mask = np.array( list( map( test, Xs ) ) )
    # left_mask = np.array([ True if test( xs ) else False for xs in Xs ])
    # left_mask = Xs.apply( test, axis=1 )

    # Applies test function example by example
    left_mask = np.apply_along_axis( test, axis=1, arr=Xs )
    # amnt = np.sum( left_mask )
    # print( len( Xs ) )
    # print( left_mask.any() )
    right_mask = ~left_mask
    # amnt = np.sum( right_mask )
    # print( 'Does the right mask contain any True?', right_mask.any() )

    return ( left_mask, right_mask )


  @staticmethod
  def _most_freq_label( ys: np.ndarray ) -> int:
    '''
    Computes the most frequent label in target values ys.

    Parameters
    ----------
    ys : np.ndarray
      The target values (class labels) as integers (zero or one).

    Returns
    -------
    label : int
      The most frequent label in ys (either zero or one).
    '''

    n_zeroes = len( ys[ ys == 0 ] )
    n_ones = len( ys[ ys == 1 ] )

    if n_zeroes < n_ones:
      return 1
    else:
      return 0


@dataclass
class BestSplit:
  best_feature: int
  best_gain: float
  best_set_threshold: Union[ float, set ]
  best_test: Callable[ [ np.ndarray ], bool ]
