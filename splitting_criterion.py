
import abc
import numpy as np


class SplittingCriterion( metaclass=abc.ABCMeta ):
  '''
    Abstract class representing splitting criterions for binary tree
  predictors.
  '''

  def information_gain( self,
    y: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray
  ) -> float:

    '''
    Computes the information gain for the chosen partition of labels `y`.

    Parameters
    ----------
    y : np.ndarray
      The binary labels.
    left_mask : np.ndarray
      A vector of boolean values to select a subset of values from `y`.
    right_mask : np.ndarray
        A vector of boolean values to select a subset of values from `y`.
      Given an index `ix`, `left_mask[ ix ]` and `right_mask[ ix ]` must have
      different values.
    '''
    phi_parent = self.phi( y )
    phi_left = self.phi( y[ left_mask ] )
    phi_right = self.phi( y[ right_mask ] )
    phi_children = ( np.sum( left_mask ) / len( y ) * phi_left +
      np.sum( right_mask ) / len( y ) * phi_right )

    info_gain = phi_parent - phi_children
    return info_gain

  @abc.abstractmethod
  def phi( self, y: np.ndarray ) -> float:
    '''
    Computes the heterogeneity index on the chosen labels `y`.

    Parameters
    ----------
    y : np.ndarray
      The binary labels.

    Returns
    -------
    res : float
      The heterogeneity index of values in `y`.
    '''
    pass

class Gini( SplittingCriterion ):
  '''
  Concrete subclass of `SplittingCriterion` representing the Gini index.
  '''
  def phi( self, y: np.ndarray ) -> float:
    '''
    Computes the Gini index on the chose labels `y`.

    Parameters
    ----------
    y : np.ndarray
      The binary labels.

    Returns
    -------
    gini : float
      The Gini index of values in `y`.
    '''
    if len( y ) == 0:
      return 0

    ps = np.bincount( y ) / len( y )  # Probabilities of each class
    gini = 2 * ps[ 0 ] * ( 1 - ps[ 0 ] )  # Symmetric function
    return gini

class Entropy( SplittingCriterion ):
  '''
  Concrete subclass of `SplittingCriterion` representing scaled entropy.
  '''
  def phi( self, y: np.ndarray ) -> float:
    '''
    Computes the scaled entropy on the chosen labels `y`.

    Parameters
    ----------
    y : np.ndarray
      The binary labels.

    Returns
    -------
    entropy : float
      The scaled entropy of values in `y`.
    '''
    # print( 'Evaluating entropy' )

    if len( y ) == 0:
      return 0

    ps = np.bincount( y ) / len( y )
    if ps[ 0 ] == 0 or ps[ 0 ] == 1:  # No different values in y
      return 0

    entropy = (
      - ps[ 0 ] / 2 * np.log2( ps[ 0 ] ) -
      ( 1 - ps[ 0 ] ) / 2 * np.log2( 1 - ps[ 0 ] ) )

    return entropy

class StandardDeviation( SplittingCriterion ):
  '''
  Concrete subclass of `SplittingCriterion` representing standard deviation.
  '''
  def phi( self, y: np.ndarray ) -> float:
    '''
    Computes the standard deviation on the chosen labels `y`.

    Parameters
    ----------
    y : np.ndarray
      The binary labels.

    Returns
    -------
    stdev : float
      The standard deviation of values in `y`.
    '''
    if len( y ) == 0:
      return 0

    ps = np.bincount( y ) / len( y )
    stdev = np.sqrt( ps[ 0 ] * ( 1 - ps[ 0 ] ) )
    return stdev
