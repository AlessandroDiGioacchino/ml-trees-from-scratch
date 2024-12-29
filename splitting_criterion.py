
import abc
import numpy as np

class SplittingCriterion( metaclass=abc.ABCMeta ):
  # @abc.abstractmethod
  def evaluate( self, ys: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray ) -> float:
    phi_parent = self.phi( ys )
    phi_left = self.phi( ys[ left_mask ] )
    phi_right = self.phi( ys[ right_mask ] )
    phi_children = ( np.sum( left_mask ) / len( ys ) * phi_left +
      np.sum( right_mask ) / len( ys ) * phi_right )

    return phi_parent - phi_children

  @abc.abstractmethod
  def phi( self, ys: np.ndarray ) -> float:
    pass

class Gini( SplittingCriterion ):
  def phi( self, ys: np.ndarray ):
    ps = np.bincount( ys ) / len( ys )  # Probabilities of each class
    if len( ps ) == 0:
      return 0

    return 2 * ps[ 0 ] * ( 1 - ps[ 0 ] )
