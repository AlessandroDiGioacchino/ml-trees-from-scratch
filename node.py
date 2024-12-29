
import numpy as np

from typing import Callable, Optional


class Node:

  def __init__( self,
                leaf: bool = False,
                decision_function: Optional[ Callable[ [ np.ndarray ], bool ] ] = None,
                left: Optional[ 'Node' ] = None,
                right: Optional[ 'Node' ] = None,
                class_label: Optional[ int ] = None ) -> 'Node':

    """
      initializes a Node.

      :param decision_function: A function that takes a numpy vector and returns a boolean.
      :param left: The left child node (default is None).
      :param right: The right child node (default is None).
      :param class_label: The majority class label for leaf nodes (default is None).
    """
    self.decision_function = decision_function
    self.leaf = leaf
    self.left = left
    self.right = right
    self.class_label = class_label
    # self._is_leaf = self._check_if_leaf()

  def label( self, label: int ):
    self.label = label

  def test( self, test: Callable[ [ np.ndarray ], bool ] ):
    self.decision_function = test

  # def add_left( self, l: 'Node' ):
  #   self.left = l
  #   self._is_leaf = False
  #
  # def add_right( self, r: 'Node' ):
  #   self.right = r
  #   self._is_leaf = False
  #
  # def add_decision_test( self, decision_function: Callable[ [ np.ndarray ], bool ] ):
  #   self.decision_function = decision_function
  #
  # def set_decision_function( self, feature: int, value: str ) -> None:
  #   self.decision_function = lambda x: x[ feature ] == value
  #
  # def evaluate( self, data_point: np.ndarray ) -> bool:
  #   """
  #     Evaluates the decision function on a given data point.
  #
  #     :param data_point: A numpy vector representing the data point.
  #     :return: The result of the decision function if it exists, otherwise None.
  #   """
  #   if self.decision_function is not None:
  #     return self.decision_function( data_point )
  #
  #   return None
  #
  # def _check_if_leaf( self ) -> bool:
  #   return self.left is None and self.right is None
