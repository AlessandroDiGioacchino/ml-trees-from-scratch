
import numpy as np

from typing import Callable, Optional


class Node:
  """
  Class representing nodes of binary tree predictors.

  Attributes
  ----------
  decision_criterion : Callable[ [ np.ndarray ], bool ], default=None
      A function that takes a Numpy vector and returns a boolean. None for
    leaves.
  left : Node, default=None
    The left child node. None for leaves.
  right : Node, default=None
    The right child node. None for leaves.
  label : int, default=None
    The majority class label for leaf nodes. None for inner nodes.
  decision_description : str, default=None
      A description of the decision criterion, mainly needed for printing
    purposes. None for leaves.
  """

  def __init__( self,
    decision_criterion: Optional[ Callable[ [ np.ndarray ], bool ] ] = None,
    left: Optional[ 'Node' ] = None, right: Optional[ 'Node' ] = None,
    label: Optional[ int ] = None, decision_description: str = None
  ) -> None:

    self.decision_criterion = decision_criterion
    self.left = left
    self.right = right
    self.label = label
    self.decision_description = decision_description

  def is_leaf( self ) -> bool:
    return self.left is None and self.right is None
