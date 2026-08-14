
import itertools
import numpy as np


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


#   Recipe of itertools:
# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset( iterable ):
  "Subsequences of the iterable from shortest to longest."
  # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
  s = list( iterable )
  return itertools.chain.from_iterable(
    itertools.combinations( s, r ) for r in range( len( s ) + 1 ) )
