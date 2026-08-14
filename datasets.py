
import numpy as np
import pandas as pd


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
