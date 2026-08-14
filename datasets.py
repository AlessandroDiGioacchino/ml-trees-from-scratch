
import numpy as np
import pandas as pd

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile



MUSHROOM_DATASET_URL = (
  'https://archive.ics.uci.edu/static/public/848/secondary+mushroom+dataset.zip'
)


def download_mushroom_dataset( data_dir: str | Path = '.' ) -> Path:
  root = Path( data_dir ).expanduser().resolve()
  root.mkdir( parents=True, exist_ok=True )

  expected_csv = root / 'MushroomDataset' / 'secondary_data.csv'

  if expected_csv.exists():
    return expected_csv

  zip_path = root / 'secondary+mushroom+dataset.zip'
  if not zip_path.exists():
    print( f'Downloading Mushroom dataset to {zip_path}...' )
    urlretrieve( MUSHROOM_DATASET_URL, zip_path )
    print( f'Downloaded Mushroom dataset to {zip_path}.' )

  with ZipFile( zip_path ) as archive:
    inner_zip_name = 'MushroomDataset.zip'
    inner_zip_path = root / inner_zip_name
    inner_zip_path.write_bytes( archive.read( inner_zip_name ) )

    with ZipFile( inner_zip_path ) as inner_archive:
      print( 'Extracting Mushroom dataset...' )
      inner_archive.extractall( root )
      print( 'Extracted Mushroom dataset.' )

  if expected_csv.exists():
    return expected_csv

  raise FileNotFoundError(
    'The Mushroom dataset archive was downloaded but the expected CSV file '
    f'was not found in {root}. '
    'The archive structure may have changed or the URL may no longer be valid.'
  )


def load_mushroom_dataset( data_dir: str | Path = '.' ) -> tuple[
  np.ndarray, np.ndarray, list[ int ]
]:

  csv_path = download_mushroom_dataset( data_dir )
  mushroom = pd.read_csv( csv_path, sep=';' )

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
