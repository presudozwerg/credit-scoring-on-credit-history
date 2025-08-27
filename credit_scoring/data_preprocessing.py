import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm

from pathlib import Path

from constants import *
from typing import Dict, List, Tuple

class PathsDict():
    def __init__(self):
        self.paths = {}

    def make(self,
             data_root_path: Path = DATA_ROOT,
             train_files_folder: str = TRAIN_FILES_FOLDER,
             train_target_file: str = TRAIN_TARGET_FILE,
             test_files_folder: str = TEST_FILES_FOLDER) -> Dict:
        train_folder = data_root_path / train_files_folder
        train_files_list = list(train_folder.glob('**/*.pq'))
        self.paths['train'] = sorted(train_files_list)

        test_folder = data_root_path / test_files_folder
        test_files_list = list(test_folder.glob('**/*.pq'))
        self.paths['test'] = sorted(test_files_list)

        self.paths['train_target'] = Path(data_root_path / train_target_file)
        return self.paths
    
class Preprocesser():
    def __init__(self,
                 paths_dict: dict):
        self.paths_dict = paths_dict

    @staticmethod
    def read_table(path: Path) -> Tuple[pd.DataFrame, str]:
        """Reading table with data which is placed at the given path.

        Args:
            path (pathlib.Path): Path of the table

        Raises:
            ValueError: File format should be one of the types ('.pq', '.csv')

        Returns:
            Tuple[pd.DataFrame, str]: Table located on given path with list of features
        """
        # Attempt to read the table
        if path.suffix == '.pq':
            table = pq.read_table(path).to_pandas()
        elif path.suffix == '.csv':
            table = pd.read_csv(path)
            if table.columns[0] == 'Unnamed: 0':
                table.drop(columns=['Unnamed: 0'], inplace=True)
        else:
            raise ValueError(
                "File format should be one of the types ('.pq' or '.csv')"
            )
        
        cols = list(table.columns.values)
        types_dict = dict.fromkeys(cols)
        cols_int64 = []

        # Astype some columns to int8 
        for col in cols:
            if abs(table[col].max()) > 128:
                cols_int64.append(col)
        for key in types_dict.keys():
            if key in cols_int64:
                types_dict[key] = 'int64'
            else:
                types_dict[key] = 'int8'
        table = table.astype(types_dict)

        for c in TECH_COLS:
            cols.remove(c)
        cols.remove(ID_COLUMN_NAME)
        return table, cols
    
    @staticmethod
    def generate_df(raw_table: pd.DataFrame,
                    rn_threshold: int,
                    feature_columns: List) -> Dict:
        """Generates dict from table with data

        Args:
            raw_table (pd.DataFrame): Table with data
            rn_threshold (int): Maximum rows of each id to stack
            feature_columns (List): List of features names

        Returns:
            Dict: Keys (id) and values (data contributed to these id)
        """
        data_dict = {}
        table_width = len(feature_columns)
        table = raw_table.groupby(ID_COLUMN_NAME)

        for i, frame in table:
            add_rows = np.zeros((max(rn_threshold - frame.shape[0], 0), table_width))
            processed_rows = frame[feature_columns].values
            if frame.shape[0] > rn_threshold:
                processed_rows = processed_rows[:rn_threshold]
            data_dict[i] = np.vstack((add_rows, processed_rows)).astype('int16')
        return data_dict

    def combine_train_test_features(self, 
                                    label: str,
                                    rn_threshold: int = ROWS_THRESHOLD) -> Dict:
        """Generating one dict with all train or test data

        Args:
            label (str): Type of data ('train' or 'test')
            rn_threshold (int, optional): Num of each id rows in final dict. Defaults to ROWS_THRESHOLD.

        Raises:
            ValueError: Raises if `label` value is wrong.

        Returns:
            Dict: Final dict, which contains data according to the given label.
        """
        paths_list = self.paths_dict[label]
        if label in ('train', 'test'):
            data = {}
            print(f'Next would go the process of combining {label} data')
            print('Processing the files...\n')
            for i, path in enumerate(tqdm.tqdm(paths_list)):
                table, cols = self.read_table(path)
                data.update(self.generate_df(table, rn_threshold, cols))
        else:
            raise ValueError('Wrong label!')
        return data
    
    def read_train_target(self) -> pd.DataFrame:
        """Reading files with train target

        Raises:
            ValueError: Raises if path in cofigs doesn't belong to csv file

        Returns:
            pd.DataFrame: Contains target values
        """
        df_path = self.paths_dict['train_target']
        if df_path.suffix != '.csv':
            raise ValueError('Path should belong to .csv file!')
        
        table = pd.read_csv(df_path)
        if table.columns[0] == 'Unnamed: 0':
            table.drop(columns=['Unnamed: 0'], inplace=True)
        return table

def main():
    print('Do you want to make paths dict? [y/n]')
    ans = input()
    if ans == 'y':
        paths_dict = PathsDict().make()
        print(paths_dict)
    else:
        return 0

    print('Do you want to preprocess data? [y/n]')
    ans = input()
    if ans == 'y':
        preproc = Preprocesser(paths_dict)

        print('Type label of the data [train/test]')
        typ = input()
        print('Type id you want to show [int]')
        id = int(input())
        frame = preproc.combine_train_test_features(typ)[id]
        print(frame[:5])
        print(frame.shape)
    return 0

if __name__ == '__main__':
    main()
