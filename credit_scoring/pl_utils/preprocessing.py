from omegaconf import DictConfig
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm


ID_COLUMN_NAME = "id"


class Preprocesser:
    """Preprocessing data class"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.train_files_dir = Path(config.train)
        self.test_files_dir = Path(config.test)
        self.train_target_path = Path(config.train_target)
        self.rn_threshold = config.rn_threshold

    def paths_dict(self) -> dict:
        dct = {}

        train_files_list = list(self.train_files_dir.glob("**/*.pq"))
        test_files_list = list(self.test_files_dir.glob("**/*.pq"))

        dct["train"] = sorted(train_files_list)
        dct["test"] = sorted(test_files_list)
        dct["train_target"] = self.train_target_path
        return dct

    @staticmethod
    def read_table(path: Path,
                   tech_cols: list,
                   id_col_name: str) -> Tuple[pd.DataFrame, str]:
        """Reading table with data placed at the given path.

        Args:
            path (pathlib.Path): Path of the table.

        Raises:
            ValueError: File format should be one of the
                types ('.pq', '.csv').

        Returns:
            Tuple[pd.DataFrame, str]: Table located on given
                path with list of features
        """
        # Attempt to read the table
        if path.suffix == ".pq":
            table = pq.read_table(path).to_pandas()
        elif path.suffix == ".csv":
            table = pd.read_csv(path)
            if table.columns[0] == "Unnamed: 0":
                table.drop(columns=["Unnamed: 0"], inplace=True)
        else:
            raise ValueError("File format should be one of the types ('.pq' or '.csv')")

        cols = list(table.columns.values)
        types_dict = dict.fromkeys(cols)
        cols_int64 = []

        # Astype some columns to int8
        for col in cols:
            if abs(table[col].max()) > 128:
                cols_int64.append(col)
        for key in types_dict.keys():
            if key in cols_int64:
                types_dict[key] = "int64"
            else:
                types_dict[key] = "int8"
        table = table.astype(types_dict)

        # remove all columns except feature columns
        for c in tech_cols:
            cols.remove(c)
        cols.remove(id_col_name)
        return table, cols

    @staticmethod
    def generate_df(
        raw_table: pd.DataFrame, 
        rn_threshold: int, 
        feature_columns: List,
        id_column_name: str
    ) -> Dict:
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
        table = raw_table.groupby(id_column_name)

        for i, frame in table:
            diff = max(rn_threshold - frame.shape[0], 0)
            add_rows = np.zeros((diff, table_width))
            source_rows = frame[feature_columns].values
            if frame.shape[0] > rn_threshold:
                source_rows = source_rows[:rn_threshold]
            data_dict[i] = np.vstack((add_rows, source_rows)).astype("int16")
        return data_dict

    def read_train_test_features(self, label: str) -> dict:
        """Generating dict with all train or test data

        Args:
            label (str): Type of data ('train' or 'test')
            rn_threshold (int, optional): Number of each id rows in final
                dict. Defaults to ROWS_THRESHOLD.

        Raises:
            ValueError: Raises if `label` value is wrong.

        Returns:
            Dict: Final dict, which contains data according to
                the given label.
        """
        paths = self.paths_dict()
        paths_list = paths[label]
        if label in ("train", "test"):
            data = {}
            tech_cols = self.config.tech_cols
            id_col = self.config.id_col
            print(f"Starting preprocessing pipeline for {label} data.\n")
            print("Processing the files...\n")
            for path in tqdm.tqdm(paths_list):
                
                table, cols = self.read_table(path, tech_cols, id_col)
                data.update(
                    self.generate_df(
                        table, 
                        self.rn_threshold, 
                        cols,
                        id_col
                    )
                )
        else:
            raise ValueError("Wrong label!")
        return data

    def read_train_target(self) -> pd.DataFrame:
        """Reading file with train target

        Raises:
            ValueError: Raises if path in configs doesn't
                belong to `.csv` file.

        Returns:
            pd.DataFrame: Contains target values
        """
        paths = self.paths_dict()
        df_path = paths["train_target"]
        if df_path.suffix != ".csv":
            raise ValueError("Path should belong to .csv file!")

        table = pd.read_csv(df_path)
        if table.columns[0] == "Unnamed: 0":
            table.drop(columns=["Unnamed: 0"], inplace=True)
        return table


def main():
    print("Do you want to preprocess data? [y/n]")
    ans = input()
    if ans == "y":
        preproc = Preprocesser()

        print("Type label of the data [train/test]")
        typ = input()
        print("Type id you want to show [int]")
        id = int(input())
        print()
        frame = preproc.read_train_test_features(typ)[id]
        print(frame[:5])
        print(frame.shape)
    return 0


if __name__ == "__main__":
    main()
