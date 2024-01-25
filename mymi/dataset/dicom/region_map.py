import numpy as np
import os
import pandas as pd
import re
from typing import List, Optional, Tuple, Union

from mymi.regions import is_region
from mymi.types import PatientID

class RegionMap:
    def __init__(
        self,
        data: pd.DataFrame):
        self.__data = data

    @staticmethod
    def load(filepath: str) -> Optional['RegionMap']:
        if os.path.exists(filepath):
            # Load file.
            df = pd.read_csv(filepath)

            # Convert patient ID only/except to array.
            cols = ['except', 'only']
            for col in cols:
                if col in df.columns:
                    def split_fn(e: Union[float, int, str]) -> List[str]:
                        if isinstance(e, float):
                            if np.isnan(e):         # Handle empty cells.
                                return []
                            else:                   # Handle patient IDs parsed as floats by pandas.
                                return [str(int(e))]
                        elif isinstance(e, str):   # Split comma-separated patient IDs.
                            return e.split(',')
                        else:
                            raise ValueError(f"Can't split unrecognised type '{type(e)}'.")
                    df[col] = df[col].apply(split_fn)
                else:
                    df[col] = [[]] * len(df)

            # # Check that internal region names are entered correctly.
            # for region in map_df.internal:
            #     if not is_region(region):
            #         raise ValueError(f"Error in region map. '{region}' is not an internal region.")
            
            return RegionMap(df)
        else:
            return None

    @property
    def data(self) -> pd.DataFrame:
        return self.__data

    def to_internal(
        self,
        region: str,
        pat_id: Optional[PatientID] = None) -> Tuple[str, int]:

        # Iterate over map rows.
        match = None
        priority = -np.inf
        for _, row in self.__data.iterrows():
            # Check only/except rules that map regions to specific patients.
            excpt = row['except']
            if pat_id in excpt:
                continue
            only = row['only']
            if len(only) > 0 and pat_id not in only: 
                continue

            # Add case sensitivity to regexp match args.
            args = []
            if 'case' in row:
                case = row['case']
                if not np.isnan(case) and not case:
                    args += [re.IGNORECASE]
            else:
                args += [re.IGNORECASE]
                
            # Perform match.
            if re.match(row['before'], region, *args):
                if 'priority' in row and not np.isnan(row['priority']):
                    if row['priority'] > priority:
                        match = row['after']
                        priority = row['priority']
                else:
                    match = row['after']

        if match is None:
            match = region
        
        return match, priority
