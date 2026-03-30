from typing import Literal

# Basic types.
DatasetID = str
DatasetType = Literal['dicom', 'nifti', 'raw', 'training']
DirPath = str
FilePath = str
GroupID = int   # For patient groups, for now.
PatientID = str
SeriesID = str
StudyID = str
