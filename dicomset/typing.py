from jaxtyping import Bool, Float, Int
import numpy as np
import pydicom as dcm
from typing import Literal, Tuple

# Basic types.
AffineMatrix2D = Float[np.ndarray, "3 3"]
AffineMatrix3D = Float[np.ndarray, "4 4"]
Box2D = Float[np.ndarray, "2 2"]
Box3D = Float[np.ndarray, "2 3"]
CtDicom = dcm.dataset.FileDataset
DatasetID = str
DatasetType = Literal['dicom', 'nifti', 'raw', 'training']
DicomModality = Literal['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
DirPath = str
FilePath = str
GroupID = int   # For patient groups, for now.
Image2D = Float[np.ndarray, "H W"]
Image3D = Float[np.ndarray, "D H W"]
LabelImage2D = Bool[np.ndarray, "H W"]
LabelImage3D = Bool[np.ndarray, "D H W"]
LandmarkID = str
NiftiModality = Literal['ct', 'dose', 'landmarks', 'mr', 'plan', 'regions']
Number = int | float
PatientID = str
Pixel = Tuple[int, int]
Point2D = Tuple[Number, Number] | Float[np.ndarray, "2"]
Point3D = Tuple[Number, Number, Number] | Float[np.ndarray, "3"]
RegionID = str
SeriesID = str
Size2D = Tuple[int, int] | Int[np.ndarray, "2"]
Size3D = Tuple[int, int, int] | Int[np.ndarray, "3"]
Spacing2D = Tuple[Number, Number] | Float[np.ndarray, "2"]
Spacing3D = Tuple[Number, Number, Number] | Float[np.ndarray, "3"]
SpatialDim = Literal[2, 3]
StudyID = str
Voxel = Tuple[int, int, int]

# First-order types.
AffineMatrix = AffineMatrix2D | AffineMatrix3D
Box = Box2D | Box3D
Image = Image2D | Image3D
LabelImage = LabelImage2D | LabelImage3D 
Point = Point2D | Point3D
Size = Size2D | Size3D
Spacing = Spacing2D | Spacing3D
