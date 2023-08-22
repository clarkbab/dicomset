from torchio import Transform
from torchio.transforms import Clamp, Compose, RandomAffine, ZNormalization
from typing import Optional, Tuple

from mymi.transforms import Standardise

def get_transforms(
    thresh_high: Optional[float] = None,
    thresh_low: Optional[float] = None,
    use_stand: bool = False,
    use_thresh: bool = False) -> Tuple[Transform, Transform]:

    # Create transforms.
    rotation = (-5, 5)
    translation = (-50, 50)
    scale = (0.8, 1.2)
    transform_train = RandomAffine(
        degrees=rotation,
        scales=scale,
        translation=translation,
        default_pad_value='minimum')
    transform_val = None

    if use_thresh:
        transform_train = Compose([
            transform_train,
            Clamp(out_min=thresh_low, out_max=thresh_high)
        ])
        transform_val = Clamp(out_min=thresh_low, out_max=thresh_high)

    if use_stand:
        stand = Standardise(-832.2, 362.1)
        transform_train = Compose([
            transform_train,
            stand
        ])
        if transform_val is None:
            transform_val = stand
        else:
            transform_val = Compose([
                transform_val,
                stand
            ])

    return transform_train, transform_val