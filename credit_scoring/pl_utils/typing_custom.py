from typing import Tuple

import numpy.typing as npt
from torchtyping import TensorType


DataType = TensorType | Tuple[TensorType, npt.NDArray]
