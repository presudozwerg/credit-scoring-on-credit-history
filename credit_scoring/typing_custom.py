import numpy.typing as npt
from torchtyping import TensorType
from typing import Tuple

DataType = TensorType | Tuple[TensorType, npt.NDArray]