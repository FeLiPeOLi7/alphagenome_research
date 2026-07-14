# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for type annotations."""

from typing import TypeAlias
import jax.numpy as jnp
import numpy as np

# Type alias for JAX or NumPy arrays. Unlike jax.ArrayLike or np.ArrayLike, this
# is a type alias for the array types themselves, rather than a type that can
# be converted to an array.
ArrayType: TypeAlias = jnp.ndarray | np.ndarray
