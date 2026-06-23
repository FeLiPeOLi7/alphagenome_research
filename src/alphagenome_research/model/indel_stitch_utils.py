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

"""Utilities for stitching predictions for indel variant scoring.

Utils used to implement intermediate activation stitching for insertions and
deletions. This improves the numerics when scoring indels. This is similar to
the approach detailed here but deviates slightly due to stitching activations at
multiple resolutions:

Korsakova, A., Srivastava, D. and Kelley, D.R., 2025. Shift augmentation
improves DNA convolutional neural network indel effect predictions. bioRxiv,
pp.2025-04.
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32, Int32  # pylint: disable=g-importing-member, g-multiple-import


@chex.dataclass(frozen=True)
class IndelStitchInput:
  """Indel stitching input supporting insertions, deletions, and SNPs.

  Attributes:
    left_sequence: Left-aligned one-hot DNA sequence. [B, S, 4]
    right_sequence: Right-aligned one-hot DNA sequence. [B, S, 4]
    left_shift: Left shift in base pairs. [B, 1]
    right_shift: Right shift in base pairs. [B, 1]
    stitch_index: Stitch index in base pairs. [B, 1]
    indel_start_index: Indel start index in base pairs. [B, 1]
    is_deletion: Boolean flag indicating if the variant is a deletion. [B]
  """

  left_sequence: Float32[Array, 'B S 4']
  right_sequence: Float32[Array, 'B S 4']
  left_shift: Int32[Array, 'B 1']
  right_shift: Int32[Array, 'B 1']
  stitch_index: Int32[Array, 'B 1']
  indel_start_index: Int32[Array, 'B 1']
  is_deletion: Bool[Array, 'B']


def stitch_sequence_base_resolution(
    *,
    left: Float32[Array, 'B S C'],
    right: Float32[Array, 'B S C'],
    left_shift: Int32[Array, 'B 1'],
    right_shift: Int32[Array, 'B 1'],
    stitch_index: Int32[Array, 'B 1'],
) -> Float32[Array, 'B S C']:
  """Stitches 1D embeddings at 1bp resolution (no binning)."""

  def stitch(l_b, r_b, l_shift, r_shift, stitch_idx):
    left_shifted = jnp.roll(l_b, -l_shift, axis=0)
    right_shifted = jnp.roll(r_b, r_shift, axis=0)
    index_array = jnp.arange(l_b.shape[0])
    mask = index_array >= stitch_idx
    return jnp.where(mask[..., None], right_shifted, left_shifted)

  return jax.vmap(stitch)(
      left, right, left_shift[:, 0], right_shift[:, 0], stitch_index[:, 0]
  )


def compute_stitch_params_coarse_resolution(
    *,
    stitch_idx: Int32[Array, ''],
    start_idx: Int32[Array, ''],
    bin_size: int,
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
  """Compute stitch bin and shift for stitching at coarser resolutions."""
  start_bin = start_idx // bin_size
  end_bin = (stitch_idx - 1) // bin_size
  stitch_bin = end_bin
  shift_bins = end_bin - start_bin
  return stitch_bin, shift_bins


def stitch_sequence_coarse_resolution(
    *,
    left: Float32[Array, 'B S C'],
    right: Float32[Array, 'B S C'],
    left_shift: Int32[Array, 'B 1'],
    right_shift: Int32[Array, 'B 1'],
    stitch_index: Int32[Array, 'B 1'],
    indel_start_index: Int32[Array, 'B 1'],
    bin_size: int,
) -> Float32[Array, 'B S C']:
  """Stitches 1D embeddings with boundary-aware binning logic."""

  def stitch(l_b, r_b, stitch_idx, start_idx, l_shift, r_shift):
    stitch_bin, shift_bins = compute_stitch_params_coarse_resolution(
        stitch_idx=stitch_idx, start_idx=start_idx, bin_size=bin_size
    )

    # Ensure stitch_bin is within bounds.
    stitch_bin = jnp.maximum(0, jnp.minimum(stitch_bin, l_b.shape[0]))

    left_roll = jnp.where(l_shift > 0, -shift_bins, 0)
    right_roll = jnp.where(r_shift > 0, shift_bins, 0)

    left_shifted = jnp.roll(l_b, left_roll, axis=0)
    right_shifted = jnp.roll(r_b, right_roll, axis=0)

    index_array = jnp.arange(l_b.shape[0])
    mask = index_array >= stitch_bin
    return jnp.where(mask[..., None], right_shifted, left_shifted)

  return jax.vmap(stitch)(
      left,
      right,
      stitch_index[:, 0],
      indel_start_index[:, 0],
      left_shift[:, 0],
      right_shift[:, 0],
  )


def stitch_2d_contact_maps(
    *,
    left: Float32[Array, 'B S S C'],
    right: Float32[Array, 'B S S C'],
    left_shift: Int32[Array, 'B 1'],
    right_shift: Int32[Array, 'B 1'],
    stitch_index: Int32[Array, 'B 1'],
    indel_start_index: Int32[Array, 'B 1'],
    bin_size: int,
) -> Float32[Array, 'B S S C']:
  """Stitches 2D contact map embeddings with boundary-aware binning logic."""

  def stitch_2d(l_b, r_b, stitch_idx, start_idx, l_shift, r_shift):
    stitch_bin, shift_bins = compute_stitch_params_coarse_resolution(
        stitch_idx=stitch_idx, start_idx=start_idx, bin_size=bin_size
    )

    stitch_bin = jnp.maximum(0, jnp.minimum(stitch_bin, l_b.shape[0]))

    left_roll = jnp.where(l_shift > 0, -shift_bins, 0)
    right_roll = jnp.where(r_shift > 0, shift_bins, 0)

    left_shifted = jnp.roll(l_b, left_roll, axis=(0, 1))
    right_shifted = jnp.roll(r_b, right_roll, axis=(0, 1))

    index_array = jnp.arange(l_b.shape[0])
    mask_1d = index_array >= stitch_bin
    mask_2d = mask_1d[..., None] * mask_1d[None, ...]
    return jnp.where(mask_2d[..., None], right_shifted, left_shifted)

  return jax.vmap(stitch_2d)(
      left,
      right,
      stitch_index[:, 0],
      indel_start_index[:, 0],
      left_shift[:, 0],
      right_shift[:, 0],
  )
