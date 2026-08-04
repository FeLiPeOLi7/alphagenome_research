import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
import time
import unittest
import numpy as np
from absl import flags

try:
    flags.FLAGS(sys.argv[:1], known_only=True)
except Exception:
    pass

from alphagenome.data import genome
from alphagenome.models import dna_client as cloud_client
from alphagenome.models import dna_model
from alphagenome.models import dna_output

from alphagenome_dgx import AlphaGenomeDGX

CLOUD_API_KEY = os.environ.get("ALPHAGENOME_KEY", "")
LOCAL_HOST = os.environ.get("ALPHAGENOME_HOST", "10.9.8.193")
LOCAL_PORT = int(os.environ.get("ALPHAGENOME_PORT", "50051"))

SEQUENCE_LENGTH = 16384
CHROMOSOME = "chr22"
VARIANT_POS = 36201698
VARIANT_START = VARIANT_POS - SEQUENCE_LENGTH // 2
VARIANT_END = VARIANT_POS + SEQUENCE_LENGTH // 2
REF_BASE = "A"
ALT_BASE = "C"

ONTOLOGY_TERMS = ["UBERON:0001157"]
REQUESTED_OUTPUTS = [dna_output.OutputType.RNA_SEQ]


def extract_track_values(output_obj, output_type):
    """Extract numpy values from a cloud/local Output object for a given OutputType."""
    track = output_obj.get(output_type)
    if track is None:
        return None
    return track.values


class TestCloudVsLocalPredictVariant(unittest.TestCase):
    """Compare predict_variant between AlphaGenome cloud and local gRPC server."""

    @classmethod
    def setUpClass(cls):
        if not CLOUD_API_KEY:
            raise unittest.SkipTest("ALPHAGENOME_KEY env var not set")

        print("\n[SETUP] Connecting to AlphaGenome cloud...")
        t0 = time.perf_counter()
        cls.cloud = cloud_client.create(api_key=CLOUD_API_KEY)
        print(f"  Cloud client ready in {time.perf_counter() - t0:.2f}s")

        print(f"[SETUP] Connecting to local gRPC server at {LOCAL_HOST}:{LOCAL_PORT}...")
        cls.local = AlphaGenomeDGX(host=LOCAL_HOST, port=LOCAL_PORT)

        cls.interval = genome.Interval(
            chromosome=CHROMOSOME, start=VARIANT_START, end=VARIANT_END
        )
        cls.variant = genome.Variant(
            chromosome=CHROMOSOME,
            position=VARIANT_POS,
            reference_bases=REF_BASE,
            alternate_bases=ALT_BASE,
        )

        print("[SETUP] Warm-up: running predict_variant on cloud...")
        t0 = time.perf_counter()
        cls.cloud.predict_variant(
            interval=cls.interval,
            variant=cls.variant,
            organism=dna_model.Organism.HOMO_SAPIENS,
            requested_outputs=REQUESTED_OUTPUTS,
            ontology_terms=ONTOLOGY_TERMS,
        )
        print(f"  Cloud warm-up done in {time.perf_counter() - t0:.2f}s")

        print("[SETUP] Warm-up: running predict_variant on local server...")
        t0 = time.perf_counter()
        cls.local.predict_variant(
            chromosome=CHROMOSOME,
            position=VARIANT_POS,
            ref=REF_BASE,
            alt=ALT_BASE,
            start=VARIANT_START,
            end=VARIANT_END,
            requested_outputs=REQUESTED_OUTPUTS,
        )
        print(f"  Local warm-up done in {time.perf_counter() - t0:.2f}s")

    def test_predict_variant_reference(self):
        """Cloud and local reference outputs must match numerically."""
        print("\n[TEST] predict_variant - reference")

        t0 = time.perf_counter()
        cloud_out = self.cloud.predict_variant(
            interval=self.interval,
            variant=self.variant,
            organism=dna_model.Organism.HOMO_SAPIENS,
            requested_outputs=REQUESTED_OUTPUTS,
            ontology_terms=ONTOLOGY_TERMS,
        )
        t_cloud = time.perf_counter() - t0

        t0 = time.perf_counter()
        local_out = self.local.predict_variant(
            chromosome=CHROMOSOME,
            position=VARIANT_POS,
            ref=REF_BASE,
            alt=ALT_BASE,
            start=VARIANT_START,
            end=VARIANT_END,
            requested_outputs=REQUESTED_OUTPUTS,
        )
        t_local = time.perf_counter() - t0

        cloud_ref = extract_track_values(cloud_out.reference, dna_output.OutputType.RNA_SEQ)
        local_ref = local_out.reference

        self.assertIsNotNone(cloud_ref, "Cloud reference is None")
        self.assertIsNotNone(local_ref, "Local reference is None")

        if local_ref.shape != cloud_ref.shape and local_ref.size == cloud_ref.size:
            local_ref = local_ref.reshape(cloud_ref.shape)

        self.assertEqual(cloud_ref.shape, local_ref.shape,
                         f"Shape mismatch: cloud={cloud_ref.shape} local={local_ref.shape}")

        max_diff = float(np.max(np.abs(cloud_ref - local_ref)))
        np.testing.assert_allclose(cloud_ref, local_ref, rtol=1e-3, atol=1e-3,
                                   err_msg=f"Max diff: {max_diff}")

        print(f"  [OK] Reference match | shape={cloud_ref.shape} | max_diff={max_diff:.6e}")
        print(f"  [TIME] Cloud: {t_cloud:.2f}s | Local: {t_local:.2f}s")

    def test_predict_variant_alternate(self):
        """Cloud and local alternate outputs must match numerically."""
        print("\n[TEST] predict_variant - alternate")

        cloud_out = self.cloud.predict_variant(
            interval=self.interval,
            variant=self.variant,
            organism=dna_model.Organism.HOMO_SAPIENS,
            requested_outputs=REQUESTED_OUTPUTS,
            ontology_terms=ONTOLOGY_TERMS,
        )
        local_out = self.local.predict_variant(
            chromosome=CHROMOSOME,
            position=VARIANT_POS,
            ref=REF_BASE,
            alt=ALT_BASE,
            start=VARIANT_START,
            end=VARIANT_END,
            requested_outputs=REQUESTED_OUTPUTS,
        )

        cloud_alt = extract_track_values(cloud_out.alternate, dna_output.OutputType.RNA_SEQ)
        local_alt = local_out.alternate

        self.assertIsNotNone(cloud_alt, "Cloud alternate is None")
        self.assertIsNotNone(local_alt, "Local alternate is None")

        if local_alt.shape != cloud_alt.shape and local_alt.size == cloud_alt.size:
            local_alt = local_alt.reshape(cloud_alt.shape)

        self.assertEqual(cloud_alt.shape, local_alt.shape,
                         f"Shape mismatch: cloud={cloud_alt.shape} local={local_alt.shape}")

        max_diff = float(np.max(np.abs(cloud_alt - local_alt)))
        np.testing.assert_allclose(cloud_alt, local_alt, rtol=1e-3, atol=1e-3,
                                   err_msg=f"Max diff: {max_diff}")

        print(f"  [OK] Alternate match | shape={cloud_alt.shape} | max_diff={max_diff:.6e}")

    def test_predict_variant_diff(self):
        """Cloud and local variant diffs (alt - ref) must match."""
        print("\n[TEST] predict_variant - diff (alt - ref)")

        cloud_out = self.cloud.predict_variant(
            interval=self.interval,
            variant=self.variant,
            organism=dna_model.Organism.HOMO_SAPIENS,
            requested_outputs=REQUESTED_OUTPUTS,
            ontology_terms=ONTOLOGY_TERMS,
        )
        local_out = self.local.predict_variant(
            chromosome=CHROMOSOME,
            position=VARIANT_POS,
            ref=REF_BASE,
            alt=ALT_BASE,
            start=VARIANT_START,
            end=VARIANT_END,
            requested_outputs=REQUESTED_OUTPUTS,
        )

        cloud_ref = extract_track_values(cloud_out.reference, dna_output.OutputType.RNA_SEQ)
        cloud_alt = extract_track_values(cloud_out.alternate, dna_output.OutputType.RNA_SEQ)
        cloud_diff = cloud_alt - cloud_ref

        local_diff = local_out.diff

        self.assertIsNotNone(local_diff, "Local diff is None")

        if local_diff.shape != cloud_diff.shape and local_diff.size == cloud_diff.size:
            local_diff = local_diff.reshape(cloud_diff.shape)

        max_diff = float(np.max(np.abs(cloud_diff - local_diff)))
        np.testing.assert_allclose(cloud_diff, local_diff, rtol=1e-3, atol=1e-3,
                                   err_msg=f"Max diff: {max_diff}")

        print(f"  [OK] Diff match | shape={cloud_diff.shape} | max_diff={max_diff:.6e}")


def _random_dna_sequence(length, seed=42):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(np.array(list("ACGT")), size=length))


class TestCloudVsLocalPredictSequence(unittest.TestCase):
    """Compare predict_sequence between AlphaGenome cloud and local gRPC server."""

    @classmethod
    def setUpClass(cls):
        if not CLOUD_API_KEY:
            raise unittest.SkipTest("ALPHAGENOME_KEY env var not set")

        cls.cloud = cloud_client.create(api_key=CLOUD_API_KEY)
        cls.local = AlphaGenomeDGX(host=LOCAL_HOST, port=LOCAL_PORT)

        cls.sequence = _random_dna_sequence(SEQUENCE_LENGTH)

        print("[SETUP] Warm-up: running predict_sequence on cloud...")
        t0 = time.perf_counter()
        cls.cloud.predict_sequence(
            sequence=cls.sequence,
            organism=dna_model.Organism.HOMO_SAPIENS,
            requested_outputs=REQUESTED_OUTPUTS,
            ontology_terms=ONTOLOGY_TERMS,
        )
        print(f"  Cloud warm-up done in {time.perf_counter() - t0:.2f}s")

        print("[SETUP] Warm-up: running predict_sequence on local server...")
        t0 = time.perf_counter()
        cls.local.predict_sequence(
            sequence=cls.sequence,
            requested_outputs=REQUESTED_OUTPUTS,
        )
        print(f"  Local warm-up done in {time.perf_counter() - t0:.2f}s")

    def test_predict_sequence_values(self):
        """Cloud and local sequence predictions must match numerically."""
        print("\n[TEST] predict_sequence - values")

        t0 = time.perf_counter()
        cloud_out = self.cloud.predict_sequence(
            sequence=self.sequence,
            organism=dna_model.Organism.HOMO_SAPIENS,
            requested_outputs=REQUESTED_OUTPUTS,
            ontology_terms=ONTOLOGY_TERMS,
        )
        t_cloud = time.perf_counter() - t0

        t0 = time.perf_counter()
        local_out = self.local.predict_sequence(
            sequence=self.sequence,
            requested_outputs=REQUESTED_OUTPUTS,
        )
        t_local = time.perf_counter() - t0

        cloud_vals = extract_track_values(cloud_out, dna_output.OutputType.RNA_SEQ)
        local_vals = local_out.values

        self.assertIsNotNone(cloud_vals, "Cloud output is None")
        self.assertIsNotNone(local_vals, "Local output is None")

        if local_vals.shape != cloud_vals.shape and local_vals.size == cloud_vals.size:
            local_vals = local_vals.reshape(cloud_vals.shape)

        self.assertEqual(cloud_vals.shape, local_vals.shape,
                         f"Shape mismatch: cloud={cloud_vals.shape} local={local_vals.shape}")

        max_diff = float(np.max(np.abs(cloud_vals - local_vals)))
        np.testing.assert_allclose(cloud_vals, local_vals, rtol=1e-3, atol=1e-3,
                                   err_msg=f"Max diff: {max_diff}")

        print(f"  [OK] Sequence match | shape={cloud_vals.shape} | max_diff={max_diff:.6e}")
        print(f"  [TIME] Cloud: {t_cloud:.2f}s | Local: {t_local:.2f}s")

    def test_predict_sequence_different_dna(self):
        """Compare on a second synthetic random DNA sequence with a different seed."""
        print("\n[TEST] predict_sequence - different synthetic DNA")

        sequence = _random_dna_sequence(SEQUENCE_LENGTH, seed=99)

        cloud_out = self.cloud.predict_sequence(
            sequence=sequence,
            organism=dna_model.Organism.HOMO_SAPIENS,
            requested_outputs=REQUESTED_OUTPUTS,
            ontology_terms=ONTOLOGY_TERMS,
        )
        local_out = self.local.predict_sequence(
            sequence=sequence,
            requested_outputs=REQUESTED_OUTPUTS,
        )

        cloud_vals = extract_track_values(cloud_out, dna_output.OutputType.RNA_SEQ)
        local_vals = local_out.values

        self.assertIsNotNone(cloud_vals, "Cloud output is None for different DNA")
        self.assertIsNotNone(local_vals, "Local output is None for different DNA")

        if local_vals.shape != cloud_vals.shape and local_vals.size == cloud_vals.size:
            local_vals = local_vals.reshape(cloud_vals.shape)

        self.assertEqual(cloud_vals.shape, local_vals.shape,
                         f"Shape mismatch: cloud={cloud_vals.shape} local={local_vals.shape}")

        max_diff = float(np.max(np.abs(cloud_vals - local_vals)))
        np.testing.assert_allclose(cloud_vals, local_vals, rtol=1e-3, atol=1e-3,
                                   err_msg=f"Max diff: {max_diff}")

        print(f"  [OK] Different DNA match | shape={cloud_vals.shape} | max_diff={max_diff:.6e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
