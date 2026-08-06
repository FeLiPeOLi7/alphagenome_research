import numpy as np
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers as variant_scorers_lib

# 1. Conecta ao servidor da DGX via cliente oficial
client = dna_client.create(
    api_key="dummy_key",
    address="10.9.8.193:50051"
)

# Intervalo e variante de teste (exemplo do README)
interval_1mb = genome.Interval(
    chromosome="chr22",
    start=35677410,
    end=36725986
)
variant = genome.Variant(
    chromosome="chr22",
    position=36201698,
    reference_bases="A",
    alternate_bases="C",
)

# 2. Predição de variante (1MB, ref vs. alt)
print("\nPredictVariant (1MB)...")
variant_out = client.predict_variant(
    interval=interval_1mb,
    variant=variant,
    requested_outputs=[dna_client.OutputType.RNA_SEQ],
    ontology_terms=["UBERON:0001157"]
)

ref_track = variant_out.reference.rna_seq
alt_track = variant_out.alternate.rna_seq

if ref_track is not None and ref_track.values is not None:
    print(f"  Reference: shape={ref_track.values.shape} "
          f"mean={ref_track.values.mean():.4f}")

if alt_track is not None and alt_track.values is not None:
    print(f"  Alternate: shape={alt_track.values.shape} "
          f"mean={alt_track.values.mean():.4f}")

diff = variant_out.alternate - variant_out.reference
if diff.rna_seq is not None and diff.rna_seq.values is not None:
    print(f"  Impacto (alt-ref): shape={diff.rna_seq.values.shape} "
          f"max|Δ|={np.abs(diff.rna_seq.values).max():.6f}")

# 4. Predição de intervalo (16kb)
print("\nPredictInterval (16kb)...")
interval_out = client.predict_interval(
    interval=genome.Interval(
        chromosome="chr22",
        start=35677410,
        end=35677410 + 2**14
    ),
    requested_outputs=[dna_client.OutputType.RNA_SEQ],
    ontology_terms=["UBERON:0001157", "CL:0000540"]
)

rna_seq_track = interval_out.rna_seq

if rna_seq_track is not None and rna_seq_track.values is not None:
    # .values contém o ndarray do NumPy
    print(f"Formato do Tensor: {rna_seq_track.values.shape}")
    print(f"Média de Expressão: {rna_seq_track.values.mean():.4f}")

# 5. Score de variante (2 scorers explícitos para teste rápido)
print("\nScoreVariant (1MB, 2 scorers)...")
scores = client.score_variant(
    interval=interval_1mb,
    variant=variant,
    variant_scorers=[
        variant_scorers_lib.GeneMaskLFCScorer(
            requested_output=dna_client.OutputType.RNA_SEQ
        ),
        variant_scorers_lib.CenterMaskScorer(
            requested_output=dna_client.OutputType.RNA_SEQ,
            width=501,
            aggregation_type=(
                variant_scorers_lib.AggregationType.DIFF_LOG2_SUM
            ),
        ),
    ],
)
print(f"Número de scores retornados: {len(scores)}")
for i, score in enumerate(scores):
    print(f"  Score {i}: X={score.X.shape} "
          f"genes={score.n_obs} tracks={score.n_vars} "
          f"mean={score.X.mean():.6f}")

# 6. Score de intervalo
print("\nScoreInterval (1MB)...")
interval_scores = client.score_interval(
    interval=interval_1mb
)
print(f"Número de scores de intervalo: {len(interval_scores)}")
for i, score in enumerate(interval_scores):
    print(f"  Score {i}: X={score.X.shape} "
          f"genes={score.n_obs} tracks={score.n_vars} "
          f"mean={score.X.mean():.6f}")

# 7. Predição de sequência (16kb de bases aleatórias apenas para teste)
print("\nPredictSequence (16kb sintética)...")
rng = np.random.default_rng(42)
seq = "".join(rng.choice(list("ACGT"), size=2**14))
seq_out = client.predict_sequence(
    sequence=seq,
    requested_outputs=[dna_client.OutputType.RNA_SEQ],
    ontology_terms=["UBERON:0001157"],
)
seq_track = seq_out.rna_seq
if seq_track is not None and seq_track.values is not None:
    print(f"Formato do Tensor da Sequência: {seq_track.values.shape}")
    print(f"Média de Expressão da Sequência: {seq_track.values.mean():.4f}")
