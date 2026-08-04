import numpy as np
from alphagenome_dgx import AlphaGenomeDGX

# 1. Conecta ao servidor local (ou futuramente via IP da DGX)
client = AlphaGenomeDGX(host="10.9.8.193", port=50051)

# 2. Faz a predição com apenas 1 chamada limpa
print("Enviando requisição simples via SDK...")
output = client.predict_variant(
    chromosome="chr22",
    position=36201698,
    ref="A",
    alt="C",
    start=35677410,
    end=36725986,
    ontology_type="UBERON",
    ontology_id=1157
)

# 3. Acessa os arrays diretamente
if output.alternate is not None:
    print(f"Formato do Tensor Alternativo: {output.alternate.shape}")
    print(f"Média de Expressão: {output.alternate.mean():.4f}")

if output.diff is not None:
    print(f"Variância máxima do impacto: {output.diff.max():.4f}")

# 4. Predição de intervalo (16kb)
print("\nPredictInterval (16kb)...")
interval_out = client.predict_interval(
    chromosome="chr22",
    start=35677410,
    end=35677410 + 2**14,
    ontology_type="UBERON",
    ontology_id=1157
)
if interval_out is not None and interval_out.values is not None:
    print(f"Formato do Tensor do Intervalo: {interval_out.values.shape}")
    print(f"Média de Expressão do Intervalo: {interval_out.mean:.4f}")

# 5. Score de variante (usa os scorers recomendados pelo servidor)
print("\nScoreVariant...")
scores = client.score_variant(
    chromosome="chr22",
    position=36201698,
    ref="A",
    alt="C",
    start=35677410,
    end=36725986
)
print(f"Número de scores retornados: {len(scores)}")
for i, score in enumerate(scores):
    if score.values is not None:
        print(f"  Score {i}: shape={score.values.shape}, "
              f"genes={len(score.gene_metadata)}, "
              f"mean={score.mean:.6f}")

# 6. Score de intervalo
print("\nScoreInterval (1MB)...")
interval_scores = client.score_interval(
    chromosome="chr22",
    start=35677410,
    end=35677410 + 2**20
)
print(f"Número de scores de intervalo: {len(interval_scores)}")
for i, score in enumerate(interval_scores):
    if score.values is not None:
        print(f"  Score {i}: shape={score.values.shape}, mean={score.mean:.6f}")

# 7. Predição de sequência (16kb de 'A' alternando com 'C' apenas para teste)
print("\nPredictSequence (16kb sintética)...")
rng = np.random.default_rng(42)
seq = "".join(rng.choice(list("ACGT"), size=2**14))
seq_out = client.predict_sequence(sequence=seq, requested_outputs=None)
if seq_out is not None and seq_out.values is not None:
    print(f"Formato do Tensor da Sequência: {seq_out.values.shape}")
    print(f"Média de Expressão da Sequência: {seq_out.mean:.4f}")
