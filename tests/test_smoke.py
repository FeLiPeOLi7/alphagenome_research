"""Smoke test rápido: apenas verifica se o pipeline funciona de ponta a ponta.

Usa uma janela de 16kb (largura mínima do modelo) e um único scorer, em vez da
bateria completa do test_sdk.py (1Mb + 20 scorers).

Uso: python test_smoke.py
Saída: [OK]/[FAIL] por etapa; exit code 1 se algo falhar.
"""
import sys
import time

from alphagenome_dgx import AlphaGenomeDGX

CHROM = "chr22"
POS = 36201698          # chr22:36201698 (A->C)
WINDOW = 2**14          # 16kb - menor largura suportada pelo modelo
START = POS - WINDOW // 2
END = START + WINDOW


def run_step(name, fn):
    print(f"Iniciando {name}...")
    start = time.perf_counter()
    try:
        result = fn()
        if not result:
            raise AssertionError("etapa retornou vazio/None")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"[OK]   {name} ({time.perf_counter() - start:.2f}s)")
    return result


def main():
    print(f"Conectando em localhost:50051 (janela 16kb: {START}-{END})...")
    client = AlphaGenomeDGX(host="localhost", port=50051)

    # 1. Predição de variante - prova toda a cadeia (conexão, servidor,
    #    forward do modelo, proto, unpack no cliente).
    output = run_step("predict_variant", lambda: client.predict_variant(
        chromosome=CHROM,
        position=POS,
        ref="A",
        alt="C",
        start=START,
        end=END,
    ))
    if output.alternate is None or output.alternate.size == 0:
        print("[FAIL] predict_variant: alternate vazio")
        sys.exit(1)
    print(f"       alternate shape={output.alternate.shape} mean={output.alternate.mean():.4f}")

    # 2. Score de variante com UM scorer (RNA_SEQ LFC) em vez dos 20 padrão.
    scorer = client.variant_scorer("RNA_SEQ", kind="GENE_MASK_LFC")
    scores = run_step("score_variant (1 scorer)", lambda: client.score_variant(
        chromosome=CHROM,
        position=POS,
        ref="A",
        alt="C",
        start=START,
        end=END,
        variant_scorers=[scorer],
    ))
    if not scores or scores[0].values is None or scores[0].values.size == 0:
        print("[FAIL] score_variant: score vazio")
        sys.exit(1)
    print(f"       score shape={scores[0].values.shape} mean={scores[0].values.mean():.6f}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
