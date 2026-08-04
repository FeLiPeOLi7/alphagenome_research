import time
import jax
from alphagenome.data import genome
from alphagenome_research.model import dna_model

print("🔍 Dispositivos JAX detectados:", jax.devices())

print("\n📥 Baixando e carregando pesos do AlphaGenome (Hugging Face)...")
start = time.time()

# Baixa e carrega o modelo automaticamente na GPU
model = dna_model.create_from_huggingface('all_folds')

print(f"✅ Modelo carregado na GPU em {time.time() - start:.2f} segundos!")

# Definindo intervalo e variante de teste (exemplo do README)
interval = genome.Interval(chromosome='chr22', start=35677410, end=36725986)
variant = genome.Variant(
    chromosome='chr22',
    position=36201698,
    reference_bases='A',
    alternate_bases='C',
)

print("\n🧬 Rodando predição de variante na GPU...")
outputs = model.predict_variant(
    interval=interval,
    variant=variant,
    ontology_terms=['UBERON:0001157'],
    requested_outputs=[dna_model.OutputType.RNA_SEQ],
)

print("🎉 Predição concluída com sucesso!")
print("Estrutura do resultado:", outputs)
