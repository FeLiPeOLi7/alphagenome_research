from concurrent import futures
import os
import grpc
import numpy as np

# Forçar download estável no HF Hub
os.environ["HF_HUB_DISABLE_XET"] = "1"

MAX_MESSAGE_LENGTH = 100 * 1024 * 1024

options = [
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
]

from alphagenome.data import genome
from alphagenome_research.model import dna_model
from alphagenome.protos import dna_model_service_pb2_grpc
from alphagenome.protos import dna_model_service_pb2
from alphagenome.protos import dna_model_pb2

def fill_track_data_proto(track_data_proto, track_obj, interval_proto):
    """Converte a matriz NumPy do AlphaGenome para a estrutura Protobuf TrackData"""
    if track_obj is None or track_obj.values is None:
        return

    arr = track_obj.values
    
    # 1. Preenche as dimensões (shape)
    track_data_proto.values.shape.extend(arr.shape)
    
    # 2. Atribui os bytes diretamente no campo array.data (TensorChunk)
    track_data_proto.values.array.data = arr.astype(np.float32).tobytes()

    # 3. Preenche resolução e intervalo genômico
    if hasattr(track_obj, 'resolution') and track_obj.resolution:
        track_data_proto.resolution = int(track_obj.resolution)

    track_data_proto.interval.chromosome = interval_proto.chromosome
    track_data_proto.interval.start = interval_proto.start
    track_data_proto.interval.end = interval_proto.end

class AlphaGenomeServer(dna_model_service_pb2_grpc.DnaModelServiceServicer):
    def __init__(self):
        print("⚡ Carregando o modelo AlphaGenome na GPU da DGX...")
        self.model = dna_model.create_from_huggingface('all_folds')
        print("✅ Modelo carregado e pronto na VRAM!")

    def PredictVariant(self, request_iterator, context):
        print("\n📥 Requisição PredictVariant recebida...")

        for request in request_iterator:
            interval = genome.Interval(
                chromosome=request.interval.chromosome,
                start=request.interval.start,
                end=request.interval.end
            )
            variant = genome.Variant(
                chromosome=request.variant.chromosome,
                position=request.variant.position,
                reference_bases=request.variant.reference_bases,
                alternate_bases=request.variant.alternate_bases
            )

            # Extrai os termos ontológicos
            ontology_list = []
            for term in request.ontology_terms:
                prefix = dna_model_pb2.OntologyType.Name(term.ontology_type).replace("ONTOLOGY_TYPE_", "")
                ontology_list.append(f"{prefix}:{term.id:07d}")

            # Define os outputs desejados
            requested_outputs = [dna_model.OutputType.RNA_SEQ]
            if hasattr(request, 'requested_outputs') and request.requested_outputs:
                requested_outputs = [
                    dna_model.OutputType[dna_model_pb2.OutputType.Name(o)]
                    for o in request.requested_outputs
                ]

            print(f"🧬 Processando variante {variant.chromosome}:{variant.position} ({variant.reference_bases}->{variant.alternate_bases}) na GPU...")

            outputs = self.model.predict_variant(
                interval=interval,
                variant=variant,
                ontology_terms=ontology_list,
                requested_outputs=requested_outputs,
            )
            
            print("🔍 Atributos do objeto outputs:", [attr for attr in dir(outputs) if not attr.startswith('_')])
            print("✅ Inferência concluída! Embalando tensores em Protobuf...")

            response = dna_model_service_pb2.PredictVariantResponse()

            # Processa o output de referência
            if hasattr(outputs, 'reference') and outputs.reference:
                if hasattr(outputs.reference, 'rna_seq') and outputs.reference.rna_seq is not None:
                    fill_track_data_proto(response.reference_output.track_data, outputs.reference.rna_seq, request.interval)
                    response.reference_output.output_type = dna_model_pb2.OUTPUT_TYPE_RNA_SEQ

            # Processa o output de alternativa
            if hasattr(outputs, 'alternate') and outputs.alternate:
                if hasattr(outputs.alternate, 'rna_seq') and outputs.alternate.rna_seq is not None:
                    fill_track_data_proto(response.alternate_output.track_data, outputs.alternate.rna_seq, request.interval)
                    response.alternate_output.output_type = dna_model_pb2.OUTPUT_TYPE_RNA_SEQ

            print("📤 Transmitindo tensores para o cliente via gRPC!")
            yield response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    dna_model_service_pb2_grpc.add_DnaModelServiceServicer_to_server(
        AlphaGenomeServer(), server
    )
    port = "50051"
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"\n🚀 Servidor AlphaGenome DGX ativo e escutando na porta {port}!")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
