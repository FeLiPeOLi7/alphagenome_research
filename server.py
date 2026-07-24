from concurrent import futures
import os
import grpc
import numpy as np

# Configurações de ambiente
os.environ["HF_HUB_DISABLE_XET"] = "1"
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MB

from alphagenome.data import genome
from alphagenome_research.model import dna_model
from alphagenome.protos import dna_model_service_pb2_grpc
from alphagenome.protos import dna_model_service_pb2
from alphagenome.protos import dna_model_pb2

def fill_track_data_proto(track_data_proto, track_obj, interval_proto):
    """Converte a matriz NumPy do AlphaGenome para a estrutura Protobuf TrackData"""
    if track_obj is None or getattr(track_obj, 'values', None) is None:
        return

    arr = track_obj.values
    
    # 1. Copia o Shape da matriz
    track_data_proto.values.shape.extend(arr.shape)
    
    # 2. Copia os bytes da matriz Float32 diretamente para o TensorChunk
    track_data_proto.values.array.data = arr.astype(np.float32).tobytes()

    # 3. Copia a Resolução e o Intervalo Genômico
    if hasattr(track_obj, 'resolution') and track_obj.resolution:
        track_data_proto.resolution = int(track_obj.resolution)

    track_data_proto.interval.chromosome = interval_proto.chromosome
    track_data_proto.interval.start = interval_proto.start
    track_data_proto.interval.end = interval_proto.end


def populate_output_proto(output_proto, model_output, interval_proto):
    """Varre todas as faixas (tracks) disponíveis e preenche a resposta Protobuf"""
    if model_output is None:
        return

    # Mapeamento dos tipos de saída suportados pelo AlphaGenome
    track_mapping = [
        ('rna_seq', dna_model_pb2.OUTPUT_TYPE_RNA_SEQ),
        ('cage', dna_model_pb2.OUTPUT_TYPE_CAGE),
        ('atac', dna_model_pb2.OUTPUT_TYPE_ATAC),
        ('dnase', dna_model_pb2.OUTPUT_TYPE_DNASE),
        ('chip_tf', dna_model_pb2.OUTPUT_TYPE_CHIP_TF),
        ('chip_histone', dna_model_pb2.OUTPUT_TYPE_CHIP_HISTONE),
        ('splice_sites', dna_model_pb2.OUTPUT_TYPE_SPLICE_SITES),
        ('splice_site_usage', dna_model_pb2.OUTPUT_TYPE_SPLICE_SITE_USAGE),
        ('splice_junctions', dna_model_pb2.OUTPUT_TYPE_SPLICE_JUNCTIONS),
        ('contact_maps', dna_model_pb2.OUTPUT_TYPE_CONTACT_MAPS),
        ('procap', dna_model_pb2.OUTPUT_TYPE_PROCAP),
    ]

    for attr_name, output_enum in track_mapping:
        if hasattr(model_output, attr_name):
            track_obj = getattr(model_output, attr_name)
            if track_obj is not None:
                output_proto.output_type = output_enum
                fill_track_data_proto(output_proto.track_data, track_obj, interval_proto)
                break  # Processa o primeiro track encontrado no objeto de saída


class AlphaGenomeServer(dna_model_service_pb2_grpc.DnaModelServiceServicer):
    def __init__(self):
        print("[DGX] Carregando o modelo AlphaGenome na GPU...")
        self.model = dna_model.create_from_huggingface('all_folds')
        print("[DGX] Modelo carregado com sucesso na VRAM!")

    def PredictVariant(self, request_iterator, context):
        print("\n[gRPC] Nova requisição PredictVariant recebida...")

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

            # Processa Ontologias
            ontology_list = []
            for term in request.ontology_terms:
                prefix = dna_model_pb2.OntologyType.Name(term.ontology_type).replace("ONTOLOGY_TYPE_", "")
                ontology_list.append(f"{prefix}:{term.id:07d}")

            # Define as saídas requisitadas
            requested_outputs = [dna_model.OutputType.RNA_SEQ]
            if hasattr(request, 'requested_outputs') and request.requested_outputs:
                requested_outputs = [
                    dna_model.OutputType[dna_model_pb2.OutputType.Name(o)]
                    for o in request.requested_outputs
                ]

            print(f"Executando variante {variant.chromosome}:{variant.position} ({variant.reference_bases}->{variant.alternate_bases}) na GPU...")

            outputs = self.model.predict_variant(
                interval=interval,
                variant=variant,
                ontology_terms=ontology_list,
                requested_outputs=requested_outputs,
            )

            response = dna_model_service_pb2.PredictVariantResponse()

            # Preenche Referência e Alternativa dinamicamente
            if hasattr(outputs, 'reference') and outputs.reference:
                populate_output_proto(response.reference_output, outputs.reference, request.interval)

            if hasattr(outputs, 'alternate') and outputs.alternate:
                populate_output_proto(response.alternate_output, outputs.alternate, request.interval)

            print("[gRPC] Transmitindo matrizes de inferência para o cliente...")
            yield response


def serve():
    options = [
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=options
    )
    dna_model_service_pb2_grpc.add_DnaModelServiceServicer_to_server(
        AlphaGenomeServer(), server
    )
    
    # Escuta em todas as interfaces de rede da DGX (0.0.0.0)
    port = "50051"
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    print(f"\nServidor AlphaGenome DGX ativo e escutando em 0.0.0.0:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
