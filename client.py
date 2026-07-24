import grpc
import numpy as np

from alphagenome.protos import dna_model_service_pb2_grpc
from alphagenome.protos import dna_model_service_pb2
from alphagenome.protos import dna_model_pb2

MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
options = [
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
]

def unpack_tensor(tensor_proto):
    """Reconstrói a matriz NumPy a partir do protobuf Tensor"""
    if not tensor_proto.array.data or not tensor_proto.shape:
        return None
    shape = tuple(tensor_proto.shape)
    arr = np.frombuffer(tensor_proto.array.data, dtype=np.float32)
    return arr.reshape(shape)

def run_client():
    channel = grpc.insecure_channel('localhost:50051', options=options)
    stub = dna_model_service_pb2_grpc.DnaModelServiceStub(channel)

    term = dna_model_pb2.OntologyTerm(
        ontology_type=dna_model_pb2.ONTOLOGY_TYPE_UBERON,
        id=1157
    )

    request = dna_model_service_pb2.PredictVariantRequest(
        interval=dna_model_pb2.Interval(
            chromosome='chr22',
            start=35677410,
            end=36725986
        ),
        variant=dna_model_pb2.Variant(
            chromosome='chr22',
            position=36201698,
            reference_bases='A',
            alternate_bases='C'
        ),
        ontology_terms=[term]
    )

    print("Enviando requisição gRPC para o servidor AlphaGenome...")
    responses = stub.PredictVariant(iter([request]))

    for i, response in enumerate(responses):
        print(f"\n[Pacote {i+1}] Dados recebidos do servidor!")

        ref_data = unpack_tensor(response.reference_output.track_data.values)
        alt_data = unpack_tensor(response.alternate_output.track_data.values)

        if ref_data is not None:
            print(f"Tensor Referência: Shape = {ref_data.shape}, Média = {ref_data.mean():.4f}")
        
        if alt_data is not None:
            print(f"Tensor Alternativa: Shape = {alt_data.shape}, Média = {alt_data.mean():.4f}")

        if ref_data is not None and alt_data is not None:
            diff = alt_data - ref_data
            print(f"Impacto da Variante (Alt - Ref): Min = {diff.min():.4f}, Max = {diff.max():.4f}")
        
        # Salvar o que estiver disponível
        output_file = "variant_predictions.npz"
        np.savez_compressed(
            output_file,
            reference=ref_data if ref_data is not None else np.array([]),
            alternate=alt_data if alt_data is not None else np.array([])
        )
        print(f"Resultados salvos no arquivo: '{output_file}'")

if __name__ == '__main__':
    run_client()
