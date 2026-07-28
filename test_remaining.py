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

def safe_has_field(proto, field_name):
    """Verifica com segurança se um campo Protobuf está presente (compatível com proto3)."""
    if proto is None or not hasattr(proto, field_name):
        return False
    try:
        return proto.HasField(field_name)
    except ValueError:
        val = getattr(proto, field_name, None)
        if isinstance(val, (bytes, bytearray)):
            return len(val) > 0
        return val is not None


def unpack_tensor(proto):
    """
    Desempacota recursivamente um Protobuf (ScoreIntervalOutput, ScoreVariantOutput, ModelOutput,
    TrackData, IntervalData, VariantData, Tensor, TensorChunk) para um NumPy array (float32).
    """
    if proto is None:
        return None

    # 1. Se proto for bytes puro
    if isinstance(proto, (bytes, bytearray)):
        return np.frombuffer(proto, dtype=np.float32)

    # 2. Se proto for TensorChunk (contém o campo 'data' com os bytes)
    if hasattr(proto, 'data'):
        data_val = getattr(proto, 'data', None)
        if isinstance(data_val, (bytes, bytearray)):
            return np.frombuffer(data_val, dtype=np.float32)

    # 3. Se proto for Tensor (tem 'shape' e 'array' -> TensorChunk)
    if safe_has_field(proto, 'array'):
        arr = unpack_tensor(proto.array)
        if arr is not None:
            if hasattr(proto, 'shape') and proto.shape:
                shape = tuple(proto.shape)
                try:
                    return arr.reshape(shape)
                except Exception as e:
                    print(f"[unpack_tensor Warning] Reshape para {shape} falhou: {e}")
            return arr

    # 4. Se proto for TrackData, IntervalData ou VariantData (tem 'values' -> Tensor)
    if safe_has_field(proto, 'values'):
        return unpack_tensor(proto.values)

    # 5. Se proto for ScoreIntervalOutput (tem 'interval_data')
    if safe_has_field(proto, 'interval_data'):
        return unpack_tensor(proto.interval_data)

    # 6. Se proto for ScoreVariantOutput (tem 'variant_data')
    if safe_has_field(proto, 'variant_data'):
        return unpack_tensor(proto.variant_data)

    # 7. Se proto for ModelOutput (tem 'track_data', 'data', 'junction_data')
    if safe_has_field(proto, 'track_data'):
        return unpack_tensor(proto.track_data)
    if safe_has_field(proto, 'data'):
        return unpack_tensor(proto.data)

    return None


def extract_tensor_from_output(output_proto):
    return unpack_tensor(output_proto)

def run_all_tests():
    with grpc.insecure_channel('localhost:50051', options=options) as channel:
        stub = dna_model_service_pb2_grpc.DnaModelServiceStub(channel)
        
        organism = dna_model_pb2.ORGANISM_HOMO_SAPIENS
        
        interval_1mb = dna_model_pb2.Interval(
            chromosome='chr22',
            start=35677410,
            end=36725986
        )
        
        ism_window = dna_model_pb2.Interval(
            chromosome='chr22',
            start=36201690,
            end=36201710
        )

        ontology_uberon = dna_model_pb2.OntologyTerm(
            ontology_type=dna_model_pb2.ONTOLOGY_TYPE_UBERON,
            id=1157
        )

        # --- TESTE 1: GetMetadata ---
        print("\n--- Testando GetMetadata ---")
        try:
            meta_req = dna_model_service_pb2.MetadataRequest()
            for res in stub.GetMetadata(meta_req):
                print("[GetMetadata] Sucesso! Resposta recebida.")
        except grpc.RpcError as e:
            print(f"Erro no GetMetadata: {e.details()}")

        # --- TESTE 2: PredictInterval ---
        print("\n--- Testando PredictInterval ---")
        try:
            req = dna_model_service_pb2.PredictIntervalRequest(
                interval=interval_1mb,
                organism=organism,
                ontology_terms=[ontology_uberon],
                requested_outputs=[dna_model_pb2.OUTPUT_TYPE_RNA_SEQ]
            )
            for i, res in enumerate(stub.PredictInterval(iter([req]))):
                if hasattr(res, 'output'):
                    data = extract_tensor_from_output(res.output)
                    fields = [f[0].name for f in res.output.ListFields()]
                    print(f"[Pacote {i+1}] PredictInterval Shape: {data.shape if data is not None else 'Vazio'} | Campos preenchidos: {fields}")
        except grpc.RpcError as e:
            print(f"Erro no PredictInterval: {e.details()}")

        # --- TESTE 3: ScoreInterval ---
        print("\n--- Testando ScoreInterval ---")
        try:
            req = dna_model_service_pb2.ScoreIntervalRequest(
                interval=interval_1mb,
                organism=organism
            )
            for i, res in enumerate(stub.ScoreInterval(iter([req]))):
                if hasattr(res, 'output'):
                    data = extract_tensor_from_output(res.output)
                    fields = [f[0].name for f in res.output.ListFields()]
                    print(f"[Pacote {i+1}] ScoreInterval Shape: {data.shape if data is not None else 'Vazio'} | Campos preenchidos: {fields}")
        except grpc.RpcError as e:
            print(f"Erro no ScoreInterval: {e.details()}")

        # --- TESTE 4: ScoreIsmVariant ---
        print("\n--- Testando ScoreIsmVariant ---")
        try:
            req = dna_model_service_pb2.ScoreIsmVariantRequest(
                interval=interval_1mb,
                organism=organism,
                ism_interval=ism_window
            )
            for i, res in enumerate(stub.ScoreIsmVariant(iter([req]))):
                if hasattr(res, 'output'):
                    data = extract_tensor_from_output(res.output)
                    fields = [f[0].name for f in res.output.ListFields()]
                    print(f"[Pacote {i+1}] ScoreIsmVariant Shape: {data.shape if data is not None else 'Vazio'} | Campos preenchidos: {fields}")
        except grpc.RpcError as e:
            print(f"Erro no ScoreIsmVariant: {e.details()}")

if __name__ == '__main__':
    run_all_tests()
