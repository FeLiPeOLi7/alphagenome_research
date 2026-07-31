import grpc
import numpy as np

from alphagenome.protos import dna_model_service_pb2_grpc
from alphagenome.protos import dna_model_service_pb2
from alphagenome.protos import dna_model_pb2

class AlphaGenomePrediction:
    """Objeto de resposta contendo os resultados numéricos em NumPy"""
    def __init__(self, reference, alternate):
        self.reference = reference
        self.alternate = alternate
        
    @property
    def diff(self):
        """Calcula o impacto da variante (Alternativa - Referência)"""
        if self.reference is not None and self.alternate is not None:
            return self.alternate - self.reference
        return None

class AlphaGenomeDGX:
    """Cliente SDK para inferência acelerada na DGX local"""
    def __init__(self, host="localhost", port=50051):
        self.target = f"{host}:{port}"
        options = [
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(self.target, options=options)
        self.stub = dna_model_service_pb2_grpc.DnaModelServiceStub(self.channel)

    def _unpack_tensor(self, tensor_proto):
        if not tensor_proto.array.data or not tensor_proto.shape:
            return None
        shape = tuple(tensor_proto.shape)
        arr = np.frombuffer(tensor_proto.array.data, dtype=np.float32)
        return arr.reshape(shape)

    def predict_variant(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        start: int,
        end: int,
        ontology_type: str = "UBERON",
        ontology_id: int = 1157
    ) -> AlphaGenomePrediction:
        """
        Executa a predição da variante diretamente na GPU da DGX.
        """
        enum_name = f"ONTOLOGY_TYPE_{ontology_type.upper()}"
        ontology_enum = getattr(dna_model_pb2, enum_name, dna_model_pb2.ONTOLOGY_TYPE_UBERON)

        term = dna_model_pb2.OntologyTerm(
            ontology_type=ontology_enum,
            id=ontology_id
        )

        request = dna_model_service_pb2.PredictVariantRequest(
            interval=dna_model_pb2.Interval(
                chromosome=chromosome,
                start=start,
                end=end
            ),
            variant=dna_model_pb2.Variant(
                chromosome=chromosome,
                position=position,
                reference_bases=ref,
                alternate_bases=alt
            ),
            ontology_terms=[term]
        )

        responses = self.stub.PredictVariant(iter([request]))
        
        ref_data = None
        alt_data = None

        # reference_output e alternate_output pertencem ao mesmo 'oneof payload' do proto,
        # entao cada pacote carrega apenas um deles. Acumula conforme chegam.
        for response in responses:
            payload = response.WhichOneof('payload')
            if payload == 'reference_output' and ref_data is None:
                ref_data = self._unpack_tensor(response.reference_output.track_data.values)
            elif payload == 'alternate_output' and alt_data is None:
                alt_data = self._unpack_tensor(response.alternate_output.track_data.values)

        return AlphaGenomePrediction(reference=ref_data, alternate=alt_data)
