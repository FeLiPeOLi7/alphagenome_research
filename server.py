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
from alphagenome.data.genome import Interval

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
    
    if interval_proto is not None:
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

    def _parse_ontology_terms(self, terms_proto):
        ontology_list = []
        for term in terms_proto:
            prefix = dna_model_pb2.OntologyType.Name(term.ontology_type).replace("ONTOLOGY_TYPE_", "")
            ontology_list.append(f"{prefix}:{term.id:07d}")
        return ontology_list

    def _parse_requested_outputs(self, requested_outputs):
        parsed = []

        for o in requested_outputs:
            # Pega o nome do enum do Protobuf (ex: 'OUTPUT_TYPE_RNA_SEQ')
            name = dna_model_pb2.OutputType.Name(o)
        
            # Remove o prefixo 'OUTPUT_TYPE_' para bater com o Enum Python ('RNA_SEQ')
            clean_name = name.replace("OUTPUT_TYPE_", "")
        
            if hasattr(dna_model.OutputType, clean_name):
                parsed.append(getattr(dna_model.OutputType, clean_name))
            elif clean_name in dna_model.OutputType.__members__:
                parsed.append(dna_model.OutputType[clean_name])
            elif hasattr(dna_model.OutputType, name):
                parsed.append(getattr(dna_model.OutputType, name))
            
        return parsed 
        
    def _parse_organism(self, organism_proto):
        if not organism_proto:
            return dna_model.Organism.HOMO_SAPIENS  # Padrão é humano
         
        # Extrai 'HOMO_SAPIENS' de 'ORGANISM_HOMO_SAPIENS'
        enum_name = dna_model_pb2.Organism.Name(organism_proto).replace("ORGANISM_", "")
        return getattr(dna_model.Organism, enum_name, dna_model.Organism.HOMO_SAPIENS)

    def _parse_variant_scorers(self, scorers_proto):
        if not scorers_proto:
            return None  # Se omitido, o AlphaGenome usa os scorers padrão
        
        scorers = []
        for scorer_enum in scorers_proto:
            scorer_name = dna_model_pb2.VariantScorer.Name(scorer_enum).replace("VARIANT_SCORER_", "")
            scorers.append(scorer_name)

        return scorers if scorers else None

    def _parse_interval_scorers(self, scorers_proto):
        if not scorers_proto:
            return None
        scorers = []
        for scorer_enum in scorers_proto:
            scorer_name = dna_model_pb2.IntervalScorer.Name(scorer_enum).replace("INTERVAL_SCORER_", "")
            scorers.append(scorer_name)
        return scorers if scorers else None

    def _parse_interval(self, interval_proto):
        """Converte a mensagem Protobuf Interval para o objeto Interval do SDK do AlphaGenome."""
        return Interval(
            chromosome=interval_proto.chromosome,
            start=interval_proto.start,
            end=interval_proto.end
        )

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
            ontology_list = self._parse_ontology_terms(request.ontology_terms)

            # Define as saídas requisitadas
            requested_outputs = self._parse_requested_outputs(request.requested_outputs)
            
            # Define o organismo
            organism = self._parse_organism(request.organism)

            print(f"Executando variante {variant.chromosome}:{variant.position} ({variant.reference_bases}->{variant.alternate_bases}) na GPU...")

            outputs = self.model.predict_variant(
                interval=interval,
                organism=organism,
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

    def PredictInterval(self, request_iterator, context):
        print("\n[gRPC] Nova requisição PredictInterval recebida...")

        for request in request_iterator:
            interval = genome.Interval(
                chromosome=request.interval.chromosome,
                start=request.interval.start,
                end=request.interval.end
            )
            
            # Processa Ontologias
            ontology_list = self._parse_ontology_terms(request.ontology_terms)

            # Define as saídas requisitadas
            requested_outputs = self._parse_requested_outputs(request.requested_outputs)

            # Define o organismo
            organism = self._parse_organism(request.organism)


            output = self.model.predict_interval(
                interval=interval,
                organism=organism,
                ontology_terms=ontology_list,
                requested_outputs=requested_outputs,
            )

            response = dna_model_service_pb2.PredictIntervalResponse()

            populate_output_proto(response.output, output, request.interval)
            yield response

    def PredictSequence(self, request_iterator, context):
        print("\n[gRPC] Nova requisição PredictSequence recebida...")

        for request in request_iterator:
            seq_str = request.sequence
            
            # Processa Ontologias
            ontology_list = self._parse_ontology_terms(request.ontology_terms)

            # Define as saídas requisitadas
            requested_outputs = self._parse_requested_outputs(request.requested_outputs)

            # Define o organismo
            organism = self._parse_organism(request.organism)


            output = self.model.predict_sequence(
                sequence=seq_str,
                organism=organism,
                ontology_terms=ontology_list,
                requested_outputs=requested_outputs,
            )

            response = dna_model_service_pb2.PredictSequenceResponse()

            populate_output_proto(response.output, output, interval_proto=None)
            yield response

    def ScoreVariant(self, request_iterator, context):
        print("\n[gRPC] Nova requisição ScoreVariant recebida...")

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

            kwargs = {
                "interval": interval,
                "variant": variant,
                "organism": self._parse_organism(request.organism),
            }

            variant_scorers = self._parse_variant_scorers(request.variant_scorers)
            if variant_scorers:
                kwargs["variant_scorers"] = variant_scorers

            scores = self.model.score_variant(**kwargs)

            # O modelo pode retornar um único resultado ou uma lista (chunks)
            if not isinstance(scores, list):
                scores = [scores]

            for item in scores:
                response = dna_model_service_pb2.ScoreVariantResponse()
                
                item_to_proto = getattr(item, 'to_proto', None)
                if callable(item_to_proto):
                    response.CopyFrom(item_to_proto())
                else:
                    # Usamos a mesma função de popular que já varre os tipos (rna_seq, etc)
                    populate_output_proto(response.output, item, request.interval)
                
                yield response

    def ScoreInterval(self, request_iterator, context):
        print("\n[gRPC] Nova requisição ScoreInterval recebida...")

        try:
            for request in request_iterator:
                organism = self._parse_organism(request.organism)
                interval = self._parse_interval(request.interval)

                scores = self.model.score_interval(
                    interval=interval,
                    organism=organism
                )

                if not isinstance(scores, list):
                    scores = [scores]

                for item in scores:
                    response = dna_model_service_pb2.ScoreIntervalResponse()
                    item_to_proto = getattr(item, 'to_proto', None)
                    if callable(item_to_proto):
                        response.CopyFrom(item_to_proto())
                    else:
                        populate_output_proto(response.output, item, request.interval)
                    yield response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise e

    def ScoreIsmVariant(self, request_iterator, context):
        print("\n[gRPC] Nova requisição ScoreIsmVariant recebida...")

        try:
            for request in request_iterator:
                organism = self._parse_organism(request.organism)
                interval = self._parse_interval(request.interval)
                ism_interval = self._parse_interval(request.ism_interval)

                scores = self.model.score_ism_variants(
                    interval=interval,
                    organism=organism,
                    ism_interval=ism_interval
                )

                if not isinstance(scores, list):
                    scores = [scores]

                for item in scores:
                    response = dna_model_service_pb2.ScoreIsmVariantResponse()
                    item_to_proto = getattr(item, 'to_proto', None)
                    if callable(item_to_proto):
                        response.CopyFrom(item_to_proto())
                    else:
                        populate_output_proto(response.output, item, request.interval)
                    yield response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise e    

    def GetMetadata(self, request, context):
        print("\n[gRPC] Nova requisição GetMetadata recebida...")
        response = dna_model_service_pb2.MetadataResponse()
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
