from concurrent import futures
import os
import grpc
import numpy as np
import anndata

# Configurações de ambiente
os.environ["HF_HUB_DISABLE_XET"] = "1"
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MB

from alphagenome import tensor_utils
from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.models import dna_output
from alphagenome.models import interval_scorers as interval_scorers_lib
from alphagenome.models import track_data_utils
from alphagenome.models import variant_scorers as variant_scorers_lib
from alphagenome_research.model import dna_model
from alphagenome.protos import dna_model_service_pb2_grpc
from alphagenome.protos import dna_model_service_pb2
from alphagenome.protos import dna_model_pb2
from alphagenome.protos import tensor_pb2
from alphagenome.data.genome import Interval

def extract_numpy_array(obj):
    """Extrai uma matriz NumPy (float32) de objetos AlphaGenome, AnnData, DataFrames, escopos de score ou arrays."""
    if obj is None:
        return None

    if isinstance(obj, np.ndarray):
        return obj

    # AnnData ou objeto de score com atributo .X (matriz principal)
    if hasattr(obj, 'X') and getattr(obj, 'X') is not None:
        val = getattr(obj, 'X')
        if hasattr(val, 'toarray'):  # scipy.sparse matrix
            return val.toarray()
        return np.asarray(val)

    # Objeto com atributo .values (Track, DataFrame, Series, etc.)
    if hasattr(obj, 'values') and getattr(obj, 'values') is not None:
        val = getattr(obj, 'values')
        if isinstance(val, np.ndarray):
            return val
        if hasattr(val, 'toarray'):
            return val.toarray()
        return np.asarray(val)

    # Objeto com atributo .scores, .array ou .data
    for attr in ['scores', 'array', 'data']:
        if hasattr(obj, attr) and getattr(obj, attr) is not None:
            val = getattr(obj, attr)
            if not callable(val):
                if isinstance(val, np.ndarray):
                    return val
                if hasattr(val, 'toarray'):
                    return val.toarray()
                try:
                    return np.asarray(val)
                except Exception:
                    pass

    # Dicionário contendo dados numéricos
    if isinstance(obj, dict):
        for k in ['X', 'values', 'scores', 'data', 'array']:
            if k in obj and obj[k] is not None:
                res = extract_numpy_array(obj[k])
                if res is not None:
                    return res

    # Tentativa genérica para iteráveis/array-like (evitando strings ou protobufs)
    if not isinstance(obj, (str, bytes)) and not hasattr(obj, 'DESCRIPTOR'):
        try:
            return np.asarray(obj)
        except Exception:
            pass

    return None


def fill_tensor_chunk_proto(tensor_chunk_proto, arr):
    """Converte a matriz NumPy para a estrutura Protobuf TensorChunk (shape e bytes)."""
    if arr is None or tensor_chunk_proto is None:
        return
    arr_np = np.asarray(arr, dtype=np.float32)
    tensor_chunk_proto.shape.clear()
    tensor_chunk_proto.shape.extend(arr_np.shape)
    tensor_chunk_proto.data_type = tensor_pb2.DataType.DATA_TYPE_FLOAT32
    tensor_chunk_proto.array.data = arr_np.tobytes()


def fill_data_proto(data_proto, arr, interval_proto=None):
    """
    Preenche uma estrutura Protobuf (TrackData, IntervalData ou VariantData)
    com dados de um array NumPy e o intervalo genômico.
    """
    if arr is None or data_proto is None:
        return

    try:
        arr_np = np.asarray(arr, dtype=np.float32)
    except (TypeError, ValueError):
        print(f"[gRPC] Aviso: dados não numéricos ignorados ({type(arr).__name__})")
        return

    # Preenche os valores no campo 'values' ou 'array'
    if hasattr(data_proto, 'values'):
        fill_tensor_chunk_proto(data_proto.values, arr_np)
    elif hasattr(data_proto, 'array'):
        fill_tensor_chunk_proto(data_proto.array, arr_np)
    else:
        fill_tensor_chunk_proto(data_proto, arr_np)

    # Copia a Resolução (se disponível no objeto original)
    if hasattr(arr, 'resolution') and hasattr(data_proto, 'resolution'):
        try:
            data_proto.resolution = int(arr.resolution)
        except (ValueError, TypeError):
            pass

    # Copia o Intervalo Genômico
    if interval_proto is not None and hasattr(data_proto, 'interval'):
        data_proto.interval.chromosome = interval_proto.chromosome
        data_proto.interval.start = interval_proto.start
        data_proto.interval.end = interval_proto.end


def fill_track_data_proto(track_data_proto, track_obj, interval_proto=None):
    """Converte a matriz do AlphaGenome para a estrutura Protobuf TrackData (compatibilidade)."""
    if track_obj is None or track_data_proto is None:
        return

    arr = extract_numpy_array(track_obj)
    if arr is not None:
        fill_data_proto(track_data_proto, arr, interval_proto)


def populate_output_proto(output_proto, model_output, interval_proto=None, output_category=None):
    """Varre todas as faixas (tracks) e/ou objetos AnnData/Score e preenche a resposta Protobuf."""
    if model_output is None or output_proto is None:
        return

    # 1. Mapeamento dos tipos de saída suportados para tracks de predição direta
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

    has_track_attr = False
    for attr_name, output_enum in track_mapping:
        if hasattr(model_output, attr_name):
            track_obj = getattr(model_output, attr_name)
            if track_obj is not None:
                has_track_attr = True
                output_proto.output_type = output_enum
                if isinstance(track_obj, track_data.TrackData):
                    try:
                        track_data_proto, _ = track_data_utils.to_protos(track_obj)
                        output_proto.track_data.CopyFrom(track_data_proto)
                    except Exception as e:
                        print(f"[gRPC] Aviso: falha ao serializar TrackData ({e}); usando fallback manual")
                        fill_track_data_proto(output_proto.track_data, track_obj, interval_proto)
                else:
                    fill_track_data_proto(output_proto.track_data, track_obj, interval_proto)
                break

    if has_track_attr:
        return

    # 2. Caso seja um objeto AnnData ou resultado de Scoring (ScoreInterval / ScoreIsmVariant / ScoreVariant)
    arr = extract_numpy_array(model_output)
    if arr is not None:
        valid_fields = set(f.name for f in output_proto.DESCRIPTOR.fields)

        if output_category == 'variant' and 'variant_data' in valid_fields:
            fill_data_proto(output_proto.variant_data, arr, interval_proto)
        elif output_category == 'interval' and 'interval_data' in valid_fields:
            fill_data_proto(output_proto.interval_data, arr, interval_proto)
        elif 'interval_data' in valid_fields and output_category == 'interval':
            fill_data_proto(output_proto.interval_data, arr, interval_proto)
        elif 'variant_data' in valid_fields and output_category == 'variant':
            fill_data_proto(output_proto.variant_data, arr, interval_proto)
        elif 'interval_data' in valid_fields:
            fill_data_proto(output_proto.interval_data, arr, interval_proto)
        elif 'variant_data' in valid_fields:
            fill_data_proto(output_proto.variant_data, arr, interval_proto)
        elif 'track_data' in valid_fields:
            fill_data_proto(output_proto.track_data, arr, interval_proto)


def flatten_scores(scores):
    """Achata recursivamente listas/tuplas aninhadas de objetos de score ou AnnData."""
    flat = []
    if isinstance(scores, (list, tuple)):
        for element in scores:
            flat.extend(flatten_scores(element))
    elif scores is not None:
        flat.append(scores)
    return flat


def score_anndata_to_variant_response(score, variant=None):
    """Serializa um AnnData de score de variante para o proto ScoreVariantResponse.

    Segue o mesmo formato usado pelo cliente oficial (dna_client_test.py
    _generate_variant_scoring_protos): values = X (com quantiles se houver),
    variant = uns['variant'], track_metadata = var, gene_metadata = obs.
    """
    response = dna_model_service_pb2.ScoreVariantResponse()

    track_metadata_protos = [
        dna_model_pb2.TrackMetadata(
            name=name,
            strand=genome.Strand.from_str(strand).to_proto(),
        )
        for name, strand in score.var[['name', 'strand']].values
    ]

    gene_metadata_protos = []
    if score.obs is not None and 'gene_id' in score.obs:
        for _, row in score.obs.iterrows():
            strand_proto = None
            if (strand := row.get('strand')) is not None:
                strand_proto = genome.Strand.from_str(strand).to_proto()
            gene_metadata_protos.append(
                dna_model_pb2.GeneScorerMetadata(
                    gene_id=row['gene_id'],
                    strand=strand_proto,
                    name=row.get('gene_name'),
                    type=row.get('gene_type'),
                    junction_start=row.get('junction_Start'),
                    junction_end=row.get('junction_End'),
                )
            )

    if 'quantiles' in score.layers:
        score_tensor = np.stack([score.X, score.layers['quantiles']])
    else:
        score_tensor = score.X[np.newaxis]
    tensor_proto, chunks = tensor_utils.pack_tensor(score_tensor, bytes_per_chunk=0)
    if chunks:
        raise ValueError("bytes_per_chunk=0 deve gerar 0 chunks")

    variant_proto = None
    uns_variant = score.uns.get('variant') if score.uns else None
    if uns_variant is None:
        uns_variant = variant
    if uns_variant is not None:
        variant_proto = uns_variant.to_proto()

    response.output.variant_data.values.CopyFrom(tensor_proto)
    if variant_proto is not None:
        response.output.variant_data.metadata.variant.CopyFrom(variant_proto)
    response.output.variant_data.metadata.track_metadata.extend(track_metadata_protos)
    response.output.variant_data.metadata.gene_metadata.extend(gene_metadata_protos)
    return response


class AlphaGenomeServer(dna_model_service_pb2_grpc.DnaModelServiceServicer):
    def __init__(self, model=None):
        print("[DGX] Carregando o modelo AlphaGenome na GPU ('all_folds')...", flush=True)
        if model is None:
            self.model = dna_model.create_from_huggingface('all_folds')
        else:
            self.model = model
        print("[DGX] Modelo carregado com sucesso na VRAM!", flush=True)

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

        # Sem requested_outputs o modelo roda com conjunto vazio (todas as tracks None),
        # o que fazia o populate_output_proto tentar converter o próprio 'Output' em float.
        # Usa RNA_SEQ como padrão para o fluxo sempre retornar dados de expressão.
        if not parsed:
            parsed = [dna_model.OutputType.RNA_SEQ]

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
        for scorer_proto in scorers_proto:
            which = scorer_proto.WhichOneof('scorer')
            if which == 'center_mask':
                cm = scorer_proto.center_mask
                scorers.append(variant_scorers_lib.CenterMaskScorer(
                    requested_output=dna_output.OutputType(cm.requested_output),
                    width=cm.width if cm.HasField('width') else None,
                    aggregation_type=variant_scorers_lib.AggregationType(cm.aggregation_type),
                ))
            elif which == 'gene_mask':
                scorers.append(variant_scorers_lib.GeneMaskLFCScorer(
                    requested_output=dna_output.OutputType(scorer_proto.gene_mask.requested_output),
                ))
            elif which == 'gene_mask_active':
                scorers.append(variant_scorers_lib.GeneMaskActiveScorer(
                    requested_output=dna_output.OutputType(scorer_proto.gene_mask_active.requested_output),
                ))
            elif which == 'gene_mask_splicing':
                gms = scorer_proto.gene_mask_splicing
                scorers.append(variant_scorers_lib.GeneMaskSplicingScorer(
                    requested_output=dna_output.OutputType(gms.requested_output),
                    width=gms.width if gms.HasField('width') else None,
                ))
            elif which == 'pa_qtl':
                scorers.append(variant_scorers_lib.PolyadenylationScorer())
            elif which == 'splice_junction':
                scorers.append(variant_scorers_lib.SpliceJunctionScorer())
            elif which == 'contact_map':
                scorers.append(variant_scorers_lib.ContactMapScorer())
            else:
                raise ValueError(f"Variant scorer não suportado: {which}")

        return scorers if scorers else None

    def _parse_interval_scorers(self, scorers_proto):
        if not scorers_proto:
            return None

        scorers = []
        for scorer_proto in scorers_proto:
            which = scorer_proto.WhichOneof('scorer')
            if which == 'gene_mask':
                gm = scorer_proto.gene_mask
                scorers.append(interval_scorers_lib.GeneMaskScorer(
                    requested_output=dna_output.OutputType(gm.requested_output),
                    width=gm.width if gm.HasField('width') else None,
                    aggregation_type=interval_scorers_lib.IntervalAggregationType(gm.aggregation_type),
                ))
            else:
                raise ValueError(f"Interval scorer não suportado: {which}")

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

            # 1. Resgate blindado (suporta tanto se 'outputs' for uma classe quanto um dicionário)
            ref_data = getattr(outputs, 'reference', None)
            if ref_data is None and isinstance(outputs, dict):
                ref_data = outputs.get('reference')
                
            alt_data = getattr(outputs, 'alternate', None)
            if alt_data is None and isinstance(outputs, dict):
                alt_data = outputs.get('alternate')

            # 2. Preenche Referência dinamicamente (is not None resolve a armadilha do truthiness)
            # NOTA: reference_output e alternate_output pertencem ao mesmo 'oneof payload' no proto,
            # entao precisam ser enviados em respostas separadas (o segundo apagaria o primeiro).
            if ref_data is not None:
                try:
                    ref_response = dna_model_service_pb2.PredictVariantResponse()
                    populate_output_proto(ref_response.reference_output, ref_data, interval, output_category="variant")
                    yield ref_response
                except Exception as e:
                    print(f"\n[gRPC ERRO FATAL] Falha interna no populate_output_proto (Referência): {e}")

            # 3. Preenche Alternativa dinamicamente
            if alt_data is not None:
                try:
                    alt_response = dna_model_service_pb2.PredictVariantResponse()
                    populate_output_proto(alt_response.alternate_output, alt_data, interval, output_category="variant")
                    yield alt_response
                except Exception as e:
                    print(f"\n[gRPC ERRO FATAL] Falha interna no populate_output_proto (Alternativa): {e}")

            print("[gRPC] Transmitindo matrizes de inferência para o cliente...")

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
            scores = flatten_scores(scores)

            for item in scores:
                response = dna_model_service_pb2.ScoreVariantResponse()
                
                if isinstance(item, anndata.AnnData):
                    response = score_anndata_to_variant_response(item, variant)
                    yield response
                    continue

                item_to_proto = getattr(item, 'to_proto', None)
                proto_obj = None
                if callable(item_to_proto):
                    try:
                        proto_obj = item_to_proto()
                    except Exception:
                        pass

                if proto_obj is not None and hasattr(proto_obj, 'DESCRIPTOR'):
                    response.CopyFrom(proto_obj)
                else:
                    populate_output_proto(response.output, item, request.interval, output_category='variant')
                
                yield response

    def ScoreInterval(self, request_iterator, context):
        print("\n[gRPC] Nova requisição ScoreInterval recebida...")

        try:
            for request in request_iterator:
                organism = self._parse_organism(request.organism)
                interval = self._parse_interval(request.interval)

                kwargs = {
                    "interval": interval,
                    "organism": organism,
                }
                interval_scorers = self._parse_interval_scorers(request.interval_scorers)
                if interval_scorers:
                    kwargs["interval_scorers"] = interval_scorers

                scores = self.model.score_interval(**kwargs)
                scores = flatten_scores(scores)

                for item in scores:
                    response = dna_model_service_pb2.ScoreIntervalResponse()
                    item_to_proto = getattr(item, 'to_proto', None)
                    proto_obj = None
                    if callable(item_to_proto):
                        try:
                            proto_obj = item_to_proto()
                        except Exception:
                            pass

                    if proto_obj is not None and hasattr(proto_obj, 'DESCRIPTOR'):
                        response.CopyFrom(proto_obj)
                    else:
                        populate_output_proto(response.output, item, request.interval, output_category='interval')
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

                kwargs = {
                    "interval": interval,
                    "organism": organism,
                    "ism_interval": ism_interval,
                }
                if request.HasField('interval_variant'):
                    iv = request.interval_variant
                    kwargs["interval_variant"] = genome.Variant(
                        chromosome=iv.chromosome,
                        position=iv.position,
                        reference_bases=iv.reference_bases,
                        alternate_bases=iv.alternate_bases,
                    )
                variant_scorers = self._parse_variant_scorers(request.variant_scorers)
                if variant_scorers:
                    kwargs["variant_scorers"] = variant_scorers

                scores = self.model.score_ism_variants(**kwargs)
                scores = flatten_scores(scores)

                for item in scores:
                    response = dna_model_service_pb2.ScoreIsmVariantResponse()
                    item_to_proto = getattr(item, 'to_proto', None)
                    proto_obj = None
                    if callable(item_to_proto):
                        try:
                            proto_obj = item_to_proto()
                        except Exception:
                            pass

                    if proto_obj is not None and hasattr(proto_obj, 'DESCRIPTOR'):
                        response.CopyFrom(proto_obj)
                    else:
                        populate_output_proto(response.output, item, request.interval, output_category='variant')
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
    cert_path = os.environ.get("ALPHAGENOME_TLS_CERT", "certs/server.crt")
    key_path = os.environ.get("ALPHAGENOME_TLS_KEY", "certs/server.key")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        with open(key_path, "rb") as f:
            private_key = f.read()
        with open(cert_path, "rb") as f:
            certificate_chain = f.read()
        credentials = grpc.ssl_server_credentials(
            ((private_key, certificate_chain),)
        )
        server.add_secure_port(f"0.0.0.0:{port}", credentials)
        print(f"\nServidor AlphaGenome DGX ativo e escutando em 0.0.0.0:{port} (TLS)")
    else:
        server.add_insecure_port(f"0.0.0.0:{port}")
        print(f"\nServidor AlphaGenome DGX ativo e escutando em 0.0.0.0:{port} (sem TLS)")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
