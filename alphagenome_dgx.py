import grpc
import numpy as np

from alphagenome.models import dna_model
from alphagenome.models import dna_output
from alphagenome.protos import dna_model_service_pb2_grpc
from alphagenome.protos import dna_model_service_pb2
from alphagenome.protos import dna_model_pb2

# Comprimentos de sequência aceitos pelo modelo (16kb / 100kb / 500kb / 1Mb).
SUPPORTED_SEQUENCE_LENGTHS = {2**14, 2**17, 2**19, 2**20}
_VALID_SEQUENCE_CHARACTERS = frozenset('ACGTN')


def _validate_sequence_length(length):
    if length not in SUPPORTED_SEQUENCE_LENGTHS:
        raise ValueError(
            f"Intervalo/sequência com largura {length} não suportado pelo modelo. "
            "Larguras suportadas: 16384, 131072, 524288, 1048576."
        )


class AlphaGenomePrediction:
    """Objeto de resposta do predict_variant contendo os resultados em NumPy."""
    def __init__(self, reference, alternate):
        self.reference = reference
        self.alternate = alternate

    @property
    def diff(self):
        """Calcula o impacto da variante (Alternativa - Referência)."""
        if self.reference is not None and self.alternate is not None:
            return self.alternate - self.reference
        return None


class AlphaGenomeOutput:
    """Objeto de resposta do predict_interval / predict_sequence (uma saída)."""
    def __init__(self, values, output_type=None):
        self.values = values
        self.output_type = output_type

    @property
    def mean(self):
        return float(self.values.mean()) if self.values is not None else None


class AlphaGenomeScore:
    """Objeto de resposta dos métodos de score (intervalo, variante, ISM)."""
    def __init__(self, values, gene_metadata=None):
        self.values = values
        self.gene_metadata = gene_metadata or []

    @property
    def mean(self):
        return float(self.values.mean()) if self.values is not None else None


class AlphaGenomeDGX:
    """Cliente SDK para inferência acelerada na DGX local."""
    def __init__(self, host="localhost", port=50051):
        self.target = f"{host}:{port}"
        options = [
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(self.target, options=options)
        self.stub = dna_model_service_pb2_grpc.DnaModelServiceStub(self.channel)

    # ------------------------------------------------------------------ #
    # Helpers internos
    # ------------------------------------------------------------------ #
    def _unpack_tensor(self, tensor_proto):
        if not tensor_proto or not tensor_proto.array.data or not tensor_proto.shape:
            return None
        shape = tuple(tensor_proto.shape)
        arr = np.frombuffer(tensor_proto.array.data, dtype=np.float32)
        return arr.reshape(shape)

    def _ontology_term(self, ontology_type, ontology_id):
        enum_name = f"ONTOLOGY_TYPE_{ontology_type.upper()}"
        ontology_enum = getattr(
            dna_model_pb2, enum_name, dna_model_pb2.ONTOLOGY_TYPE_UBERON
        )
        return dna_model_pb2.OntologyTerm(
            ontology_type=ontology_enum,
            id=ontology_id
        )

    def _interval(self, chromosome, start, end):
        return dna_model_pb2.Interval(
            chromosome=chromosome, start=start, end=end
        )

    def _requested_outputs(self, requested_outputs):
        # Sem requested_outputs o servidor não computa nenhuma track.
        # Padrão é RNA_SEQ (expressão).
        if requested_outputs is None:
            requested_outputs = [dna_output.OutputType.RNA_SEQ]
        return [getattr(o, 'value', o) for o in requested_outputs]

    def _organism(self, organism):
        if organism is None:
            return dna_model_pb2.Organism.ORGANISM_HOMO_SAPIENS
        if isinstance(organism, dna_model.Organism):
            return organism.value
        return organism

    def _unpack_output_proto(self, output_proto):
        """Extrai (values, output_type) de uma mensagem Output do proto."""
        if output_proto is None:
            return None, None
        payload = output_proto.WhichOneof('payload')
        if payload == 'track_data':
            return (
                self._unpack_tensor(output_proto.track_data.values),
                output_proto.output_type,
            )
        if payload == 'junction_data':
            return (
                self._unpack_tensor(output_proto.junction_data.values),
                output_proto.output_type,
            )
        if payload == 'data':
            return self._unpack_tensor(output_proto.data), output_proto.output_type
        return None, None

    def _unpack_score_proto(self, score_proto):
        """Extrai um AlphaGenomeScore de um ScoreInterval/ScoreVariantOutput."""
        if score_proto is None:
            return None

        valid_fields = set(f.name for f in score_proto.DESCRIPTOR.fields)

        for attr in ['interval_data', 'variant_data', 'track_data']:
            if attr in valid_fields and score_proto.HasField(attr):
                data = getattr(score_proto, attr)
                
                if data == None:
                    break

                values = self._unpack_tensor(data.values)
                genes = []
                for g in data.metadata.gene_metadata:
                    genes.append({
                        'gene_id': g.gene_id,
                        'gene_name': g.name if g.HasField('name') else None,
                        'strand': g.strand,
                        'gene_type': g.type if g.HasField('type') else None,
                        'junction_start': (
                            g.junction_start if g.HasField('junction_start') else None
                        ),
                        'junction_end': (
                            g.junction_end if g.HasField('junction_end') else None
                        ),
                    })
                return AlphaGenomeScore(values=values, gene_metadata=genes)
                
        return None

    def _unpack_output_metadata(self, metadata_proto):
        info = {'output_type': metadata_proto.output_type}
        payload = metadata_proto.WhichOneof('payload')
        if payload == 'tracks':
            info['tracks'] = [
                {
                    'name': t.name,
                    'strand': t.strand,
                    'ontology_type': t.ontology_term.ontology_type,
                    'ontology_id': t.ontology_term.id,
                    'biosample': t.biosample.name,
                    'assay': t.assay if t.HasField('assay') else None,
                    'gtex_tissue': t.gtex_tissue if t.HasField('gtex_tissue') else None,
                }
                for t in metadata_proto.tracks.metadata
            ]
        elif payload == 'junctions':
            info['junctions'] = [
                j.name for j in metadata_proto.junctions.metadata
            ]
        return info

    def _output_type_value(self, output_type):
        if isinstance(output_type, dna_output.OutputType):
            return output_type.value
        if isinstance(output_type, int):
            return output_type
        name = str(output_type).upper()
        if not name.startswith("OUTPUT_TYPE_"):
            name = f"OUTPUT_TYPE_{name}"
        return getattr(dna_model_pb2, name)

    def variant_scorer(self, output_type, kind="GENE_MASK_LFC", width=None):
        """Constrói um dna_model_pb2.VariantScorer simples para passar ao
        score_variant. kind aceita: CENTER_MASK, GENE_MASK_LFC, GENE_MASK_ACTIVE,
        GENE_MASK_SPLICING, PA_QTL, SPLICE_JUNCTION, CONTACT_MAP."""
        ot = self._output_type_value(output_type)
        kind = str(kind).upper()
        if kind == "CENTER_MASK":
            return dna_model_pb2.VariantScorer(
                center_mask=dna_model_pb2.CenterMaskScorer(
                    requested_output=ot,
                    width=width,
                    aggregation_type=dna_model_pb2.AggregationType.AGGREGATION_TYPE_DIFF_LOG2_SUM,
                )
            )
        if kind == "GENE_MASK_LFC":
            return dna_model_pb2.VariantScorer(
                gene_mask=dna_model_pb2.GeneMaskLFCScorer(requested_output=ot)
            )
        if kind == "GENE_MASK_ACTIVE":
            return dna_model_pb2.VariantScorer(
                gene_mask_active=dna_model_pb2.GeneMaskActiveScorer(requested_output=ot)
            )
        if kind == "GENE_MASK_SPLICING":
            return dna_model_pb2.VariantScorer(
                gene_mask_splicing=dna_model_pb2.GeneMaskSplicingScorer(
                    requested_output=ot, width=width
                )
            )
        if kind == "PA_QTL":
            return dna_model_pb2.VariantScorer(pa_qtl=dna_model_pb2.PolyadenylationScorer())
        if kind == "SPLICE_JUNCTION":
            return dna_model_pb2.VariantScorer(splice_junction=dna_model_pb2.SpliceJunctionScorer())
        if kind == "CONTACT_MAP":
            return dna_model_pb2.VariantScorer(contact_map=dna_model_pb2.ContactMapScorer())
        raise ValueError(f"Variant scorer não suportado: {kind}")

    # ------------------------------------------------------------------ #
    # Predição de variante (referência vs. alternativa)
    # ------------------------------------------------------------------ #
    def predict_variant(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        start: int,
        end: int,
        ontology_type: str = "UBERON",
        ontology_id: int = 1157,
        requested_outputs=None,
        organism=None
    ) -> AlphaGenomePrediction:
        """Executa a predição da variante diretamente na GPU da DGX."""
        _validate_sequence_length(end - start)

        request = dna_model_service_pb2.PredictVariantRequest(
            interval=self._interval(chromosome, start, end),
            variant=dna_model_pb2.Variant(
                chromosome=chromosome,
                position=position,
                reference_bases=ref,
                alternate_bases=alt
            ),
            organism=self._organism(organism),
            ontology_terms=[self._ontology_term(ontology_type, ontology_id)],
            requested_outputs=self._requested_outputs(requested_outputs)
        )

        responses = self.stub.PredictVariant(iter([request]))

        ref_data = None
        alt_data = None

        # reference_output e alternate_output pertencem ao mesmo 'oneof payload'
        # do proto, entao cada pacote carrega apenas um deles.
        for response in responses:
            payload = response.WhichOneof('payload')
            if payload == 'reference_output' and ref_data is None:
                ref_data, _ = self._unpack_output_proto(response.reference_output)
            elif payload == 'alternate_output' and alt_data is None:
                alt_data, _ = self._unpack_output_proto(response.alternate_output)

        return AlphaGenomePrediction(reference=ref_data, alternate=alt_data)

    # ------------------------------------------------------------------ #
    # Predição de intervalo
    # ------------------------------------------------------------------ #
    def predict_interval(
        self,
        chromosome: str,
        start: int,
        end: int,
        ontology_type: str = "UBERON",
        ontology_id: int = 1157,
        requested_outputs=None,
        organism=None
    ) -> AlphaGenomeOutput:
        """Gera predições para um intervalo genômico."""
        _validate_sequence_length(end - start)

        request = dna_model_service_pb2.PredictIntervalRequest(
            interval=self._interval(chromosome, start, end),
            organism=self._organism(organism),
            ontology_terms=[self._ontology_term(ontology_type, ontology_id)],
            requested_outputs=self._requested_outputs(requested_outputs)
        )

        responses = self.stub.PredictInterval(iter([request]))

        output = None
        for response in responses:
            if response.WhichOneof('payload') == 'output':
                values, output_type = self._unpack_output_proto(response.output)
                output = AlphaGenomeOutput(values=values, output_type=output_type)
        return output

    # ------------------------------------------------------------------ #
    # Predição de sequência
    # ------------------------------------------------------------------ #
    def predict_sequence(
        self,
        sequence: str,
        requested_outputs=None,
        ontology_type: str = "UBERON",
        ontology_id: int = 1157,
        organism=None
    ) -> AlphaGenomeOutput:
        """Gera predições para uma sequência de DNA crua ('ACGTN')."""
        invalid = set(sequence) - _VALID_SEQUENCE_CHARACTERS
        if invalid:
            raise ValueError(
                f'Sequência inválida. Use apenas caracteres "ACGTN". '
                f'Encontrados: {",".join(sorted(invalid))}'
            )
        _validate_sequence_length(len(sequence))

        ontology_terms = [self._ontology_term(ontology_type, ontology_id)]

        request = dna_model_service_pb2.PredictSequenceRequest(
            sequence=sequence,
            organism=self._organism(organism),
            ontology_terms=ontology_terms,
            requested_outputs=self._requested_outputs(requested_outputs)
        )

        responses = self.stub.PredictSequence(iter([request]))

        output = None
        for response in responses:
            if response.WhichOneof('payload') == 'output':
                values, output_type = self._unpack_output_proto(response.output)
                output = AlphaGenomeOutput(values=values, output_type=output_type)
        return output

    # ------------------------------------------------------------------ #
    # Score de intervalo
    # ------------------------------------------------------------------ #
    def score_interval(
        self,
        chromosome: str,
        start: int,
        end: int,
        interval_scorers=None,
        organism=None
    ):
        """Gera scores para um intervalo. interval_scorers pode ser uma lista de
        mensagens dna_model_pb2.IntervalScorer; se vazio o servidor usa os padrões."""
        _validate_sequence_length(end - start)

        request = dna_model_service_pb2.ScoreIntervalRequest(
            interval=self._interval(chromosome, start, end),
            organism=self._organism(organism),
            interval_scorers=interval_scorers or [],
            merge_stranded_gene_tracks=True
        )

        responses = self.stub.ScoreInterval(iter([request]))

        scores = []
        for response in responses:
            if response.WhichOneof('payload') == 'output':
                scores.append(self._unpack_score_proto(response.output))
        return scores

    # ------------------------------------------------------------------ #
    # Score de variante
    # ------------------------------------------------------------------ #
    def score_variant(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        start: int,
        end: int,
        variant_scorers=None,
        organism=None
    ):
        """Gera scores de impacto para uma variante. variant_scorers pode ser uma
        lista de mensagens dna_model_pb2.VariantScorer; se vazio o servidor usa os
        scorers recomendados."""
        _validate_sequence_length(end - start)

        request = dna_model_service_pb2.ScoreVariantRequest(
            interval=self._interval(chromosome, start, end),
            variant=dna_model_pb2.Variant(
                chromosome=chromosome,
                position=position,
                reference_bases=ref,
                alternate_bases=alt
            ),
            organism=self._organism(organism),
            variant_scorers=variant_scorers or [],
            merge_stranded_gene_tracks=True
        )

        responses = self.stub.ScoreVariant(iter([request]))

        scores = []
        for response in responses:
            if response.WhichOneof('payload') == 'output':
                scores.append(self._unpack_score_proto(response.output))
        return scores

    # ------------------------------------------------------------------ #
    # Score ISM (in-silico mutagenesis)
    # ------------------------------------------------------------------ #
    def score_ism_variant(
        self,
        chromosome: str,
        start: int,
        end: int,
        ism_start: int,
        ism_end: int,
        variant_scorers=None,
        interval_variant=None,
        organism=None
    ):
        """Gera scores ISM para um intervalo. interval_variant é opcional e deve
        ser uma mensagem dna_model_pb2.Variant aplicada ao intervalo de referência."""
        _validate_sequence_length(end - start)

        request = dna_model_service_pb2.ScoreIsmVariantRequest(
            interval=self._interval(chromosome, start, end),
            ism_interval=self._interval(chromosome, ism_start, ism_end),
            organism=self._organism(organism),
            variant_scorers=variant_scorers or [],
            merge_stranded_gene_tracks=True
        )
        if interval_variant is not None:
            request.interval_variant.CopyFrom(interval_variant)

        responses = self.stub.ScoreIsmVariant(iter([request]))

        scores = []
        for response in responses:
            if response.WhichOneof('payload') == 'output':
                scores.append(self._unpack_score_proto(response.output))
        return scores

    # ------------------------------------------------------------------ #
    # Metadados
    # ------------------------------------------------------------------ #
    def get_metadata(self, organism=None):
        """Retorna metadados (tracks) do organismo consultado."""
        request = dna_model_service_pb2.MetadataRequest(
            organism=self._organism(organism)
        )

        responses = self.stub.GetMetadata(request)

        output_metadata = []
        for response in responses:
            for metadata_proto in response.output_metadata:
                output_metadata.append(self._unpack_output_metadata(metadata_proto))
        return output_metadata
