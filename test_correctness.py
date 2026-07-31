import os
# Desativa a pre-alocacao total de VRAM pelo JAX antes das importacoes
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
import time
import unittest
from concurrent import futures
import numpy as np
import grpc
from absl import flags

# Evita conflitos de flags entre o absl e o pytest/unittest
try:
    flags.FLAGS(sys.argv[:1], known_only=True)
except Exception:
    pass

from alphagenome.data import genome
from alphagenome_research.model import dna_model
from alphagenome.protos import dna_model_service_pb2_grpc
from alphagenome.protos import dna_model_service_pb2
from alphagenome.protos import dna_model_pb2

from server import AlphaGenomeServer

MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
OPTIONS = [
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
]


def safe_has_field(proto, field_name):
    """Verifica com seguranca se um campo Protobuf esta presente."""
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
    """Desempacota recursivamente a mensagem gRPC para matriz NumPy."""
    if proto is None:
        return None

    # 1. Caso base: bytes puros
    if isinstance(proto, (bytes, bytearray)):
        if len(proto) == 0: 
            return None
        return np.frombuffer(proto, dtype=np.float32)

    def _apply_shape(arr, p):
        if arr is not None and hasattr(p, 'shape') and p.shape:
            try:
                return arr.reshape(tuple(p.shape))
            except Exception:
                pass
        return arr

    # Fast Path
    if hasattr(proto, 'data'):
        data_val = getattr(proto, 'data', None)
        if isinstance(data_val, (bytes, bytearray)) and len(data_val) > 0:
            return _apply_shape(np.frombuffer(data_val, dtype=np.float32), proto)

    # Mapas Protobuf (Resolve o PredictVariant)
    if hasattr(proto, 'values') and callable(getattr(proto, 'values')):
        for val in proto.values():
            res = unpack_tensor(val)
            if res is not None: 
                return _apply_shape(res, proto)

    # Listas (Repeated Fields)
    if hasattr(proto, '__iter__') and not isinstance(proto, (str, bytes, bytearray, dict)):
        for item in proto:
            res = unpack_tensor(item)
            if res is not None: 
                return _apply_shape(res, proto)

    # Busca limpa em Sub-mensagens 
    if hasattr(proto, 'ListFields'):
        for desc, val in proto.ListFields():
            # Ignoramos metadados irrelevantes pra ir mais rápido
            if desc.name in ['shape', 'organism', 'ontology_terms', 'chromosome', 'start', 'end', 'position']:
                continue
            
            res = unpack_tensor(val)
            if res is not None: 
                return _apply_shape(res, proto)

    return None


def extract_numpy_from_sdk_output(output_obj):
    """Extrai matrizes NumPy de objetos SDK locais (TrackData, AnnData, DataFrames, etc.)."""
    if output_obj is None:
        return None

    if isinstance(output_obj, np.ndarray):
        return output_obj

    if hasattr(output_obj, 'X') and getattr(output_obj, 'X') is not None:
        val = getattr(output_obj, 'X')
        if hasattr(val, 'toarray'):
            return val.toarray()
        return np.asarray(val)

    if hasattr(output_obj, 'values') and getattr(output_obj, 'values') is not None:
        val = getattr(output_obj, 'values')
        if isinstance(val, np.ndarray):
            return val
        if hasattr(val, 'toarray'):
            return val.toarray()
        return np.asarray(val)

    for attr in ['rna_seq', 'cage', 'atac', 'dnase', 'chip_tf', 'chip_histone']:
        if hasattr(output_obj, attr):
            sub_obj = getattr(output_obj, attr)
            if sub_obj is not None:
                return extract_numpy_from_sdk_output(sub_obj)

    return None


class TestAlphaGenomeParity(unittest.TestCase):
    """
    Suite de Testes Unitarios de Paridade Numerica e Desempenho (Benchmarking).
    Compara tempo de execucao e resultados entre chamadas diretas da SDK e o servidor gRPC.
    """
    
    metrics = {}

    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("[INICIO] Suite de Validacao de Paridade e Benchmark")
        print("=======================================================")

        t0 = time.perf_counter()
        print("[SETUP] Carregando modelo AlphaGenome local...")
        cls.local_model = dna_model.create_from_huggingface('all_folds')
        load_time = time.perf_counter() - t0
        print(f"[SETUP] Modelo carregado com sucesso em {load_time:.2f}s\n")

        print("[SETUP] Iniciando servidor gRPC local (In-Process)...")
        cls.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4),
            options=OPTIONS
        )
        servicer = AlphaGenomeServer(cls.local_model)
        dna_model_service_pb2_grpc.add_DnaModelServiceServicer_to_server(servicer, cls.grpc_server)
        
        cls.port = cls.grpc_server.add_insecure_port("127.0.0.1:50051")
        cls.grpc_server.start()
        print(f"[SETUP] Servidor gRPC local ativo na porta {cls.port}\n")

        cls.channel = grpc.insecure_channel(f"127.0.0.1:{cls.port}", options=OPTIONS)
        cls.stub = dna_model_service_pb2_grpc.DnaModelServiceStub(cls.channel)

        cls.interval = genome.Interval(chromosome='chr22', start=35677410, end=36725986)
        cls.interval_proto = dna_model_pb2.Interval(chromosome='chr22', start=35677410, end=36725986)

        print("[SETUP] Executando warm-up na SDK (Compilacao JAX)... Isso levara alguns minutos.")
        t0_warmup = time.perf_counter()
        
        cls.local_model.predict_interval(
            interval=cls.interval,
            organism=dna_model.Organism.HOMO_SAPIENS,
            ontology_terms=['UBERON:0001157'],
            requested_outputs=[dna_model.OutputType.RNA_SEQ]
        )
        
        t_warmup = time.perf_counter() - t0_warmup
        print(f"[SETUP] Warm-up concluido em {t_warmup:.2f}s! O modelo agora esta 'quente'.\n")

        cls.variant = genome.Variant(
            chromosome='chr22',
            position=36201698,
            reference_bases='A',
            alternate_bases='C'
        )
        cls.variant_proto = dna_model_pb2.Variant(
            chromosome='chr22',
            position=36201698,
            reference_bases='A',
            alternate_bases='C'
        )

        cls.ontology_uberon = dna_model_pb2.OntologyTerm(
            ontology_type=dna_model_pb2.ONTOLOGY_TYPE_UBERON,
            id=1157
        )
        cls.organism = dna_model_pb2.ORGANISM_HOMO_SAPIENS

    @classmethod
    def tearDownClass(cls):
        print("\n[TEARDOWN] Finalizando conexoes e desligando servidor gRPC...")
        cls.channel.close()
        cls.grpc_server.stop(grace=0)
        
        print("\n=======================================================")
        print("     RELATORIO COMPARATIVO DE TEMPO (BENCHMARK)        ")
        print("=======================================================")
        print(f"{'Endpoint':<22} | {'SDK Local':<12} | {'gRPC/Cloud':<12} | {'Overhead':<10}")
        print("-" * 65)
        
        for endpoint, data in cls.metrics.items():
            t_sdk = data.get('sdk', 0.0)
            t_grpc = data.get('grpc', 0.0)
            
            if t_sdk > 0:
                diff_pct = ((t_grpc - t_sdk) / t_sdk) * 100
                diff_str = f"{diff_pct:+.1f}%"
            else:
                diff_str = "N/A"
                
            sdk_str = f"{t_sdk:.4f}s" if t_sdk > 0 else "N/A"
            grpc_str = f"{t_grpc:.4f}s"
            
            print(f"{endpoint:<22} | {sdk_str:<12} | {grpc_str:<12} | {diff_str:<10}")
            
        print("=======================================================\n")

    def assert_arrays_allclose(self, direct_arr, grpc_arr, rtol=1e-3, atol=1e-3, test_name="Endpoint"):
        """Valida que dois arrays possuem formato compativel e valores numericos equivalentes."""
        self.assertIsNotNone(direct_arr, f"[{test_name}] Array direto da SDK retornou None")
        self.assertIsNotNone(grpc_arr, f"[{test_name}] Array desempacotado do gRPC retornou None")

        # Se o gRPC retornou um array 1D achatado, ajusta o formato para coincidir com o da SDK
        if direct_arr.shape != grpc_arr.shape and direct_arr.size == grpc_arr.size:
            grpc_arr = grpc_arr.reshape(direct_arr.shape)

        self.assertEqual(
            direct_arr.shape, grpc_arr.shape,
            f"[{test_name}] Divergencia no Shape! Direto: {direct_arr.shape} vs gRPC: {grpc_arr.shape}"
        )

        max_diff = float(np.max(np.abs(direct_arr - grpc_arr)))
        
        try:
            np.testing.assert_allclose(
                direct_arr, grpc_arr, rtol=rtol, atol=atol,
                err_msg=f"[{test_name}] Falha na equivalencia numerica! Diferenca maxima: {max_diff}"
            )
            print(f"  [OK] [{test_name}] Paridade Perfeita! Shape={direct_arr.shape} | Diferenca Maxima={max_diff:.6e}")
        except AssertionError as e:
            print(f"  [ERRO] [{test_name}] Divergencia numerica detectada! Shape={direct_arr.shape} | Max Diff={max_diff:.6e}")
            raise e

    def test_01_get_metadata_parity(self):
        """Valida o endpoint GetMetadata e mede a latencia gRPC."""
        print("\n[TESTE] Testando paridade e tempo: GetMetadata")
        req = dna_model_service_pb2.MetadataRequest(organism=self.organism)
        
        t0_grpc = time.perf_counter()
        responses = list(self.stub.GetMetadata(req))
        t_grpc = time.perf_counter() - t0_grpc
        
        self.assertGreater(len(responses), 0, "Nenhuma resposta de metadados retornada via gRPC")
        
        self.metrics["GetMetadata"] = {"sdk": 0.0, "grpc": t_grpc}
        print(f"  [TEMPO] gRPC: {t_grpc:.4f}s")

    def test_02_predict_interval_parity(self):
        """Valida e compara tempo do PredictInterval (SDK vs gRPC)."""
        print("\n[TESTE] Testando paridade e tempo: PredictInterval")

        t0_sdk = time.perf_counter()
        direct_output = self.local_model.predict_interval(
            interval=self.interval,
            organism=dna_model.Organism.HOMO_SAPIENS,
            ontology_terms=['UBERON:0001157'],
            requested_outputs=[dna_model.OutputType.RNA_SEQ]
        )
        direct_arr = extract_numpy_from_sdk_output(direct_output)
        t_sdk = time.perf_counter() - t0_sdk

        req = dna_model_service_pb2.PredictIntervalRequest(
            interval=self.interval_proto,
            organism=self.organism,
            ontology_terms=[self.ontology_uberon],
            requested_outputs=[dna_model_pb2.OUTPUT_TYPE_RNA_SEQ]
        )
        
        t0_grpc = time.perf_counter()
        responses = list(self.stub.PredictInterval(iter([req])))
        grpc_arr = unpack_tensor(responses[0].output)
        t_grpc = time.perf_counter() - t0_grpc

        self.assertEqual(len(responses), 1, "Esperava exatamente 1 pacote gRPC no PredictInterval")
        self.assert_arrays_allclose(direct_arr, grpc_arr, test_name="PredictInterval")

        self.metrics["PredictInterval"] = {"sdk": t_sdk, "grpc": t_grpc}
        print(f"  [TEMPO] SDK: {t_sdk:.4f}s | gRPC: {t_grpc:.4f}s")

    def test_03_predict_variant_parity(self):
        """Valida e compara tempo do PredictVariant (SDK vs gRPC)."""
        print("\n[TESTE] Testando paridade e tempo: PredictVariant")

        t0_sdk = time.perf_counter()
        direct_output = self.local_model.predict_variant(
            interval=self.interval,
            variant=self.variant,
            organism=dna_model.Organism.HOMO_SAPIENS,
            ontology_terms=['UBERON:0001157'],
            requested_outputs=[dna_model.OutputType.RNA_SEQ]
        )
        direct_ref = extract_numpy_from_sdk_output(direct_output.reference)
        direct_alt = extract_numpy_from_sdk_output(direct_output.alternate)
        t_sdk = time.perf_counter() - t0_sdk

        req = dna_model_service_pb2.PredictVariantRequest(
            interval=self.interval_proto,
            variant=self.variant_proto,
            organism=self.organism,
            ontology_terms=[self.ontology_uberon],
            requested_outputs=[dna_model_pb2.OUTPUT_TYPE_RNA_SEQ]
        )
        
        t0_grpc = time.perf_counter()
        responses = list(self.stub.PredictVariant(iter([req])))
        
        grpc_ref = None
        grpc_alt = None

        # Procura os tensores em TODOS os pacotes recebidos, 
        # pois o servidor pode estar enviando a resposta em pedaços (chunks)
        for resp in responses:
            print(f"[Pacote] oneof payload = {resp.WhichOneof('payload')} | tamanho = {len(resp.SerializeToString())} bytes")
            if grpc_ref is None:
                grpc_ref = unpack_tensor(resp.reference_output)
            if grpc_alt is None:
                grpc_alt = unpack_tensor(resp.alternate_output)

        # Se depois de olhar todos os pacotes ainda estiver None, o bug é no Servidor gRPC!
        self.assertIsNotNone(grpc_ref, "[PredictVariant (Ref)] O servidor gRPC não enviou o array de referência em nenhum pacote do stream.")
        self.assertIsNotNone(grpc_alt, "[PredictVariant (Alt)] O servidor gRPC não enviou o array alternativo em nenhum pacote do stream.")
        t_grpc = time.perf_counter() - t0_grpc

        # reference_output e alternate_output pertencem ao mesmo 'oneof payload' do proto,
        # logo o servidor envia um pacote para cada um (1 ou 2 pacotes no total).
        self.assertGreater(len(responses), 0, "PredictVariant gRPC retornou 0 pacotes")
        self.assertLessEqual(len(responses), 2, "PredictVariant gRPC retornou mais pacotes que o esperado")
        self.assert_arrays_allclose(direct_ref, grpc_ref, test_name="PredictVariant (Ref)")
        self.assert_arrays_allclose(direct_alt, grpc_alt, test_name="PredictVariant (Alt)")

        self.metrics["PredictVariant"] = {"sdk": t_sdk, "grpc": t_grpc}
        print(f"  [TEMPO] SDK: {t_sdk:.4f}s | gRPC: {t_grpc:.4f}s")

    def test_04_score_interval_parity(self):
        """Valida e compara tempo do ScoreInterval (SDK vs gRPC)."""
        print("\n[TESTE] Testando paridade e tempo: ScoreInterval")

        t0_sdk = time.perf_counter()
        direct_scores = self.local_model.score_interval(
            interval=self.interval,
            organism=dna_model.Organism.HOMO_SAPIENS
        )
        if isinstance(direct_scores, list):
            direct_scores = direct_scores[0]
        direct_arr = extract_numpy_from_sdk_output(direct_scores)
        t_sdk = time.perf_counter() - t0_sdk

        req = dna_model_service_pb2.ScoreIntervalRequest(
            interval=self.interval_proto,
            organism=self.organism
        )
        
        t0_grpc = time.perf_counter()
        responses = list(self.stub.ScoreInterval(iter([req])))
        grpc_arr = unpack_tensor(responses[0].output)
        t_grpc = time.perf_counter() - t0_grpc

        self.assertGreater(len(responses), 0, "ScoreInterval gRPC retornou 0 pacotes")
        self.assert_arrays_allclose(direct_arr, grpc_arr, test_name="ScoreInterval")

        self.metrics["ScoreInterval"] = {"sdk": t_sdk, "grpc": t_grpc}
        print(f"  [TEMPO] SDK: {t_sdk:.4f}s | gRPC: {t_grpc:.4f}s")

    def test_05_score_ism_variant_parity(self):
        """Valida e compara tempo do ScoreIsmVariant (Streaming SDK vs gRPC)."""
        print("\n[TESTE] Testando paridade e tempo: ScoreIsmVariant")

        ism_window = genome.Interval(chromosome='chr22', start=36201690, end=36201710)
        ism_window_proto = dna_model_pb2.Interval(chromosome='chr22', start=36201690, end=36201710)

        t0_sdk = time.perf_counter()
        direct_scores = self.local_model.score_ism_variants(
            interval=self.interval,
            ism_interval=ism_window,
            organism=dna_model.Organism.HOMO_SAPIENS
        )
        flat_direct = []
        def _flatten(sub):
            if isinstance(sub, (list, tuple)):
                for e in sub:
                    _flatten(e)
            elif sub is not None:
                flat_direct.append(sub)
        _flatten(direct_scores)
        t_sdk = time.perf_counter() - t0_sdk

        req = dna_model_service_pb2.ScoreIsmVariantRequest(
            interval=self.interval_proto,
            ism_interval=ism_window_proto,
            organism=self.organism
        )
        
        t0_grpc = time.perf_counter()
        grpc_responses = list(self.stub.ScoreIsmVariant(iter([req])))
        t_grpc = time.perf_counter() - t0_grpc

        self.assertEqual(
            len(flat_direct), len(grpc_responses),
            f"Divergencia na quantidade de pacotes ISM! SDK={len(flat_direct)} vs gRPC={len(grpc_responses)}"
        )

        for idx in range(min(3, len(flat_direct))):
            direct_arr = extract_numpy_from_sdk_output(flat_direct[idx])
            grpc_arr = unpack_tensor(grpc_responses[idx].output)
            self.assert_arrays_allclose(direct_arr, grpc_arr, test_name=f"ScoreIsmVariant (Pacote {idx+1})")

        self.metrics["ScoreIsmVariant"] = {"sdk": t_sdk, "grpc": t_grpc}
        print(f"  [TEMPO] SDK: {t_sdk:.4f}s | gRPC: {t_grpc:.4f}s")


if __name__ == '__main__':
    unittest.main(verbosity=2)
