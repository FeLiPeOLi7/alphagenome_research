from alphagenome.protos import dna_model_pb2 as _dna_model_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VariantScorerCalibration(_message.Message):
    __slots__ = ("quantiles", "quantile_probabilities", "tracks_metadata")
    QUANTILES_FIELD_NUMBER: _ClassVar[int]
    QUANTILE_PROBABILITIES_FIELD_NUMBER: _ClassVar[int]
    TRACKS_METADATA_FIELD_NUMBER: _ClassVar[int]
    quantiles: _containers.RepeatedScalarFieldContainer[float]
    quantile_probabilities: _containers.RepeatedScalarFieldContainer[float]
    tracks_metadata: _dna_model_pb2.TracksMetadata
    def __init__(self, quantiles: _Optional[_Iterable[float]] = ..., quantile_probabilities: _Optional[_Iterable[float]] = ..., tracks_metadata: _Optional[_Union[_dna_model_pb2.TracksMetadata, _Mapping]] = ...) -> None: ...

class CalibrationScores(_message.Message):
    __slots__ = ("scorer_to_calibration",)
    class ScorerToCalibrationEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: VariantScorerCalibration
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[VariantScorerCalibration, _Mapping]] = ...) -> None: ...
    SCORER_TO_CALIBRATION_FIELD_NUMBER: _ClassVar[int]
    scorer_to_calibration: _containers.MessageMap[str, VariantScorerCalibration]
    def __init__(self, scorer_to_calibration: _Optional[_Mapping[str, VariantScorerCalibration]] = ...) -> None: ...
