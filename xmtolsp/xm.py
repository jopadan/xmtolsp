from __future__ import annotations
import struct
from typing import ClassVar, Self
from dataclasses import dataclass, field, InitVar
from enum import Enum, auto
import numpy as np

POST_INIT = field(init=False)


@dataclass
class Struct:
    _pack_size: ClassVar[int]
    _pack_string: ClassVar[str]
    raw: InitVar[bytes]

    @classmethod
    def init(cls, raw, **kwargs) -> Self:
        args = struct.unpack(cls._pack_string, raw[:cls._pack_size])
        return cls(raw, *args, **kwargs)


@dataclass
class Header(Struct):
    _pack_size = 80
    _pack_string = '<17s20sB20sBBLHHHHHHHH'
    id: bytes
    mod_name: bytes
    strip: int
    tracker_name: bytes
    version_minor: int
    version_major: int
    header_size: int
    song_length: int
    song_restart_pos: int
    num_channels: int
    num_patterns: int
    num_instruments: int
    flags: int
    tempo: int
    bpm: int
    pattern_order: list[int] = POST_INIT

    def __post_init__(self, raw: bytes) -> None:
        self.pattern_order = list(raw[self._pack_size:self.size()])
        if self.id != b'Extended Module: ':
            raise ValueError("Input is not an XM file")

    def size(self) -> int:
        return 80 + self.header_size - 20


class PatternRow:
    note: int | None = None
    inst: int | None = None
    volume: int | None = None
    effect: int | None = None
    effect_param: int | None = None

    def __init__(self, data: bytes) -> None:
        if not data[0] & 0x80:
            self.note, self.inst, self.volume, self.effect, self.effect_param = data[0:5]
            return
        pack_byte = data[0]
        i = 1
        if pack_byte & 0x01:
            self.note = data[i]
            i += 1
        if pack_byte & 0x02:
            self.inst = data[i]
            i += 1
        if pack_byte & 0x04:
            self.volume = data[i]
            i += 1
        if pack_byte & 0x08:
            self.effect = data[i]
            i += 1
        if pack_byte & 0x10:
            self.effect_param = data[i]
            i += 1


@dataclass
class Pattern(Struct):
    _pack_size = 9
    _pack_string = '<LBHH'
    num_channels: InitVar[int] = field(kw_only=True)
    length: int
    pack_type: int
    num_rows: int
    data_size: int
    packed_data: bytes = POST_INIT
    rows: list[list[PatternRow]] = POST_INIT

    @classmethod
    def init(cls, raw: bytes, **kwargs) -> Self:
        obj = super().init(raw, num_channels=kwargs['num_channels'])
        return obj

    def __post_init__(self, raw: bytes, num_channels: int) -> None:
        self.packed_data = raw[self._pack_size:self.size()]
        data_left = self.packed_data
        rows = []
        while data_left:
            channels: list[PatternRow] = []
            for _ in range(num_channels):
                pack_byte = data_left[0]
                if not pack_byte & 0x80:
                    row_size = 5
                else:
                    row_size = pack_byte.bit_count()
                channels.append(PatternRow(data_left[:row_size]))
                data_left = data_left[row_size:]
            rows.append(channels)
        self.rows = rows


    def size(self) -> int:
        return 9 + self.data_size


@dataclass
class SampleInfo(Struct):
    _pack_size = 234
    _pack_string = '<L96s48s48sBBBBBBBBBBBBBBH22s'
    sample_header_size: int
    sample_keymaps: list[int]
    vol_env_pts: list[int]
    pan_env_pts: list[int]
    num_vol_pts: int
    num_pan_pts: int
    vol_sustain: int
    vol_loop_start: int
    vol_loop_end: int
    pan_sustain: int
    pan_loop_start: int
    pan_loop_end: int
    vol_type: int
    pan_type: int
    vib_type: int
    vib_sweep: int
    vib_depth: int
    vib_rate: int
    vol_fadeout: int
    reserved: bytes

    def __post_init__(self, _) -> None:
        vep = self.vol_env_pts
        pep = self.pan_env_pts
        self.vol_env_pts = [vep[i+1] << 8 + vep[i] for i in range(len(vep), 2)]
        self.pan_env_pts = [pep[i+1] << 8 + pep[i] for i in range(len(pep), 2)]


@dataclass
class Sample(Struct):
    _pack_size = 40
    _pack_string = '<LLLBbBBbB22s'
    length: int
    loop_start: int
    loop_length: int
    volume: int
    finetune: int
    flags: int
    pan: int
    rel_note: int
    reserved: int
    name: bytes
    data: bytes = POST_INIT

    class LoopType(Enum):
        NO_LOOP = auto()
        FORWARD_LOOP = auto()
        BI_LOOP = auto()

    @property
    def loop_type(self) -> LoopType:
        match self.flags & 3:
            case 0: return self.LoopType.NO_LOOP
            case 1: return self.LoopType.FORWARD_LOOP
            case 2: return self.LoopType.BI_LOOP
            case _: raise ValueError("Unknown loop type")

    @property
    def pcm16(self) -> bool:
        return self.flags & 16 != 0

    def __post_init__(self, raw: bytes) -> None:
        # Convert data from XM deltas to signed 8-bit RAW
        raw_data = raw[self._pack_size:self._pack_size+self.length]
        dtype = np.dtype('int16').newbyteorder('<') if self.pcm16 else np.int8
        data = np.frombuffer(raw_data, dtype)
        data = np.cumsum(data).astype(dtype)
        self.data = bytes(data)


@dataclass
class Instrument(Struct):
    _pack_size = 29
    _pack_string = '<L22sBH'
    inst_header_size: int
    inst_name: bytes
    inst_kind: int
    num_samples: int
    sample_info: SampleInfo | None = POST_INIT
    samples: list[Sample] = POST_INIT

    def __post_init__(self, raw: bytes) -> None:
        self.sample_info = SampleInfo.init(raw[self._pack_size:263]) if self.num_samples else None
        self.samples = []
        if self.sample_info is None:
            return
        # Parse sample headers/data
        raw = raw[self.inst_header_size:]
        header_size = self.sample_info.sample_header_size
        header = raw[:header_size * self.num_samples]
        data = raw[header_size * self.num_samples:]
        for _ in range(self.num_samples):
            sample = Sample.init(header[:header_size] + data)
            self.samples.append(sample)
            header = header[header_size:]
            data = data[sample.length:]
            
    def size(self) -> int:
        if self.sample_info is None:
            return self.inst_header_size
        header_size = self.sample_info.sample_header_size
        return self.inst_header_size + sum(header_size + sample.length for sample in self.samples)


class XmFile:
    header: Header
    patterns: list[Pattern]
    instruments: list[Instrument]

    def __init__(self, file: bytes) -> None:
        self.header = Header.init(file)
        file = file[self.header.size():]
        self.patterns = []
        for _ in range(self.header.num_patterns):
            self.patterns.append(pattern := Pattern.init(file, num_channels=self.header.num_channels))
            file = file[pattern.size():]
        self.instruments = []
        for _ in range(self.header.num_instruments):
            self.instruments.append(inst := Instrument.init(file))
            file = file[inst.size():]


def load(path: str) -> XmFile:
    with open(path, 'rb') as f:
        return XmFile(f.read())
