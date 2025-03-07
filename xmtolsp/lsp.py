from __future__ import annotations
from dataclasses import dataclass, field
import struct
import collections
from enum import IntEnum
from hashlib import md5
from pathlib import Path
import numpy as np
import sox
import libxmplite
from libxmplite import FrameInfo, ChannelInfo
import xm
from xm import Sample


@dataclass(frozen=True)
class SampleRef:
    """ A reference to a sample found when parsing the XM. LSP makes a unique
        "instrument" for each offset the sample is played at (e.g. using the
        Set Offset module command). So, each unique offset is also a unique
        SampleRef."""
    sample_id: int
    sample_offset: int


@dataclass(frozen=True)
class Instrument:
    """ An LSP instrument. The offsets are offsets from the beginning of the
        final LSP soundbank."""
    sample_offset: int
    word_count: int
    loop_offset: int
    loop_word_count: int

    def to_bytes(self):
        return struct.pack('>LHLH',
                           self.sample_offset,
                           self.word_count,
                           self.loop_offset,
                           self.loop_word_count)


class LspSample:
    """ An audio sample with all the info needed to generate the final
        soundbank and instrument entries."""
    sample_data: bytes
    length: int
    loop_start: int
    loop_length: int
    downsample_factor: float = 1

    def __init__(self, smp: Sample, max_len: int, min_per: float, target_per: int, no_trim: bool, no_downsample: bool) -> None:
        """ smp: The XM sample.
            max_len: The max length this sample was detected to have been
                      played in the XM (used for truncating).
            min_per: The minimum period this sample was detected to have been
                      played at in the XM (used for downsampling).
            target_per: The target period to downsample to.
            no_trim: Do not trim this sample at the max length.
            no_downsample: Do not downsample this sample."""
        if max_len == 0:
            self.sample_data = bytes()
            self.length = 0
            self.loop_start = 0
            self.loop_length = 0
            return
        # Check if downsampling is needed
        if not no_downsample and min_per < target_per:
            self.downsample_factor = min_per / target_per
        tfm = sox.Transformer()
        if self.downsample_factor < 1:
            tfm.set_output_format(bits=8, encoding='signed-integer', rate=22050*self.downsample_factor)
        else:
            tfm.set_output_format(bits=8, encoding='signed-integer')
        data = smp.data
        # Truncate data based on max_len
        # TODO: For now, looping samples can't be truncated within their loop
        # region, even if the sample never loops
        if not no_trim:
            max_len = max(max_len, smp.loop_start + smp.loop_length)
            max_len += max_len % 2
            data = data[:max_len]
        self.length = len(data)
        self.loop_start = smp.loop_start
        self.loop_length = smp.loop_length
        # Convert to 8-bit and/or downsample if necessary
        # NOTE: Giving sox in/out rates of 22050 is arbitrary and has no
        # effect. All that matters is the downsample factor from in to out.
        if smp.pcm16:
            data = bytes(tfm.build_array(input_array=np.frombuffer(data, np.dtype('int16').newbyteorder('<')), sample_rate_in=22050))
            self.length //= 2
            self.loop_start //= 2
            self.loop_length //= 2
        elif self.downsample_factor < 1:
            data = bytes(tfm.build_array(input_array=np.frombuffer(data, np.int8), sample_rate_in=22050))
        self.length = int(self.length * self.downsample_factor) >> 1 << 1
        self.loop_start = int(self.loop_start * self.downsample_factor) >> 1 << 1
        self.loop_length = int(self.loop_length * self.downsample_factor) >> 1 << 1
        # Check for ping-pong loop and render
        if smp.loop_type == smp.LoopType.BI_LOOP:
            loop_end = self.loop_start + self.loop_length
            data += data[loop_end:self.loop_start:-1]
            self.length += self.loop_length
            self.loop_length *= 2
        # Ensure first two bytes of non-looping sample are zero
        if not smp.loop_length and any(data[:2]):
            data = bytes(2) + data
            self.length += 2
        self.sample_data = data + bytes(len(data) % 2)


class LspFile:
    """ The final LSP score (.lsmusic) file."""
    version_major: int = 1
    version_minor: int = 25
    support_flags: int = 0 # %01 supports GetPos, %10 supports SetPos
    bpm: int
    esc_value_rewind: int
    esc_value_setbpm: int
    esc_value_getpos: int
    frame_count: int
    instrument_count: int
    instruments: list[Instrument]
    command_list_size: int
    commands: list[int]
    seq_count: int = 0
    seqs: list = []
    word_stream_size: int
    byte_stream_loop_pos: int = 0
    word_stream_loop_pos: int = 0
    word_stream: list[int]
    byte_stream: list[int]

    def __init__(self,
                 frame_count: int,
                 bpm: int,
                 rewind: int,
                 setbpm: int,
                 getpos: int,
                 samples: list[LspSample],
                 smp_refs: list[SampleRef],
                 cmd_table: list[int],
                 byte_loop: int,
                 word_loop: int,
                 word_stream: list[int],
                 byte_stream: list[int]) -> None:
        self.bpm = bpm
        self.esc_value_rewind = rewind
        self.esc_value_setbpm = setbpm
        self.esc_value_getpos = getpos
        self.frame_count = frame_count
        self.instrument_count = len(smp_refs)
        self.instruments = []
        bank_offsets = np.cumsum([0] + [len(smp.sample_data) for smp in samples]) + 4
        for smp_ref in smp_refs:
            sample = samples[smp_ref.sample_id]
            sample_offset = bank_offsets[smp_ref.sample_id] + smp_ref.sample_offset
            word_count = (sample.length - smp_ref.sample_offset) // 2
            if sample.loop_length == 0:
                loop_offset = sample_offset
                loop_word_count = 1
            else:
                loop_offset = sample_offset + sample.loop_start
                loop_word_count = (sample.loop_length - smp_ref.sample_offset) // 2
            self.instruments.append(Instrument(int(sample_offset), word_count, int(loop_offset), loop_word_count))
        self.command_list_size = len(cmd_table)
        self.commands = cmd_table
        self.word_stream_size = len(word_stream) * 2
        self.byte_stream_loop_pos = byte_loop
        self.word_stream_loop_pos = word_loop
        self.word_stream = word_stream
        self.byte_stream = byte_stream

    def to_bytes(self) -> bytes:
        out = bytes()
        out += struct.pack('>BBHHHHHLH',
                           self.version_major,
                           self.version_minor,
                           self.support_flags,
                           self.bpm,
                           self.esc_value_rewind,
                           self.esc_value_setbpm,
                           self.esc_value_getpos,
                           self.frame_count,
                           self.instrument_count)
        out += b''.join(inst.to_bytes() for inst in self.instruments)
        out += self.command_list_size.to_bytes(2)
        out += b''.join(cmd.to_bytes(2) for cmd in self.commands)
        out += self.seq_count.to_bytes(2)
        out += b''.join(seq.to_bytes() for seq in self.seqs) # TODO
        out += struct.pack('>LLL',
                           self.word_stream_size,
                           self.byte_stream_loop_pos,
                           self.word_stream_loop_pos)
        out += b''.join(word.to_bytes(2) for word in self.word_stream)
        out += b''.join(byte.to_bytes(1) for byte in self.byte_stream)
        return out


@dataclass
class ChannelState:
    """ The current state of an audio channel in a frame. Values that have not
        been updated this frame are None to indicate no LSP command is
        needed."""
    _last_vol: int
    _last_per: float
    _last_inst: SampleRef | None
    vol: int | None = field(default=None, init=False)
    per: float | None = field(default=None, init=False)
    inst: SampleRef | None = field(default=None, init=False)


class VoiceCode(tuple[IntEnum, IntEnum, IntEnum, IntEnum]):
    """ LSP instrument command data. A frame has flags for each audio channel
        that determine whether to play an instrument, reset AUDxLEN (always the
        frame after an instrument was played), or do nothing."""

    class Code(IntEnum):
        """ An enum of the possible instrument commands."""
        NONE = 0
        """ No action."""
        RESET_LEN = 1
        """ Reset AUDxPT and AUDxLEN to the instrument's loop values. This
            happens the frame after the instrument was played."""
        PLAY_WITHOUT_NOTE = 2
        """ Set an instrument without actually playing a note."""
        PLAY_INSTRUMENT = 3
        """ Set and play an instrument."""

    def __new__(cls, channels: list[ChannelState], last_channels: list[ChannelState]) -> VoiceCode:
        values = [cls.Code.NONE, ] * 4
        for i in range(4):
            # NOTE: Pretty sure PLAY_WITHOUT_NOTE happens when an instrument
            # is selected but no note is played. We're currently ignoring that
            # because it seems to have no effect in XM files.
            if last_channels[i].inst is not None and channels[i].inst is None:
                values[i] = cls.Code.RESET_LEN
            elif channels[i].inst is not None:
                values[i] = cls.Code.PLAY_INSTRUMENT
        return super().__new__(cls, tuple(values))


class LspFrame:
    """ A frame containing the channel states on that frame, as well as a
        command value calculated based on the channel states."""
    command: int
    channels: list[ChannelState]

    def __init__(self, channels: list[ChannelState], last_channels: list[ChannelState]) -> None:
        self.channels = channels
        voice_code = VoiceCode(channels, last_channels)
        cmd_inst = 0
        cmd_vol = 0
        cmd_per = 0
        for code, chan in zip(voice_code[::-1], self.channels[::-1]):
            cmd_inst <<= 2
            cmd_vol <<= 1
            cmd_per <<= 1
            cmd_inst |= code
            if chan.vol is not None:
                cmd_vol |= 1
            if chan.per is not None:
                cmd_per |= 1
        self.command = cmd_inst << 8 | cmd_vol << 4 | cmd_per

    def __eq__(self, value: object, /) -> bool:
        if type(value) is int:
            return self.command == value
        if type(value) is LspFrame:
            return self.command == value.command
        return False

    def __hash__(self) -> int:
        return hash(self.command)


class Lsp:
    xm_samples: list[Sample]
    xm_frames: list[FrameInfo]
    lsp_samples: list[LspSample]
    bpm: int
    loop_index: int
    ntsc: bool
    no_downsample: bool
    no_trim: bool
    min_per: int

    def __init__(self,
                 input_file: Path,
                 ntsc: bool = False,
                 no_trim: bool = False,
                 no_downsample: bool = False,
                 min_per: int = 124) -> None:
        self.ntsc = ntsc
        self.no_trim = no_trim
        self.no_downsample = no_downsample
        self.min_per = min_per
        xm_file = xm.load(str(input_file))
        self.bpm = xm_file.header.bpm
        self.xm_samples = [smp for inst in xm_file.instruments for smp in inst.samples]
        xmp = libxmplite.Xmp()
        xmp.load(str(input_file))
        xmp.start(22050)
        # Render frames
        self.xm_frames = []
        while (frame := xmp.play_frame()).loop_count == 0:
            del frame.channel_info[4:]
            self.xm_frames.append(frame)
        self.loop_index = [(f.pos, f.row) for f in self.xm_frames].index((frame.pos, frame.row))
        # Get commands and all unique sample references
        self.lsp_samples = self._get_samples()
        self.lsp_frames = self._get_frames()

    def _get_period(self, chan: ChannelInfo) -> float:
        """ Gets the Paula period that a channel is currently playing a sample
            at."""
        # NOTE: There are various different ways to calculate Amiga period
        # which don't seem to be fully consistent with one another. This method
        # gets the note frequency in Hz and divides it into the Amiga HRM clock
        # constant. Also, libxmp pitchbend is range -100 to 100.
        ticks_per_second = 3579545 if self.ntsc else 3546895
        note = chan.note + self.xm_samples[chan.sample].rel_note + chan.pitchbend / 100
        freq = 8363 * pow(2, note / 12.0) / 32
        return ticks_per_second / freq

    def _get_samples(self) -> list[LspSample]:
        """ Gets LSP samples given a list of XM samples and XM frames."""
        sample_count = len(self.xm_samples)
        # Get max replay pos of all samples
        max_lens = [0] * sample_count
        len_per_frame = [0] * sample_count
        min_pers = [65535.0] * sample_count
        # Get max length per frame of each sample (taking BPM and pitch into
        # account) and max position of each sample, which is usually less than
        # actual max because we only have the position as it is on the end of
        # each frame.
        # TODO: Better detect if a sample ever loops (or ends) because then
        # max_len should just be the whole sample length
        for frame in self.xm_frames:
            for chan in frame.channel_info:
                if chan.note == 255:
                    continue
                period = self._get_period(chan)
                frames_per_min = frame.bpm * 4 * 6
                us_per_frame = 1 / frames_per_min * 60 * 1000000
                us_per_interval = 0.279365 if self.ntsc else 0.281937
                us_per_sample = us_per_interval * period
                samples_per_frame = int(us_per_frame / us_per_sample + 0.5)
                s = chan.sample
                len_per_frame[s] = max(samples_per_frame, len_per_frame[s])
                min_pers[s] = min(period, min_pers[s])
                max_lens[s] = max(chan.position, max_lens[s])
        # Now, round up max_lens to the next frame using len_per_frame
        for i in range(sample_count):
            if len_per_frame[i] == 0:
                continue
            lpf = len_per_frame[i]
            max_lens[i] = lpf * int(max_lens[i] / lpf) + lpf
        # Build samples. Starting offset 4 accounts for .lsbank 4-byte header.
        samples: list[LspSample] = []
        for xm_smp, max_len, min_per in zip(self.xm_samples, max_lens, min_pers):
            samples.append(LspSample(xm_smp, max_len, min_per, self.min_per, self.no_trim, self.no_downsample))
        return samples

    def _get_frames(self) -> list[LspFrame]:
        """ Gets LSP frames from the list of XM frames, XM samples (for root
            note), and LSP samples (for downsample factor)."""
        last_changed_pers = [0.0] * 4
        last_changed_vols = [0] * 4
        last_changed_insts: list[SampleRef | None] = [None] * 4
        last_chan_cmds = [ChannelState(0, 0, None)] * 4
        lsp_frames: list[LspFrame] = []
        event_this_row = [False] * 4
        for frame in self.xm_frames:
            # Hack to only allow one event per row. libxmplite reports an event
            # on every frame of the row, so this avoids duplicate events.
            if frame.frame == 0:
                event_this_row = [False] * 4
            chan_cmds: list[ChannelState] = []
            for i, chan in enumerate(frame.channel_info):
                period = self._get_period(chan)
                chan_cmd = ChannelState(chan.volume, period, last_changed_insts[i])
                if not event_this_row[i] and chan.event and (chan.event.fxt == 9 or chan.event.note and chan.position == 0):
                    smp_ref = SampleRef(chan.sample, int(chan.position * self.lsp_samples[chan.sample].downsample_factor))
                    chan_cmd.inst = smp_ref
                    chan_cmd._last_inst = smp_ref
                    last_changed_insts[i] = smp_ref
                    event_this_row[i] = True
                if chan.volume != last_changed_vols[i]:
                    chan_cmd.vol = chan_cmd._last_vol
                    last_changed_vols[i] = chan_cmd.vol
                if chan_cmd._last_inst:
                    smp_ref = chan_cmd._last_inst
                    period /= self.lsp_samples[smp_ref.sample_id].downsample_factor
                    if period != last_changed_pers[i]:
                        chan_cmd.per = period
                        chan_cmd._last_per = period
                        last_changed_pers[i] = period
                chan_cmds.append(chan_cmd)
            lsp_frames.append(LspFrame(chan_cmds, last_chan_cmds))
            last_chan_cmds = chan_cmds
        return lsp_frames

    def build(self) -> tuple[bytes, bytes]:
        """ Builds this object into the final LSP byte streams."""
        smp_refs = list(set(chan.inst for frame in self.lsp_frames for chan in frame.channels if chan.inst))
        # Generate command table out of all unique LSP frame commands, sorted
        # by most common (because commands >= index 256 take longer to execute)
        command_counts = collections.Counter(self.lsp_frames)
        cmd_table = [c[0].command for c in command_counts.most_common()]
        # Now, next three unused commands become magic numbers for special
        # commands. These are ensured >= index 256 so LSP can save cycles by
        # only checking for magic numbers if index >= 256.
        if len(cmd_table) < 255:
            cmd_table += [0] * (255 - len(cmd_table))
        i = 1
        while i in cmd_table: i += 1
        cmd_table.append(esc_rewind := i)
        while i in cmd_table: i += 1
        cmd_table.append(esc_setbpm := i)
        while i in cmd_table: i += 1
        cmd_table.append(esc_getpos := i)
        # Insert empty command at multiples of 256. LSP command index is 8-bit,
        # so 00 means "increment 256" and not "index 0".
        for i in range(0, len(cmd_table), 256):
            cmd_table.insert(i, 0)
        # Setup final data stream by iterating through frames
        byte_stream: list[int] = []
        word_stream: list[int] = []
        byte_loop = 0
        word_loop = 0
        for i, frame in enumerate(self.lsp_frames):
            if i == self.loop_index:
                byte_loop = len(byte_stream)
                word_loop = len(word_stream) * 2
            # Write command index, omitting index 0.
            # Each +256 index is a 00 byte, e.g. index 520 is 000008
            cmd_index = cmd_table.index(frame.command, 1)
            h, l = cmd_index // 256, cmd_index % 256
            byte_stream += [0] * h + [l]
            # Write channel commands. LSP expects reverse order (from 3 to 0).
            chans = frame.channels[::-1]
            byte_stream += [c.vol for c in chans if c.vol is not None]
            word_stream += [round(c.per) for c in chans if c.per and c._last_inst]
            # If any instruments played, write instrument offset to word stream
            current_offset = -12
            for inst in [c.inst for c in chans if c.inst]:
                inst_index = smp_refs.index(inst)
                offset = inst_index * 12 - current_offset
                word_stream.append(offset % 0x10000)
                current_offset += offset + 6
        # Store rewind special command
        cmd_index = cmd_table.index(esc_rewind)
        h, l = cmd_index // 256, cmd_index % 256
        byte_stream += [0] * h + [l]
        # Build final LSP file
        lsp_file = LspFile(len(self.lsp_frames),
                           self.bpm,
                           esc_rewind,
                           esc_setbpm,
                           esc_getpos,
                           self.lsp_samples,
                           smp_refs,
                           [cmd for cmd in cmd_table],
                           byte_loop,
                           word_loop,
                           word_stream,
                           byte_stream)
        # Get raw bytes and add final checksum header
        # NOTE: Checksum is calculated differently from original LSPConvert,
        # but it still serves the same purpose of giving LSP files a unique ID.
        bank_bytes = b''.join([s.sample_data for s in self.lsp_samples])
        lsp_bytes = lsp_file.to_bytes()
        checksum = md5(bank_bytes + lsp_bytes, usedforsecurity=False).digest()[:4]
        bank_bytes = checksum + bank_bytes
        lsp_bytes = b'LSP1' + checksum + lsp_bytes
        return bank_bytes, lsp_bytes
