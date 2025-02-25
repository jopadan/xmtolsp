# xmtolsp

xmtolsp is a tool that converts FastTracker 2 XM modules into Amiga LSP (LightSpeed Player) format.

## About

Historically, Amiga developers have primarily used ProTracker MOD format for music in games, demos, etc. Although many productions play MOD files directly using libraries like [PTPlayer](https://aminet.net/package/mus/play/ptplayer), there are computationally faster options. [LightSpeed Player](https://github.com/arnaud-carre/LSPlayer) "renders" MOD files into a lower-level format that provides the Amiga audio hardware with pre-baked, frame-by-frame instructions.

Because the LSP format is little more than a set of low-level instructions for instrument offset, length, volume, and pitch, it's feasible to render other tracker module formats into LSP.

## Advantages

XM modules have a number of advantages over MOD to make composers' lives easier:

- More than 32 samples can be used.
- More granular finetune (-128 to 127).
- An instrument can be made up of multiple samples, each assigned to a note range.
- Instruments can have an ADSR volume envelope.
- Samples can have a root note, meaning notes plotted on the tracker are accurate to the actual tone of the sample.
- Samples can be high quality (16-bit, 48000Hz).

xmtolsp supports all of these features. Notably, high-quality samples are downsampled to the best quality possible on Amiga, such that the highest note any sample reaches will correspond to Amiga's max sample rate. That means downsampling is no longer a manual process, nor does the composition of the module have any dependency on the sample rate.

## Restrictions

xmtolsp only processes the first 4 channels in the XM file, because Amiga does not support more than 4 audio channels. Any panning-related information is also ignored.

## Requirements

xmtolsp is tested on Python 3.13 and requires some external modules.

Install requirements using `pip` or the method of your choice:

```console
$ pip install -r requirements.txt
```

## Usage

```
python xmtolsp [-h] [-q] [-o DIRECTORY] [--min-per PERIOD] [--no-downsample] [--no-trim] [--ntsc] [--version] input_file
```

- `input_file`: The input XM file.
- `-q, --quiet`: Run silently (do not output any text to stdout).
- `-o DIRECTORY`: Specify output directory (default: current directory).
- `--min-per PERIOD`: Specify a Paula period that samples will be downsampled to. For each sample, the highest note it reaches will be played at this period. (default: 124)
- `--no-downsample`: Do not perform any downsampling on samples.
- `--no-trim`: Do not trim any samples to their max playback position. By default, xmtolsp trims sample data that is never reached during playback.
- `--ntsc`: Use NTSC clock rate (instead of PAL) when calculating Paula period.
- `--version`: Print xmtolsp version number and exit.

## Planned features (not yet supported)

- LSP special commands (set BPM, get/set position)
- Per-sample settings for downsample amount, and whether to trim

## Issues

xmtolsp has not yet been thoroughly stress-tested, so there will possibly be issues with modules featuring complex behavior, or samples that require a high level of accuracy in their timing. Feel free to report any issues that arise, including a link to the affected XM file if applicable.
