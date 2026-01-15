from pyannote.audio import Pipeline
import soundfile as sf
import torch

AUDIO_PATH = "/Users/fernando/Library/Application Support/Foxtrot-Delta-Pilot/meetings/2e09341e-b89e-4d4c-b87c-0f23d3838d2b/audio.wav"

pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1"
)

wav, sr = sf.read(AUDIO_PATH)

# Sørg for (channels, time)
if wav.ndim == 1:
    wav = wav[None, :]
else:
    wav = wav.T

waveform = torch.from_numpy(wav).float()

diarization = pipe({
    "waveform": waveform,
    "sample_rate": sr,
})

for segment, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{segment.start:6.2f}–{segment.end:6.2f}  {speaker}")
