'''A minimzed version of FreeVC by J. Li, W. Tu, and L. Xiao.

GitHub: https://github.com/OlaWod/FreeVC
Paper: https://arxiv.org/abs/2210.15418

See the included LICENSE.

Used to drive the analysis and resynthesis of speech samples, using one sample
as a 'source' to use the content (i.e., linguistic information) of, and another
as a 'target' to use for timbral information. This implementation assumes that
the same sample will be used as source and target, and expose the timbral
information such that it can be modified manually before resynthesis.
'''

import os
import torch
import librosa
import logging
from scipy.io.wavfile import write

import freevc.utils as utils
from freevc.models import SynthesizerTrn
from freevc.wavlm import WavLM
from freevc.mel_processing import mel_spectrogram_torch
from freevc.speaker_encoder.voice_encoder import SpeakerEncoder


logging.getLogger('numba').setLevel(logging.WARNING)

PTFILE: str = 'freevc/checkpoints/freevc-s.pth'

# From freevc(-s).json
CONFIG: dict = {
  "train": {
    "log_interval": 200,
    "eval_interval": 10000,
    "seed": 1234,
    "epochs": 10000,
    "learning_rate": 2e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 64,
    "fp16_run": False,
    "lr_decay": 0.999875,
    "segment_size": 8960,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "use_sr": True,
    "max_speclen": 128,
    "port": "8001"
  },
  "data": {
    "training_files":"filelists/train.txt",
    "validation_files":"filelists/val.txt",
    "max_wav_value": 32768.0,
    "sampling_rate": 16000,
    "filter_length": 1280,
    "hop_length": 320,
    "win_length": 1280,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": None
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [10,8,2,2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16,16,4,4],
    "n_layers_q": 3,
    "use_spectral_norm": False,
    "gin_channels": 256,
    "ssl_dim": 1024,
    "use_spk": False,  # True to use SpeakerEncoder, False to use MFCCs
  }
}


class FreeVC():
    outdir: str = 'output'
    smodel: SpeakerEncoder  # timbre
    cmodel: WavLM  # content

    def __init__(
            self,
            outdir: str = 'output',
            ptfile: str = PTFILE):
        os.makedirs(outdir, exist_ok=True)

        print("Loading model...")
        self.net_g = SynthesizerTrn(
            CONFIG["data"]["filter_length"] // 2 + 1,
            CONFIG["train"]["segment_size"] // CONFIG["data"]["hop_length"],
            **CONFIG["model"]).cpu()
        _ = self.net_g.eval()

        print("Loading checkpoint...")
        _ = utils.load_checkpoint(ptfile, self.net_g, None, True)

        print("Loading WavLM for content...")
        self.cmodel = utils.get_cmodel(0)

        if CONFIG["model"]["use_spk"]:
            print("Loading speaker encoder...")
            self.smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    def parse_text(self, path: str) -> list[tuple]:
        """Process a .txt file containing source/target/output files.

        Files should follow the format `TITLE|SRC.wav|TGT.wav`, where src.wav
        is the audio containing the content to extract, and TGT.wav is the
        audio containing the target timbre.

        Args:
            path: The filepath.

        Returns:
            A list of tuples containing (title, src, tgt) for each line.
        """
        print("Processing text...")
        # titles, srcs, tgts = [], [], []
        lines = []
        with open(path, "r") as f:
            for rawline in f.readlines():
                title, src, tgt = rawline.strip().split("|")
                lines.append((title, src, tgt))

        return lines

    def get_timbre(self, tgt: str, output: bool = True) -> torch.Tensor:
        if output:
            print(f'Getting timbre from {tgt}...', end='')

        wav_tgt, _ = librosa.load(tgt, sr=CONFIG["data"]["sampling_rate"])
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

        if CONFIG["model"]["use_spk"]:
            # Use SpeakerEncoder to analyze timbre
            timbre = self.smodel.embed_utterance(wav_tgt)
            timbre = torch.from_numpy(timbre).unsqueeze(0).cpu()

        else:
            # Use mel-spectrogram to analyze timbre
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cpu()
            timbre = mel_spectrogram_torch(
                wav_tgt,
                CONFIG["data"]["filter_length"],
                CONFIG["data"]["n_mel_channels"],
                CONFIG["data"]["sampling_rate"],
                CONFIG["data"]["hop_length"],
                CONFIG["data"]["win_length"],
                CONFIG["data"]["mel_fmin"],
                CONFIG["data"]["mel_fmax"]
            )

        if output:
            print(' done!')

        return timbre

    def get_content(self, src: str, output: bool = True) -> torch.Tensor:
        if output:
            print(f'Getting content from {src}...', end='')

        wav_src, _ = librosa.load(src, sr=CONFIG["data"]["sampling_rate"])
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).cpu()
        content = utils.get_content(self.cmodel, wav_src)

        if output:
            print(' done!')

        return content

    def synthesize(self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            output: str) -> None:
        if CONFIG["model"]["use_spk"]:
            audio = self.net_g.infer(src, g=tgt)
        else:
            audio = self.net_g.infer(src, mel=tgt)

        audio = audio[0][0].data.cpu().float().numpy()

        write(
            filename=output,
            rate=CONFIG["data"]["sampling_rate"],
            data=audio,
        )

        print(f"Output written to {output}")
