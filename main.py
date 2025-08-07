import argparse
import torch

from pathlib import Path
from freevc import FreeVC
from cvae import CVAE
from timbre_modelling import modify_timbre


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='input file')
    parser.add_argument('--output', '-o', type=str, default='output/', help='output directory')
    parser.add_argument('--dataset', '-d', type=str, default='audio_ratings.xlsx', help='path to dataset')
    parser.add_argument('--label', '-l', type=str, default='Average_Pitch', help='timbral descriptor')
    args = parser.parse_args()

    timbre_model = FreeVC(outdir=args.output, use_spk=True)
    content = timbre_model.get_content(args.path)
    timbre = timbre_model.get_timbre(args.path)

    model = CVAE(feature_size=256, latent_size=20, class_size=4)
    model.load_state_dict(torch.load('cvae.pth'))
    model.eval()  # for inference

    # c = [[breathiness, pitch, smoothness, tone]]
    modified_timbre = model(timbre, torch.tensor([[1, 1, 3, 5]]))[0]

    output = Path(args.output) / f'{Path(args.path).stem}-cvae.wav'
    timbre_model.synthesize(content, modified_timbre, output)
