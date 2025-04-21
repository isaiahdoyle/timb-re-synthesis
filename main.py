import argparse

from pathlib import Path
from freevc import FreeVC
from timbre_modelling import modify_timbre


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='input file')
    parser.add_argument('--output', '-o', type=str, default='output/', help='output directory')
    parser.add_argument('--dataset', '-d', type=str, default='audio_ratings.xlsx', help='path to dataset')
    parser.add_argument('--label', '-l', type=str, default='Average_Pitch', help='timbral descriptor')
    args = parser.parse_args()

    model = FreeVC(outdir=args.output)
    src = model.get_content(args.path)
    tgt = model.get_timbre(args.path)

    print('Modifying timbre...')
    pitch1, pitch2, pitch3 = modify_timbre(tgt, 'audio_ratings.xlsx', label=args.label, model=model)

    print('Synthesizing results...')
    model.synthesize(src, tgt, Path(args.output) / f'{Path(args.path).stem}_unmodified.wav')
    model.synthesize(src, pitch1, Path(args.output) / f'{Path(args.path).stem}_{args.label}-1.wav')
    model.synthesize(src, pitch2, Path(args.output) / f'{Path(args.path).stem}_{args.label}-2.wav')
    model.synthesize(src, pitch3, Path(args.output) / f'{Path(args.path).stem}_{args.label}-3^4.wav')
