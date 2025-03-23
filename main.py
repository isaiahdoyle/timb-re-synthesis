import argparse
from freevc import FreeVC

# How to use:
# 1. Install WavLM-Large checkpoint (WavLM-Large.pt) and put in freevc/wavlm/
# 2. Install FreeVC checkpoint (freevc-s.pth) and put in freevc/checkpoints/
# 3. `pip install -r requirements.txt`
# 4. `python main.py <input-path> <output-path>`

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='checkpoints/freevc-s.pth', help='path to audio')
    parser.add_argument('output', type=str, default='output/freevc', help='path to output dir')
    args = parser.parse_args()

    model = FreeVC()
    src = model.get_content(args.path)
    tgt = model.get_timbre(args.path)
    model.synthesize(src, tgt, args.output)
