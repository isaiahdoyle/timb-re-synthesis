import argparse
import torch

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk

from pathlib import Path

# freevc, cvae imported in main guard


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default=None, help='path to speech input')
    parser.add_argument('--output', '-o', type=str, default='output.wav', help='path to output (.wav expected)')
    parser.add_argument('--ui', action='store_true', help='user interface (IN PROGRESS)')
    parser.add_argument('--pre', type=str, default=None, help='breathiness, pitch, smoothness, tone of input rated 1-5 (e.g., 1234)')
    parser.add_argument('--post', type=str, default=None, help='desired breathiness, pitch, smoothness, tone rated 1-5 (e.g., 1234)')
    args = parser.parse_args()

    # --- UI STILL UNDER DEVELOPMENT ---
    if args.ui:
        root = Tk()
        root.title('TransVox')
        root.geometry('512x384')

        mainframe = ttk.Frame(root, padding='32 32 64 64')
        mainframe.grid(column=0, row=0, sticky=(N,W,S,E))
        root.columnconfigure((0,1,2,3), minsize=32)
        root.rowconfigure(0, weight=1)

        # --- INPUT ---
        upload_btn = Button(mainframe, text='Upload File', command=askopenfilename)
        upload_btn.grid(column=1, row=0, columnspan=2, pady=32)

        breath_pre = DoubleVar()
        breath_pre_scl = Scale(mainframe, orient=VERTICAL, length=100, from_=1, to=5, resolution=0.1, showvalue=False, variable=breath_pre)
        breath_pre_scl.grid(column=0, row=1)

        pitch_pre = DoubleVar()
        pitch_pre_scl = Scale(mainframe, from_=1, to=5, resolution=0.1, showvalue=False, variable=pitch_pre)
        pitch_pre_scl.grid(column=1, row=1)

        smooth_pre = DoubleVar()
        smooth_pre_scl = Scale(mainframe, from_=1, to=5, resolution=0.1, showvalue=False, variable=smooth_pre)
        smooth_pre_scl.grid(column=2, row=1)

        tone_pre = DoubleVar()
        tone_pre_scl = Scale(mainframe, from_=1, to=5, resolution=0.1, showvalue=False, variable=tone_pre)
        tone_pre_scl.grid(column=3, row=1)

        # --- OUTPUT ---
        generate_btn = Button(mainframe, text='Generate')
        generate_btn.grid(column=1, row=2, columnspan=2, pady=32)

        # upload_btn.focus()

        root.mainloop()
        exit()
    # ----------------------------------

    elif args.path is None:
        raise ValueError('Path is required but was not provided')
    
    from freevc import FreeVC
    from cvae import CVAE

    outdir = Path(args.output).parent

    timbre_model = FreeVC(outdir=outdir, use_spk=True)
    content = timbre_model.get_content(args.path)
    timbre = timbre_model.get_timbre(args.path)

    if args.pre is not None:
        # Load trained CVAE
        model = CVAE(feature_size=256, latent_size=20, class_size=4)
        model.load_state_dict(torch.load('cvae.pth'))
        model.eval()  # inference mode

        # Get pre- (cx) and post- (cz) encoder conditions
        # c = [[breathiness, pitch, smoothness, tone]]
        cx = torch.tensor([[int(i) for i in args.pre]])

        if args.post is None:
            cz = cx
        else:
            cz = torch.tensor([[int(i) for i in args.post]])

        # Get new timbre using condition
        modified_timbre = model(timbre, cx, cz)[0]
    
    else:
        # No change specified, just resynthesize
        modified_timbre = timbre

    timbre_model.synthesize(content, modified_timbre, args.output)
