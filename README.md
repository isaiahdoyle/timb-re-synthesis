# timb-re-synthesis

This program supports non-realtime vocal resynthesis, applying specified timbral descriptors to shift the timbre of a speech sample toward a particular target. Our end goal is to support imitation-based voice training for trans individuals, allowing users to modify their voice to define a specific timbral goal.

An initial phase of development was conducted from January to April 2025, followed by a second phase from May to August. Both resulting papers can be found in `specification/phase1.pdf` and `specification/phase2.pdf` respectively.

## Installation

1. Speech synthesis requires the installation of FreeVC and WavLM checkpoints for speech analysis and synthesis:

    - [Install the WavLM-Large checkpoint](https://github.com/microsoft/unilm/tree/master/wavlm) and place the `.pt` file in `freevc/wavlm/`.

    - [Install the FreeVC checkpoint](https://onedrive.live.com/?id=537643E55991EE7B%219178&resid=537643E55991EE7B%219178&e=UlhRR5&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbnZ1a1ZubFEzWlR4MXJqck9aMmFiQ3d1QkFoP2U9VWxoUlI1&cid=537643e55991ee7b&v=validatepermission) and place the `.pth` file in `freevc/checkpoints/`.

2. (Recommended) create a virtual environment for installing dependencies and running the program:

    - `pip install venv; venv ./virtenv/` creates a virtual environment in the current working directory.

    - Type `source ./virtenv/bin/activate` on Linux/MacOS or `.\virtenv\Scripts\activate.bat` on Windows to enter the virtual environment. Type `deactivate` to exit the virtual environment.

3. Run `pip install -r requirements.txt` to install python-specific dependencies.


## Usage

```
usage: main.py [-h] [--output OUTPUT] [--ui] [--pre PRE] [--post POST] path

positional arguments:
  path                 path to speech input

options:
  -h, --help           show this help message and exit
  --output, -o OUTPUT  path to output (.wav expected)
  --pre PRE            breathiness, pitch, smoothness, tone of input rated 1-5 (e.g., 1234)
  --post POST          desired breathiness, pitch, smoothness, tone rated 1-5 (e.g., 1234)
```

### Example.
```
$ python3 main.py input/voice1.wav -o output/voice1-out.wav --pre=1234 --post=4321
```


## Future Work

The most recent phase of development focused on the implementation of a conditional variational autoencoder (CVAE) to support the modification of timbre vectors yielded from FreeVC's speaker encoder. The CVAE in its current state is capable of reconstructing polarized timbres from the range of 'perceptibly male' and 'perceptibly female', but struggles to represent timbres outside of that traditional binary. This is seen in a considerable trough in the dataset with respect to neutrally-rated voices in the dataset (see `specification/phase2.pdf`), and is likely the cause of the CVAE's difficulty in representing these voices. To accommodate all kinds of vocal timbres, we call for the development of a gender-diverse speech dataset for CVAE training. There are a number of important considerations when developing such a dataset, so recommendations are proposed in `phase2.pdf` to guide that discussion.
