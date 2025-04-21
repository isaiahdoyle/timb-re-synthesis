# timb-re-synthesis
Non-realtime vocal resynthesis using applied timbral descriptors


## Installation

1. Speech synthesis requires the installation of FreeVC and WavLM checkpoints for analysis and synthesis, respectively:

    - [Install the WavLM-Large checkpoint](https://github.com/microsoft/unilm/tree/master/wavlm) and place the `.pt` file in `freevc/wavlm/`.

    - [Install the FreeVC checkpoint](https://onedrive.live.com/?id=537643E55991EE7B%219178&resid=537643E55991EE7B%219178&e=UlhRR5&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbnZ1a1ZubFEzWlR4MXJqck9aMmFiQ3d1QkFoP2U9VWxoUlI1&cid=537643e55991ee7b&v=validatepermission) and place the `.pth` file in `freevc/checkpoints/`.

2. (Recommended) create a virtual environment for installing dependencies and running the program:

    - `pip install venv; venv ./virtenv/` creates a virtual environment in the current working directory.

    - Type `source ./virtenv/bin/activate` on Linux/MacOS or `.\virtenv\Scripts\activate.bat` on Windows to enter the virtual environment. Type `deactivate` to exit the virtual environment.

3. Run `pip install -r requirements.txt` to install dependencies.


## Usage

```
usage: python main.py [-h] [-o OUTPUT] [-d DATASET] [-l LABEL] path

positional arguments:
  path                  input file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT             output directory
  -d DATASET            path to dataset
  -l LABEL              timbral descriptor (Average_Pitch, Average_Tone, Average_Breathiness, or Average_Smoothness)
```

*Note: currently, input audio files are required to be 24kHz sampling rate, 16-bit `.wav` files. Some example files meeting this specification are provided in `input/`.*
