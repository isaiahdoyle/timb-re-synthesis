"""Functions for creating the figures found in the supporting paper.

To run, uncomment the figures to draw in the main guard and run the following
command from the main parent directory: `python -m specification.figures`
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline
from cvae import train_model

def visualize_dataset(path: str) -> None:
    """Visualize rating distributions for each feature across the dataset.

    Args:
        path: The path to the Excel file containing the audio ratings.
    """
    dataframe = pd.read_excel(path).iloc[:, 13:]

    breathiness = dataframe.iloc[:, 0]
    pitch = dataframe.iloc[:, 1]
    smoothness = dataframe.iloc[:, 2]
    tone = dataframe.iloc[:, 3]

    domain = np.linspace(1, 5, 50)

    breath_hist, _ = np.histogram(breathiness.to_numpy(), bins=50)
    breath_hist = gaussian_filter(breath_hist, 3)
    breath_hist = breath_hist / np.linalg.norm(breath_hist)
    
    pitch_hist, _ = np.histogram(pitch.to_numpy(), bins=50)
    pitch_hist = gaussian_filter(pitch_hist, 2)
    pitch_hist = pitch_hist / np.linalg.norm(pitch_hist)
    
    smooth_hist, _ = np.histogram(smoothness.to_numpy(), bins=50)
    smooth_hist = gaussian_filter(smooth_hist, 3)
    smooth_hist = smooth_hist / np.linalg.norm(smooth_hist)
    
    tone_hist, _ = np.histogram(tone.to_numpy(), bins=50)
    tone_hist = gaussian_filter(tone_hist, 3)
    tone_hist = tone_hist / np.linalg.norm(tone_hist)

    fig, (ax_b, ax_p, ax_s, ax_t) = plt.subplots(4, 1, sharex=True)


    ax_b.plot(domain, breath_hist, color='blue', label='breathiness')
    ax_b.legend()

    ax_p.plot(domain, pitch_hist, color='green', label='pitch')
    ax_p.legend()

    ax_s.plot(domain, smooth_hist, color='orange', label='smoothness')
    ax_s.legend()

    ax_t.plot(domain, tone_hist, color='red', label='tone')
    ax_t.legend()


def visualize_training_loss(epochs: int = 10) -> None:
    """Visualize training loss during a given number of epochs.
    
    Args:
        epochs: The number of training iterations.
    """
    domain = np.linspace(1, epochs+1, epochs*7)
    losses = train_model(save=False, epochs=epochs)

    # Smooth curve
    spline = make_interp_spline(domain, losses)
    domain = np.linspace(1, epochs+1, 500)
    losses = spline(domain)

    ax = plt.subplot()
    ax.plot(domain, losses)
    ax.set_xticks(np.linspace(1, epochs, epochs))


if __name__ == '__main__':
    # Uncomment desired figures
    # visualize_dataset('../audio_ratings.xlsx')
    # visualize_training_loss(6)

    plt.show()
