import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from freevc import FreeVC
from typing import Optional


def reduce_spectrogram(samples: pd.DataFrame, model: FreeVC):
    spectra = torch.zeros(len(samples), 80)

    for num, (sample, _) in enumerate(samples.iterrows()):
        sample = str(sample).replace('.mp3', '.wav')

        spectrogram = model.get_timbre(f'audio/converted/{sample}', False)

        reduced = torch.mean(spectrogram, dim=2)
        spectra[num] = reduced[0,:]

    averaged = torch.mean(spectra, dim=0)

    return averaged


def get_spectra_by_label(dataset: str, label: str, model: FreeVC):
    '''Return the segmented average MFCC cepstra for a dataset.

    The MFCC cepstrograms are first averaged for all samples in the dataset,
    resulting in a single cepstrum representing the timbres for every sample.
    The cepstra are then grouped by their rating corresponding to the given
    label and averaged again, to create 5 spectra which represent the average
    cepstra of each sample for ratings from 1-2, 2-3, 3-4, and 4-5.

    The dataset is expected to be an Excel adhering to the following format:
            File Name | label1 | label2 | ...
    path/to/file1.wav |      1 |      2 | ...
    path/to/file2.wav |    3.5 |      5 | ...

    Args:
        dataset: The path to the .xlsx dataset.
        labels: The labels (corresponding to Excel headers) to use.

    Returns:
        A DataFrame containing the average cepstrum of all samples in the
        dataset for each segment (1-2, 2-3, 3-4, 4-5) and label.
    '''
    samples = pd.read_excel(dataset, index_col=0, header=0)

    seg1 = samples.loc[(1 <= samples[label]) & (samples[label] <= 2), [label]]
    spec1 = reduce_spectrogram(seg1, model=model)

    seg2 = samples.loc[(2 < samples[label]) & (samples[label] <= 3), [label]]
    spec2 = reduce_spectrogram(seg2, model=model)

    seg3 = samples.loc[(3 < samples[label]) & (samples[label] <= 4), [label]]
    spec3 = reduce_spectrogram(seg3, model=model)

    seg4 = samples.loc[(4 < samples[label]) & (samples[label] <= 5), [label]]
    spec4 = reduce_spectrogram(seg4, model=model)

    return (spec1, spec2, spec3, spec4)


def modify_timbre(
        timbre: torch.Tensor,
        dataset: str,
        label: str,
        model: Optional[FreeVC] = None,
        plot: bool = False) -> torch.Tensor:
    '''Modify a mel-cepstrogram timbre representation using a labelled dataset.

    Args:
        timbre: The Tensor (size [1, N, M] where N is the number of MFCCs and M
          is the number of spectra for the audio file) representing a mel-
          cepstrogram for an audio file.
        dataset: The dataset containing labelled audio files.
        label: The header for the column in the dataset corresponding to the
          timbral descriptor to be affected.
        model: The FreeVC model supporting the get_content() and get_timbre()
          methods for audio files.

    Returns:
        The modified spectrogram.
    '''
    if model is None:
        model = FreeVC()

    # Segment dataset spectra by rating
    spec1, spec2, spec3, spec4 = get_spectra_by_label(
        dataset=dataset, label=label, model=model
    )

    # Get differences between target and input segments
    diff1 = spec1 / spec4
    diff2 = spec2 / spec4
    diff3 = spec3 / spec4

    # Apply differences to input timbre
    timbre1 = timbre * torch.pow(diff1[None,:,None], 8)
    timbre2 = timbre * torch.pow(diff2[None,:,None], 8)
    timbre3 = timbre * torch.pow(diff3[None,:,None], 4)

    if plot:
        plt.title('MFCCs of segmented ratings')
        ax_mfcc = plt.subplot(3, 1, (1,2))
        ax_mfcc.plot(spec1, color='blue', label='lowest')
        ax_mfcc.plot(spec2, color='green', label='lower')
        ax_mfcc.plot(spec3, color='orange', label='low')
        ax_mfcc.plot(spec4, color='red', label='high', linestyle='--')
        ax_mfcc.legend()

        ax_diff = plt.subplot(3, 1, 3)
        ax_diff.plot(diff1, color='blue', label='lowest')
        ax_diff.plot(diff2, color='green', label='lower')
        ax_diff.plot(diff3, color='orange', label='low')
        ax_diff.plot(np.full(diff1.shape, 1), color='red', label='high', linestyle='--')

    return (timbre1, timbre2, timbre3)
