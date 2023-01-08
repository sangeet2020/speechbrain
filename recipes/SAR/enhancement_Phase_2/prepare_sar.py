"""
Data prepration recipe for SepFormer speech
enhancement fine-tuning on SAR domain noisy audio
dataset.
The noisy dataset has been synthesized synthetically
using SAR domain noises-
    - Breathing-noise
    - Emergency-vehicle-and-siren-noise
    - Engine-noise
    - Chopper-noise
    - Static-radio-noise

Author
 * Sangeet Sagar 2022

"""

import os
import csv
import glob
import logging
from tqdm import tqdm
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)


def prepare_sar_csv(
    datapath,
    savepath,
    skip_prep=False,
    fs=16000,
):
    """
    Prepare CSV files for train/valid/test set

    Arguments
    ---------
    datapath : str
        Path to synthesized SAR data
    savepath : str
        Path to save the csv files
    skip_prep : bool
        If True, data preparation is skipped.
    fs : int
        Sampling rate. Defaults to 16000.
    """

    if skip_prep:
        return

    # Create training data
    msg = "Preparing DNS data as csv files in %s " % (savepath)
    logger.info(msg)
    
    # Setting the save folder
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # Train, test, valid
    create_sar_csv(datapath, savepath, fs, set="train")
    create_sar_csv(datapath, savepath, fs, set="valid")
    create_sar_csv(datapath, savepath, fs, set="test")

def create_sar_csv(datapath, savepath, fs=16000, set="train"):
    """
    Create CSV files for train, valid and test set.

    Arguments:
    ----------
    datapath :str
        Path to synthesized SAR data
    savepath : str
        Path to save the csv files
    fs : int
        Sampling rate. Defaults to 16000.
    set : str
        CSV prepration for train/valid/test
    """
    if fs == 8000:
        sample_rate = "8k"
    elif fs == 16000:
        sample_rate = "16k"
    else:
        raise ValueError("Unsupported sampling rate")
    
    clean_fullpaths = []
    noise_fullpaths = []
    noisy_fullpaths = []
    language = []
    lang = "de"

    # If csv already exists, we skip the data preparation
    csv_path = {}
    csv_path[set] = os.path.join(
        savepath, "sar_{}_{}".format(sample_rate, set) + ".csv"
    )
    
    if skip(csv_path[set]):

        msg = "%s already exists, skipping data preparation!" % (csv_path[set])
        logger.info(msg)

        return    

    clean_f1_path = extract_files(os.path.join(datapath, set), type="clean")
    noise_f1_path = extract_files(os.path.join(datapath, set), type="noise")
    noisy_f1_path = extract_files(os.path.join(datapath, set), type="noisy")

    language.extend([lang] * len(clean_f1_path))
    clean_fullpaths.extend(clean_f1_path)
    noise_fullpaths.extend(noise_f1_path)
    noisy_fullpaths.extend(noisy_f1_path)

    # Write CSV for train and dev
    msg = "Writing " + set + " csv files"
    logger.info(msg)
    write2csv(
        language,
        clean_fullpaths,
        noise_fullpaths,
        noisy_fullpaths,
        csv_path[set],
        fs=16000,
    )

def extract_files(datapath, type=None):
    """
    Given a dir-path, it extracts full path of all wav files 
    and sorts them.

    Arguments:
    ----------
    datapath :str
        Path to synthesized SAR data
    type : str
        Type of split: clean, noisy, noise.

    Returns
    -------
    list
        Sorted list of all wav files found in the given path.
    """
    if type:
        path = os.path.join(datapath, type)
        files = glob.glob(path + "/*.wav")

        # Sort all files based on the suffixed file_id (ascending order)
        files.sort(key=lambda f: int(f.split("fileid_")[-1].strip(".wav")))
    else:
        # Sort all files by name
        files = sorted(glob.glob(datapath + "/*.wav"))

    return files

def write2csv(
    language,
    clean_fullpaths,
    noise_fullpaths,
    noisy_fullpaths,
    savepath,
    fs=16000,
):
    """
    Write data to CSV file in an appropriate format.

    Arguments
    ---------
    language : str
        Language of audio file
    clean_fullpaths : str
        Path to clean audio files of a split in the train/valid-set
    noise_fullpaths : str
        Path to noise audio files of a split in the train/valid-set
    noisy_fullpaths : str
        Path to noisy audio files of a split in the train/valid-set
    savepath : str
        Path to save the csv files
    fs : int
        Sampling rate. Defaults to 16000.
    """
    csv_columns = [
        "ID",
        "language",
        "duration",
        "clean_wav",
        "clean_wav_format",
        "clean_wav_opts",
        "noise_wav",
        "noise_wav_format",
        "noise_wav_opts",
        "noisy_wav",
        "noisy_wav_format",
        "noisy_wav_opts",
    ]

    # Retreive duration of just one signal. It is assumed
    # that all files have the same duration in MS-DNS dataset.
    

    with open(savepath, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for (i, (lang, clean_fp, noise_fp, noisy_fp),) in enumerate(
            tqdm(
                zip(language, clean_fullpaths, noise_fullpaths, noisy_fullpaths)
            )
        ):
            signal = read_audio(noisy_fp)
            duration = signal.shape[0] / fs

            row = {
                "ID": i,
                "language": lang,
                "duration": duration,
                "clean_wav": clean_fp,
                "clean_wav_format": "wav",
                "clean_wav_opts": None,
                "noise_wav": noise_fp,
                "noise_wav_format": "wav",
                "noise_wav_opts": None,
                "noisy_wav": noisy_fp,
                "noisy_wav_format": "wav",
                "noisy_wav_opts": None,
            }
            writer.writerow(row)

    # Final prints
    msg = "%s successfully created!" % (savepath)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(clean_fullpaths)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (
        str(round(duration * len(clean_fullpaths) / 3600, 2))
    )
    logger.info(msg)

def skip(save_csv):
    """
    Detects if the SAR data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if os.path.isfile(save_csv):
        skip = True

    return skip
