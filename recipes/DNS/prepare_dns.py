"""
Author
 * Sangeet Sagar 2022

Download instructions: https://github.com/microsoft/DNS-Challenge
The CSV files preperation for Deep Noise
Suppression (DNS) Challenge 4 - ICASSP 2022
"""

import os
import csv
import glob
import torch
import logging
from tqdm import tqdm
from speechbrain.dataio.dataio import read_audio
import pdb

logger = logging.getLogger(__name__)


def prepare_dns_csv(
    datapath,
    baseline_noisy_datapath,
    baseline_enhanced_datapath,
    savepath,
    skip_prep=False,
    fs=16000,
):

    if skip_prep:
        return

    # Create training data
    msg = "Preparing DNS data as csv files in %s " % (savepath)
    logger.info(msg)
    create_dns_csv(datapath, savepath, fs)

    # Create dev data
    msg = "Preparing DNS baseline dev data as csv file in %s " % (savepath)
    logger.info(msg)
    create_baseline_dev_csv(
        baseline_noisy_datapath, baseline_enhanced_datapath, savepath
    )


def create_dns_csv(datapath, savepath, fs=16000):
    """
    This is the DNS dev set, but will be used as test set.
    Prepares the csv files for wham or whamr dataset

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        skip_prep (bool): If True, skip data preparation
        fs (int): sampling frequency
    """

    if not (fs == 8000 or fs == 16000):
        raise ValueError("Unsupported sampling rate")

    set_types = [
        ["read_speech", "en"],
        ["german_speech", "de"],
        ["french_speech", "fr"],
        ["italian_speech", "it"],
        ["spanish_speech", "es"],
        ["russian_speech", "ru"],
    ]
    clean_tr_fullpaths = []
    noise_tr_fullpaths = []
    noisy_tr_fullpaths = []
    clean_dev_fullpaths = []
    noise_dev_fullpaths = []
    noisy_dev_fullpaths = []
    language_tr = []
    language_dev = []

    # If csv already exists, we skip the data preparation
    train_csv = os.path.join(savepath, "dns_16k_tr" + ".csv")
    valid_csv = os.path.join(savepath, "dns_16k_cv" + ".csv")
    if skip(train_csv, valid_csv):

        msg = "%s already exists, skipping data preparation!" % (train_csv)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (valid_csv)
        logger.info(msg)

        return

    for set_type in set_types:

        set_datapath = os.path.join(datapath, set_type[0])
        clean_f1_path = extract_files(set_datapath, type="clean")
        noise_f1_path = extract_files(set_datapath, type="noise")
        noisy_f1_path = extract_files(set_datapath, type="noisy")

        # Split 1% of data from each type to form dev set.
        (
            clean_tr_path,
            clean_dev_path,
            noise_tr_path,
            noise_dev_path,
            noisy_tr_path,
            noisy_dev_path,
        ) = train_dev_split(clean_f1_path, noise_f1_path, noisy_f1_path)

        language_tr.extend([set_type[1]] * len(clean_tr_path))
        clean_tr_fullpaths.extend(clean_tr_path)
        noise_tr_fullpaths.extend(noise_tr_path)
        noisy_tr_fullpaths.extend(noisy_tr_path)

        language_dev.extend([set_type[1]] * len(clean_dev_path))
        clean_dev_fullpaths.extend(clean_dev_path)
        noise_dev_fullpaths.extend(noise_dev_path)
        noisy_dev_fullpaths.extend(noisy_dev_path)

    # Write CSV for train and dev
    msg = "Writing train csv files"
    logger.info(msg)
    write2csv(
        language_tr,
        clean_tr_fullpaths,
        noise_tr_fullpaths,
        noisy_tr_fullpaths,
        train_csv,
        fs=16000,
    )

    msg = "Writing valid csv files"
    logger.info(msg)
    write2csv(
        language_dev,
        clean_dev_fullpaths,
        noise_dev_fullpaths,
        noisy_dev_fullpaths,
        valid_csv,
        fs=16000,
    )


def create_baseline_dev_csv(noisy_datapath, enhanced_datapath, savepath):
    """
    Prepares the csv files for wham or whamr dataset

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        skip_prep (bool): If True, skip data preparation
        fs (int): sampling frequency
    """
    savename = "dns_baseline_dev_48K"

    save_csv = os.path.join(savepath, savename + ".csv")
    # If csv already exists, we skip the data preparation
    if os.path.isfile(save_csv):

        msg = "%s already exists, skipping data preparation!" % (save_csv)
        logger.info(msg)

        return

    noisy_fullpaths = extract_files(noisy_datapath)
    enhanced_fullpath = extract_files(enhanced_datapath)

    csv_columns = [
        "ID",
        "duration",
        "noisy_wav",
        "noisy_wav_format",
        "noisy_wav_opts",
        "enhanced_wav",
        "enhanced_wav_format",
        "enhanced_wav_opts",
    ]

    with open(os.path.join(savepath, savename + ".csv"), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for (i, (noisy_fp, enhanced_fp),) in enumerate(
            zip(noisy_fullpaths, enhanced_fullpath)
        ):

            row = {
                "noisy_wav": noisy_fp,
                "noisy_wav_format": "wav",
                "noisy_wav_opts": None,
                "enhanced_wav": enhanced_fp,
                "enhanced_wav_format": "wav",
                "enhanced_wav_opts": None,
            }
            writer.writerow(row)

    msg = "%s successfully created!" % (savepath + "/" + savename + ".csv")
    logger.info(msg)


def extract_files(datapath, type=None):
    """
    Prepares the csv files for wham or whamr dataset

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        skip_prep (bool): If True, skip data preparation
        fs (int): sampling frequency
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


def train_dev_split(clean_data, noise_data, noisy_data, split_size=0.005):
    # We dont shuffle or perform raondom split, since the dataset was generated
    # randomly in the first place.
    dataset = list(zip(clean_data, noise_data, noisy_data))
    # import random
    # random.shuffle(dataset)
    k = int(len(dataset) * (1 - split_size))
    clean = dataset[:k]
    dev = dataset[k:]

    clean_tr_path, noise_tr_path, noisy_tr_path = map(list, zip(*clean))
    clean_dev_path, noise_dev_path, noisy_dev_path = map(list, zip(*dev))

    return (
        clean_tr_path,
        clean_dev_path,
        noise_tr_path,
        noise_dev_path,
        noisy_tr_path,
        noisy_dev_path,
    )


def write2csv(
    language,
    clean_fullpaths,
    noise_fullpaths,
    noisy_fullpaths,
    savepath,
    fs=16000,
):

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
    signal = read_audio(clean_fullpaths[0])
    duration = signal.shape[0] / fs

    with open(savepath, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for (i, (lang, clean_fp, noise_fp, noisy_fp),) in enumerate(
            tqdm(
                zip(language, clean_fullpaths, noise_fullpaths, noisy_fullpaths)
            )
        ):

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


def skip(save_csv_train, save_csv_dev=None):
    """
    Detects if the Common Voice data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if os.path.isfile(save_csv_train) and os.path.isfile(save_csv_dev):
        skip = True

    return skip
