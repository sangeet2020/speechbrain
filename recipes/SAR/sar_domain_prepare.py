"""
Data preparation for SAR (search and rescue) domain data- 
    - Viersen
    - TRADR
    - 22_09_14_Dortmund_Endevaluation
for ASR fine-tuning (Phase 0 and Phase 1)
Author
------
Titouan Parcollet
Sangeet Sagar
"""

import os
import csv
import re
import logging
import torchaudio
import unicodedata
from tqdm.contrib import tzip
import pdb

logger = logging.getLogger(__name__)


def prepare_SAR(
    data_folder,
    save_folder,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    accented_letters=False,
    language="en",
    skip_prep=False,
    clean_noisy_mix=False,
    noisy_data_folder=None,
):
    """
    Prepares the csv files for SAR domain audio data.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    train_tsv_file : str, optional
        Path to the Train SAR .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev SAR .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test SAR .tsv file (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    language: str
        Specify the language for text normalization.
    skip_prep: bool
        If True, skip data preparation.

    """

    if skip_prep:
        return

    # If not specified point toward standard location
    if train_tsv_file is None:
        train_tsv_file = data_folder + "/train.tsv"
    else:
        train_tsv_file = train_tsv_file

    if dev_tsv_file is None:
        dev_tsv_file = data_folder + "/dev.tsv"
    else:
        dev_tsv_file = dev_tsv_file

    if test_tsv_file is None:
        test_tsv_file = data_folder + "/test.tsv"
    else:
        test_tsv_file = test_tsv_file

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = save_folder + "/train.csv"
    save_csv_dev = save_folder + "/dev.csv"
    save_csv_test = save_folder + "/test.csv"

    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_dev, save_csv_test):
        
        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        logger.info(msg)

        return

    # Additional checks to make sure the data folder contains SAR data.
    check_sar_data_folders(data_folder)

    # Creating csv files for {train, dev, test} data
    file_pairs = zip(
        [train_tsv_file, dev_tsv_file, test_tsv_file],
        [save_csv_train, save_csv_dev, save_csv_test],
    )
    for tsv_file, save_csv in file_pairs:
        if clean_noisy_mix:
            # Prepare CSV files for Phase 1 training
            create_clean_noisy_mix_csv(
                tsv_file, save_csv, data_folder, noisy_data_folder, accented_letters, language
            )
        else:
            # Prepare CSV files for Phase 0 fine-tuning
            create_csv(
                tsv_file, save_csv, data_folder, accented_letters, language,
            )


def skip(save_csv_train, save_csv_dev, save_csv_test):
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

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip



def create_csv(
    orig_tsv_file, csv_file, data_folder, accented_letters=False, language="en"
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the SAR tsv file (standard file).
    data_folder : str
        Path of the SAR domain dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.

    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):

        line = line[0]

        # Path is at indice 1 in SAR tsv files. And .mp3 files
        # are located in datasets/lang/clips/
        mp3_path = data_folder + "/audio_files/" + line.split("\t")[1]
        file_name = ".".join(mp3_path.split(".")).split("/")[-1]
        spk_id = line.split("\t")[0]
        snt_id = file_name

        # Setting torchaudio backend to sox-io (needed to read mp3 files)
        if torchaudio.get_audio_backend() != "sox_io":
            logger.warning("This recipe needs the sox-io backend of torchaudio")
            logger.warning("The torchaudio backend is changed to sox_io")
            torchaudio.set_audio_backend("sox_io")

        # Reading the signal (to retrieve duration in seconds)
        if os.path.isfile(mp3_path):
            info = torchaudio.info(mp3_path)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            continue

        duration = info.num_frames / info.sample_rate
        total_duration += duration

        # Getting transcript
        words = line.split("\t")[2]

        # Unicode Normalization
        words = unicode_normalisation(words)

        # Perform language specific data cleaning
        words = data_cleaning(words, language)
        
        # Remove accents if specified
        if not accented_letters:
            words = strip_accents(words)
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        if len(words.split(" ")) < 3:
            continue

        # Composition of the csv_line
        csv_line = [snt_id, str(duration), mp3_path, spk_id, str(words)]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)

def create_clean_noisy_mix_csv(
    orig_tsv_file, csv_file, data_folder, noisy_data_folder, accented_letters=False, language="en"
):
    """
    Creates the csv file given a list of wav files. 
    (for Phase 1 clean noisy mix fine-tuning.)

    Arguments
    ---------
    orig_tsv_file : str
        Path to the SAR tsv file (standard file).
    data_folder : str
        Path of the SAR domain dataset.
    noisy_data_folder : str
        Path to synthesized noisy SAR domain dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.

    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    # # Load all filenames in the noisy_data_folder/noisy
    # noisy_files = 

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [[
        "ID", 
        "duration", 
        "clean_wav", 
        "noisy_wav", 
        "clean_noisy_mix", 
        "noise_wav", 
        "noise_type", 
        "snr_level", 
        "spk_id", 
        "wrd"
        ]]

    # Noise types
    noise_types = [
        "Breathing-noise",
        "Emergency-vehicle-and-siren-noise",
        "Engine-noise",
        "Chopper-noise",
        "Static-radio-noise",
    ]
    
    multiple_files_count = 0
    no_matching_files_count = 0
    idx = 0
    
    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):
        
        line = line[0]

        # Path is at indice 1 in SAR tsv files. And .mp3 files
        # are located in datasets/lang/clips/
        mp3_path = data_folder + "/audio_files/" + line.split("\t")[1]
        file_name = ".".join(mp3_path.split(".")).split("/")[-1]
        spk_id = line.split("\t")[0]
        snt_id = file_name

        # Retrieive the corresponding noisy file from noisy_data_folder
        clean_wav_bname = os.path.splitext(file_name)[0] + "_"
        noisy_file = [filename for filename in os.listdir(os.path.join(noisy_data_folder, "noisy")) if filename.startswith(clean_wav_bname)]
        if len(noisy_file) >= 1:
            idx += 1
            if len(noisy_file) > 1:
                multiple_files_count += 1

        if len(noisy_file) == 0:
            no_matching_files_count += 1
            continue
        
        noisy_file = noisy_file[0]
        noisy_file_path = os.path.join(noisy_data_folder, "noisy", noisy_file)
        
        if (idx%2==0):
            clean_noisy_mix = mp3_path
        else:
            clean_noisy_mix = noisy_file_path
        
        # Get corresponding noise file
        fields = os.path.splitext(noisy_file)[0].split("_")
        fileid = fields[fields.index("fileid")+1]
        noise_file = "noise_fileid_" + str(fileid) + ".wav"
        noise_file_path = os.path.join(noisy_data_folder, "noise", noise_file)
        
        # Get noise type
        for item in noise_types:
            if item in noisy_file:
                noise_type = item
                break
                
        # Get SNR level
        for item in fields:
            if "snr" in item:
                snr_level = item.replace("snr","")
                break

        # Setting torchaudio backend to sox-io (needed to read mp3 files)
        if torchaudio.get_audio_backend() != "sox_io":
            logger.warning("This recipe needs the sox-io backend of torchaudio")
            logger.warning("The torchaudio backend is changed to sox_io")
            torchaudio.set_audio_backend("sox_io")

        # Reading the signal (to retrieve duration in seconds)
        if os.path.isfile(mp3_path):
            info = torchaudio.info(mp3_path)
            info_noisy = torchaudio.info(noisy_file_path)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            idx += 1
            continue

        duration = info.num_frames / info.sample_rate
        
        # Do some sanity check duration of clean, and noisy must be same
        duration_noisy = info_noisy.num_frames / info_noisy.sample_rate
        if round(duration, 3) != round(duration_noisy, 3):
            print("Length mismatch detected")
            
        total_duration += duration

        # Getting transcript
        words = line.split("\t")[2]

        # Unicode Normalization
        words = unicode_normalisation(words)

        # Perform language specific data cleaning
        words = data_cleaning(words, language)
        
        # Remove accents if specified
        if not accented_letters:
            words = strip_accents(words)
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        if len(words.split(" ")) < 3:
            idx += 1
            continue

        # Composition of the csv_line
        csv_line = [
            snt_id, 
            str(duration), 
            mp3_path, 
            noisy_file_path, 
            clean_noisy_mix,
            noise_file_path,
            noise_type, 
            str(snr_level), 
            spk_id, 
            str(words)
            ]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)
    msg = "Number of files with multiple matches: %s " % (str(multiple_files_count))
    logger.info(msg)
    msg = "Number of files with no matches: %s " % (str(no_matching_files_count))
    logger.info(msg)

def check_sar_data_folders(data_folder):
    """
    Check if the data folder actually contains the SAR dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain SAR dataset.
    """

    files_str = "/audio_files"

    # Checking clips
    if not os.path.exists(data_folder + files_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the SAR dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)



def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)

def data_cleaning(words, language):
    """
    Perform language specific data cleaning

    Arguments
    ---------
        word : str
            Text that needs to be cleaned
        language : str
            Language of the text. e.g. en, fr, de, it
    
    Returns
    -------
    str
        Cleaned data
    
    """
    # !! Language specific cleaning !!
    # Important: feel free to specify the text normalization
    # corresponding to your alphabet.
    if language in ["en", "fr", "it", "rw"]:
        words = re.sub(
            "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
        ).upper()

    if language == "de":
        words = words.replace("ß","0000ß0000")
        words = re.sub("[^’'A-Za-z0-9öÖäÄüÜß]+", " ", words).upper()
        words = words.replace("'", " ")
        words = words.replace("’", " ")
        words = words.replace("0000SS0000", "ß")
    
    if language == "fr":
        # Replace J'y D'hui etc by J_ D_hui
        words = words.replace("'", " ")
        words = words.replace("’", " ")

    elif language == "ar":
        HAMZA = "\u0621"
        ALEF_MADDA = "\u0622" 
        ALEF_HAMZA_ABOVE = "\u0623"
        letters = (
            "ابتةثجحخدذرزسشصضطظعغفقكلمنهويىءآأؤإئ"
            + HAMZA
            + ALEF_MADDA
            + ALEF_HAMZA_ABOVE
        )
        words = re.sub("[^" + letters + " ]+", "", words).upper()
    elif language == "ga-IE":
        # Irish lower() is complicated, but upper() is nondeterministic, so use lowercase
        def pfxuc(a):
            return len(a) >= 2 and a[0] in "tn" and a[1] in "AEIOUÁÉÍÓÚ"

        def galc(w):
            return w.lower() if not pfxuc(w) else w[0] + "-" + w[1:].lower()

        words = re.sub("[^-A-Za-z'ÁÉÍÓÚáéíóú]+", " ", words)
        words = " ".join(map(galc, words.split(" ")))
        
    return words


def strip_accents(text):

    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)
