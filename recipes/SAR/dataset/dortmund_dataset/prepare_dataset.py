"""
Authors
 * Sangeet Sagar 2022
"""

import os
import re
import csv
import time
import string
import argparse
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
import xml.etree.ElementTree as ET
import random
from typing import List
import pdb
from pprint import pprint
import pandas as pd
random.seed(8200)

class Dortmund(object):
    def __init__(self, args):
        self.src_dir = args.src_dir
        self.transciption_dir = args.transciption_dir
    
    def collect_audio_files(self):
        self.audio_files = []
        for path in sorted(Path(self.src_dir).rglob('*.wav')):
            self.audio_files.append(str(path))
    
    def collect_transcriptions(self):
        self.csv_files = []
        for path in sorted(Path(self.transciption_dir).rglob('*.csv')):
            self.csv_files.append(str(path))
    
    def load_transcriptions(self):
        meta = []
        
        for csv_file in self.csv_files:
            try:
                data = pd.read_csv(csv_file)
            except Exception:
                data = pd.read_csv(csv_file, sep=";", dtype='object')
            
            files = data["audio_file"].values.tolist()
            seg_ids = data["seg_id"].values.tolist()
            start_ts = data["start"].values.tolist()
            end_ts = data["end"].values.tolist()
            radio_comms = data["is_radio_communication"].values.tolist()
            texts = data["transcription"].values.tolist()
            spkr_info = data["speaker"].values.tolist()
            
            for file, seg_id, start, end, text, spkr in zip(files,seg_ids,start_ts,end_ts,texts, spkr_info):
                
                start = int(str(start).replace('.', ''))
                end = int(str(end).replace('.', ''))
                start = "{:.3f}".format(start/1000)
                end = "{:.3f}".format(end/1000)
                if "_" in seg_id:
                    # multi-speaker segment
                    seg_id, sub_seg_id = seg_id.split("_")
                    seg_id = int(seg_id)
                    seg_id = str("{0:04}".format(seg_id) + "_" + sub_seg_id)
                else:
                    seg_id = int(seg_id)
                    seg_id = str("{0:04}".format(seg_id))
                
                f_name = file + "-seg." + seg_id + "-" + str(start) + "-" + str(end) + ".wav"

                norm_text = self._text_normalization([text]).strip()
                if os.path.join(self.src_dir, f_name) not in self.audio_files:
                    print("File not found:", f_name, norm_text)
                
                if len(norm_text.split()) > 1:
                    meta.append([
                    spkr,
                    f_name,
                    norm_text])
        
        shuffeled_data = meta.copy()
        random.shuffle(shuffeled_data)
        train_data, test_data = self.train_test_split_data(shuffeled_data)
        train_data, dev_data = self.train_test_split_data(train_data)
        final_data = [shuffeled_data, train_data, dev_data, test_data]
        split_list = ['dortmund_dataset.tsv','train.tsv', 'dev.tsv', 'test.tsv']
        
        for idx, split in enumerate(split_list):
            csv_lines = [[
                        "client_id",
                        "path", 
                        "sentence"
                        ]]
            csv_lines.extend(final_data[idx])
            # Writing the csv lines
            with open(split, mode="w", encoding="utf-8") as csv_f:
                csv_writer = csv.writer(
                    csv_f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )

                for line in csv_lines:
                    csv_writer.writerow(line)
                
        
    def _text_normalization(self, text):
            norm_text = []
            char = '[<'
            for line in text:
                line = " ".join( filter( lambda word: not word.startswith(char), line.split() ) )
                line = line.translate(str.maketrans('', '', string.punctuation))
                line = line.replace('–',' ').replace('„',' ').replace('“',' ').replace('-',' ')
                
                line = " ".join(filter(lambda x:x[:1]!='[<', line.split()))
                line = re.sub(r'[0-9]+', '', line)
                line = re.sub("[^’'A-Za-z0-9öÖäÄüÜß]+", " ", line)

                line = line.replace("ß","0000ß0000").upper()
                line = line.replace("0000SS0000", "ß")
                line = line.replace("UNK", "")
                line = line.replace("’", "")
                norm_text.append(line)
                
            return "".join(norm_text)
        
    def train_test_split_data(self, text: List[str], test_size=0.2):
        """ Splits the input corpus in a train and a test set
        :param text: input corpus
        :param test_size: size of the test set, in fractions of the original corpus
        :return: train and test set
        """
        k = int(len(text) * (1 - test_size))
        return text[:k], text[k:]


def main():
    """ main method """
    args = parse_arguments()
    net_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    dortmund = Dortmund(args)
    dortmund.collect_audio_files()
    dortmund.collect_transcriptions()
    dortmund.load_transcriptions()
    # dortmund.segment_audio()
    
def parse_arguments():
    """ parse arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir", help="path to audio dir with all wav files")
    parser.add_argument("transciption_dir", help="path to csv transcript files")
    parser.add_argument("out_dir", help='output dir to save the tsv dataset')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()