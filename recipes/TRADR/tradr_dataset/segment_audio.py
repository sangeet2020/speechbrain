#!/usr/bin/env python3
"""Recipe for segmenting TRADR audio using trs files that contains speaker wise
start and end time-stamps.


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
random.seed(8200)

class TRADR(object):
    def __init__(self, args):
        self.src_dir = args.src_dir
        self.out_dir = args.out_dir
    
    def collect_audio_files(self):
        self.audio_files = []
        for path in sorted(Path(self.src_dir).rglob('*.mp3')):
            if 'html' not in str(path):
                self.audio_files.append(path)
        
    def collect_trs_files(self):
        self.trs_files = []
        for path in sorted(Path(self.src_dir).rglob('*.trs')):
            if 'html' not in str(path):
                self.trs_files.append(path)
    
    def collect_stm_files(self):
        self.stm_files = []
        for path in sorted(Path(self.src_dir).rglob('*.stm')):
            if 'html' not in str(path):
                self.stm_files.append(path)

    def load_stm_files(self):
        assert len(self.audio_files) == len(self.trs_files), 'Different number of audio and trs files'
        
        spkr_name_id_map = {
            "Teamleader1"  : "spk1",
            "UGV1"         : "spk2",
            "UAV"          : "spk3",
            "UGV2"         : "spk5",
            "Teamleader2"  :  "spk6",

        }
        
        self.meta = []
        for audio_file, stm_file in zip(self.audio_files, self.stm_files):
            count = 0
            f_name = "_".join(os.path.splitext(audio_file)[0].split("/")[1:])
            data = []
            with open(str(stm_file), encoding='latin-1') as fpw:
                for line in fpw:
                    if ";;" not in line:
                        fields = line.split()
                        if len(fields) > 6:
                            count +=1 
                            sync_start = fields[3]
                            sync_end = fields[4]
                            text = " ".join(fields[6:])
                            norm_text = self._text_normalization([text])
                            if fields[2].split("_")[1] in spkr_name_id_map:
                                spkr_id = spkr_name_id_map[fields[2].split("_")[1]]
                                audio_id = f_name + "_" + str(count) + '.wav'
                                # print(audio_id, spkr_id, sync_start, sync_end, norm_text, sep='\t')
                                self.meta.append([
                                str(audio_file),
                                audio_id,
                                spkr_id,
                                str(sync_start), 
                                str(sync_end),
                                norm_text,
                            ])
        self.write_csv()
    
    def load_trs_files(self):
        assert len(self.audio_files) == len(self.trs_files), 'Different number of audio and trs files'
        
        self.meta = []
        for audio_file, trs_file in zip(self.audio_files, self.trs_files):
            
            data = []
            tree = ET.parse(str(trs_file))
            root = tree.getroot()
            
            ####### Extract speaker-id mapping, audio start and end time
            spkr_name_id = {}
            audio_start_end_time = {}
            for child in root:
                if child.tag:
                    for gchild in child:
                        if 'name' in gchild.attrib:
                            spkr_name_id[gchild.attrib['id']] = gchild.attrib['name']
                        if 'startTime' in gchild.attrib:
                            for k,v in gchild.attrib.items():
                                audio_start_end_time[k] = v

            ####### Extract rest of info
            for node in root.findall('.//Turn'):
                # Tag = Turn
                if 'speaker' in node.keys(): # Skip header
                    
                    ##### Extract spkr-id, timestamps ######
                    start_time = node.attrib['startTime']
                    end_time = node.attrib['endTime']
                    spkr_id = node.attrib['speaker']
                    if len(spkr_id.split(" ")) > 1:
                        continue
                    ##### Extract sync-times ######
                    sync_time = []
                    proc_sync_times = []
                    event_desc = []
                    transcript = []
                    
                    for item in node:
                        if item.tag == 'Event':
                            #### process sync times
                            for idx, x in enumerate(sync_time):
                                if not idx == len(sync_time)-1:
                                    proc_sync_times.append([x, sync_time[idx+1]])
                            
                            event_desc.append(item.attrib['desc'])
                            
                            if item.attrib['desc'] == '<EHM>' or item.attrib['desc'] == '<EH>':
                                if not transcript:
                                    transcript.append([item.tail.strip()])
                                else:
                                    transcript[-1].extend([item.tail.strip()])
                            if item.attrib['desc'] == 'unk.skippable':
                                sync_time.clear()
                                
                            
                        elif 'time' in item.attrib:
                            sync_time.append(item.attrib['time'])
                            if item.tail.strip():
                                transcript.append([item.tail.strip()])

                    ### Append sync time for last turn
                    if sync_time:
                        proc_sync_times.append([sync_time[0], end_time])
                    
                    for sync_time, text in zip(proc_sync_times, transcript):
                        norm_text = self._text_normalization(text)
                        # print(trs_file,
                        #     audio_start_end_time['startTime'],
                        #     audio_start_end_time['endTime'],
                        #     spkr_name_id[spkr_id], 
                        #     spkr_id, 
                        #     start_time, 
                        #     end_time, 
                        #     sync_time, 
                        #     ' '.join(text))
                        self.meta.append([
                            str(audio_file),
                            str(trs_file),
                            str(audio_start_end_time['startTime']),
                            str(audio_start_end_time['endTime']),
                            spkr_name_id[spkr_id],
                            spkr_id,
                            str(start_time),
                            str(end_time),
                            str(sync_time[0]), 
                            str(sync_time[1]),
                            norm_text,
                        ])
        
        # Write data to CSV file
        self.write_csv()
    
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
            norm_text.append(line)
            
        return "".join(norm_text)

    def segment_audio(self):
        
        data = []
        curr_audio = []
        for line in tqdm(self.meta):
            audio_path = line[0]
            segment_id = line[1]
            spkr_info = line[2]
            start_time = float(line[3]) * 1000
            end_time = float(line[4]) * 1000
            text = line[-1]
            
            save_path = self.out_dir + "/" + segment_id
            
            data.append([
                spkr_info,
                segment_id,
                text
            ])
            # newAudio = AudioSegment.from_mp3(audio_path)
            # newAudio = newAudio[start_time:end_time]
            # newAudio.export(save_path, format="wav")

        shuffeled_data = data.copy()
        random.shuffle(shuffeled_data)
        train_data, test_data = self.train_test_split_data(shuffeled_data)
        train_data, dev_data = self.train_test_split_data(train_data)
        final_data = [shuffeled_data, train_data, dev_data, test_data]
        split_list = ['tradr_dataset.tsv','train.tsv', 'dev.tsv', 'test.tsv']
        
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

    def write_csv(self):
        csv_file = 'TRADR.csv'
        csv_lines = [[
                    "AUDIOFILE",
                    "SEGMENT_ID",
                    "SPEAKER_ID",
                    "SYNC_START", "SYNC_END", 
                    "TEXT"
                    ]]
        csv_lines.extend(self.meta)
        # Writing the csv lines
        with open(csv_file, mode="w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for line in csv_lines:
                csv_writer.writerow(line)

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
    tradr = TRADR(args)
    tradr.collect_audio_files()
    tradr.collect_trs_files()
    tradr.collect_stm_files()
    tradr.load_stm_files()
    tradr.segment_audio()
    
def parse_arguments():
    """ parse arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir", help="path to source dir")
    # parser.add_argument("foreign_f", help="path to target (foreign) file ")
    parser.add_argument("out_dir", help='output dir to save the segmented audio')
    # parser.add_argument("-epochs", default=10, type=int, help='number of training epochs for EM')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()