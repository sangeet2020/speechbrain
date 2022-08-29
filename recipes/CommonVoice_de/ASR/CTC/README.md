# CommonVoice (DE) ASR fine-tuned on wav2vec2 model
This folder contains the scripts to fine-tune wav2vec2 model `facebook/wav2vec2-large-xlsr-53-german` using CommonVoice dataset.


# How to run
python train_with_wav2vec.py hparams/file.yaml

Make sure you have "transformers" installed in your environment (see extra-requirements.txt)

# Results

| Release | hyperparams file | Test CER | Test WER | HuggingFace link | Full model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :-----:| :--------:|
| 08-16-22 | train_with_wav2vec.yaml | 2.40 | 9.54 | Not Avail. | [Link](https://drive.google.com/drive/u/1/folders/1hag_U5gNT-GOrWEkr_yPbd2RtBP8OCcm) | 1xRTXA6000 48GB |

# Training Time
It takes about 5.5 hours for an epoch on a 1xRTXA6000 48GB.

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
