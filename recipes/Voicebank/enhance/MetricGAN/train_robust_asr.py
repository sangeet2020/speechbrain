from doctest import OutputChecker
import os
from re import L
import sys
import shutil
import pickle
import torch
import torchaudio
import speechbrain as sb
from pesq import pesq
from enum import Enum, auto
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler
import pdb
from pprint import pprint


import os
import sys
import torch
import torchaudio
import speechbrain as sb
from pesq import pesq
from pystoi import stoi
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main


class ASR_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        """_summary_
        """
        
        batch = batch.to(self.device)
        self.stage= stage
        
        # Enhancement pipeline begins--------------------
        predictions = {}
        noisy_wavs, lens = self.prepare_wavs(batch.noisy_sig)
        noisy_spec = self.prepare_feats(noisy_wavs)  # Shape: [2, 69, 257]

        # mask with "signal approximation (SA)"
        mask = self.modules.enhance_model(noisy_spec, lengths=lens)
        mask = mask.clamp(min=self.hparams.min_mask).squeeze(2)
        predict_spec = torch.mul(mask, noisy_spec)  # Shape: [2, 69, 257]

        # # Also return predicted wav
        predictions["wavs"] = self.hparams.resynth(     # Shape: [2, 18352]
            torch.expm1(predict_spec), noisy_wavs
        )   
        
        # Write enhanced wavs for sanity check
        self.write_wavs(batch.id, predictions["wavs"], lens)
        
        # ASR pipeline begins--------------------
        # Prepare target inputs
        asr_feats = predictions["wavs"]
        if stage == sb.Stage.TRAIN:
            asr_feats = self.hparams.augment(asr_feats, lens)
        asr_feats = self.hparams.fbank(asr_feats)
        asr_feats = self.hparams.normalizer(asr_feats, lens)
        embed = self.modules.src_embedding(asr_feats)
        
        tokens, token_lens = self.prepare_targets(batch.tokens_bos)
        tokens = self.modules.tgt_embedding(tokens)
        
        dec_out = self.modules.recognizer(tokens, embed, lens)
        out = self.modules.seq_output(dec_out[0])
        predictions["seq_pout"] = torch.log_softmax(out, dim=-1)

        if self.hparams.ctc_type is not None:
            out = self.modules.ctc_output(embed)
            predictions["ctc_pout"] = torch.log_softmax(out, dim=-1)

        if stage != sb.Stage.TRAIN:
            predictions["hyps"], _ = self.hparams.beam_searcher(
                embed.detach(), lens
            )
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        
        # Do not augment targets
        clean_wavs, lens = self.prepare_wavs(batch.clean_sig, augment=False)
        loss = 0
        
        # Compute nll loss for seq2seq model
        tokens, token_lens = self.prepare_targets(batch.tokens_eos)
        seq_loss = self.hparams.seq_loss(
            predictions["seq_pout"], tokens, token_lens
        )
        loss +=  seq_loss

        if stage != sb.Stage.TRAIN:
            if hasattr(self.hparams, "tokenizer"):
                
                pred_words = [
                    self.token_encoder.decode_ids(token_seq)
                    for token_seq in predictions["hyps"]
                ]
                target_words = [
                    self.token_encoder.decode_ids(token_seq)
                    for token_seq in undo_padding(*batch.tokens)
                ]
                self.err_rate_metrics.append(
                    batch.id, pred_words, target_words
                )

            else:
                self.err_rate_metrics.append(
                    ids=batch.id,
                    predict=predictions["hyps"],
                    target=tokens,
                    target_len=token_lens,
                    ind2lab=self.token_encoder.decode_ndim,
                )

        return loss

    def write_wavs(self, batch_id, wavs, lens):
        """Write wavs to files

        Arguments
        ---------
        batch_id : list of str
            A list of the utterance ids for the batch
        scores : torch.Tensor
            The actual scores for the corresponding utterances
        lens : torch.Tensor
            The relative lengths of each utterance
        """
        lens = lens * wavs.shape[1]
        for i, (name, pred_wav, length) in enumerate(
            zip(batch_id, wavs, lens)
        ):  
            path = os.path.join(self.hparams.MetricGAN_folder, name + ".wav")
            data = torch.unsqueeze(pred_wav[: int(length)].cpu(), 0)
            torchaudio.save(path, data, self.hparams.sample_rate)
            
    def prepare_feats(self, wavs):
        """Prepare log-magnitude spectral features expected by perceptual model"""
        stft = self.hparams.compute_stft(wavs)
        feats = self.hparams.spectral_magnitude(stft)
        feats = torch.log1p(feats)
        return feats
        
    def prepare_wavs(self, signal, augment=True):
        """Prepare possibly enhanced waveforms"""
        wavs, wav_lens = signal

        if self.stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corr"):  # True for asr
            if augment:
                wavs_noise = self.hparams.env_corr(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
            else:
                wavs = torch.cat([wavs, wavs], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        return wavs, wav_lens
    
    def prepare_targets(self, tokens):
        """Prepare target by concatenating self if "env_corr" is used"""
        tokens, token_lens = tokens

        if self.stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corr"):
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens])

        return tokens, token_lens
    
    def on_stage_start(self, stage, epoch):
        # 
        if stage != sb.Stage.TRAIN:
            self.err_rate_metrics = self.hparams.err_rate_stats()
            
        # Freeze models before training
        else:
            for model in self.hparams.frozen_models:
                if (
                    hasattr(self.hparams, "unfreeze_epoch")
                    and epoch >= self.hparams.unfreeze_epoch
                    and (
                        not hasattr(self.hparams, "unfrozen_models")
                        or model in self.hparams.unfrozen_models
                    )
                ):
                    self.modules[model].train()
                    for p in self.modules[model].parameters():
                        p.requires_grad = True  # Model's weight will be updated
                else:
                    self.modules[model].eval()
                    for p in self.modules[model].parameters():
                        p.requires_grad = False     # Model is frozen

    
    def on_stage_end(self, stage, stage_loss, epoch, tr_time):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stage_stats = {"loss": stage_loss}
            max_keys = []
            min_keys = []

            err_rate = self.err_rate_metrics.summarize("error_rate")
            err_rate_type = self.hparams.target_type + "ER"
            stage_stats[err_rate_type] = err_rate
            min_keys.append(err_rate_type)
        
        if stage == sb.Stage.VALID:
            stats_meta = {"epoch": epoch}
            if hasattr(self.hparams, "lr_annealing"):
                old_lr, new_lr = self.hparams.lr_annealing(epoch - 1)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
                stats_meta["lr"] = old_lr

            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta,
                train_stats={"tr_time": tr_time, "loss": self.train_loss},
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                max_keys=max_keys,
                min_keys=min_keys,
                num_to_keep=self.hparams.checkpoint_avg,
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.stats_file + ".txt", "w") as w:
                if self.hparams.seq_weight > 0:
                    self.err_rate_metrics.write_stats(w)
                print("stats written to ", self.hparams.stats_file)

    def on_evaluate_start(self, max_key=None, min_key=None):
        self.checkpointer.recover_if_possible(max_key=max_key, min_key=min_key)
        checkpoints = self.checkpointer.find_checkpoints(
            max_key=max_key,
            min_key=min_key,
            max_num_checkpoints=self.hparams.checkpoint_avg,
        )
        for model in self.modules:
            if (
                model not in self.hparams.frozen_models
                or hasattr(self.hparams, "unfrozen_models")
                and model in self.hparams.unfrozen_models
            ):
                model_state_dict = sb.utils.checkpoints.average_checkpoints(
                    checkpoints, model
                )
                self.modules[model].load_state_dict(model_state_dict)
        
def dataio_prep(hparams, token_encoder):
        
    # 1. Define audio pipeline
    @sb.utils.data_pipeline.takes(hparams["input_type"], "clean_wav")
    @sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
    def audio_pipeline(noisy_wav, clean_wav):
        yield sb.dataio.dataio.read_audio(noisy_wav)
        yield sb.dataio.dataio.read_audio(clean_wav)
    
    token_keys = ["wrd", "tokens_bos", "tokens_eos", "tokens"]
    # 2. Define text pipeline
    @sb.utils.data_pipeline.takes(hparams["target_type"])
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def target_pipeline(wrd):
        yield wrd
        tokens_list = token_encoder.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    hparams["train_logger"].log_stats(
        stats_meta={
            "Training on input type: ": hparams["input_type"]
        }
    )

    # Create datasets
    data = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        data[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, target_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig"] + token_keys
        )
        if dataset != "train":
            data[dataset] = data[dataset].filtered_sorted(sort_key="length")
    
    # Sort train dataset and ensure it doesn't get un-sorted
    if hparams["sorting"] in ["ascending", "descending"]:
        data["train"] = data["train"].filtered_sorted(
            sort_key="length", reverse=hparams["sorting"] == "descending"
        )
        hparams["train_loader_options"]["shuffle"] = False

    return data
    
# Recipe begins!
if __name__ == "__main__":
    
    # Load hyperparameters file
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initalize DDP
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides
    )
    if not os.path.isdir(hparams["MetricGAN_folder"]):
        os.makedirs(hparams["MetricGAN_folder"])
    
    # Prepare data
    from voicebank_prepare import prepare_voicebank
    
    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["json_dir"],
            "skip_prep": hparams["skip_prep"]
        }
    )
    
    # Load pre-trained models
    pretrained = "asr_pretrained"
    if pretrained in hparams:
        run_on_main(hparams[pretrained].collect_files)
        hparams[pretrained].load_collected()
    
    # Switch encoder based on task
    if "tokenizer" in hparams:
        token_encoder = hparams["tokenizer"]
    else:
        token_encoder = sb.dataio.encoder.CTCTextEncoder()
    
    datasets = dataio_prep(hparams, token_encoder)
    
    # Intitialize trainer
    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        run_opts=run_opts,
        hparams=hparams,
        checkpointer=hparams["checkpointer"]
    )
    
    asr_brain.token_encoder = token_encoder
    # Fit dataset
    asr_brain.fit(
        epoch_counter=asr_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_loader_options"],
        valid_loader_kwargs=hparams["valid_loader_options"]
    )
    # Evaluate best performing checkpoint
    outdir = asr_brain.hparams.output_folder
    asr_brain.hparams.stats_file = os.path.join(outdir, "test_stats")
    asr_brain.evaluate(
        datasets["test"],
        max_key=hparams["eval_max_key"],
        min_key=hparams["eval_min_key"],
        test_loader_kwargs=hparams["test_loader_options"],
    )