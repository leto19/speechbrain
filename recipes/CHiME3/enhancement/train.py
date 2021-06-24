#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with spectral masking.

To run this recipe, do the following:
> python train.py train.yaml --data_folder /path/to/save/CHiME3

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically
prepare the CHiME3 dataset for computation,
saving the json to the directory specified in the YAML.

It will automatically perform the basic GCC-PHAT DelaySum beamforming
on the multichannel data.

Authors
 * Szu-Wei Fu 2020
 * Chien-Feng Liao 2020
 * Peter Plantinga 2021
 * George Close 2021 (conversion to CHiME3)
"""
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from chime3_prepare import prepare_chime3
from speechbrain.dataio.dataio import read_audio_multichannel
from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import GccPhat
from speechbrain.processing.multi_mic import DelaySum
import pysepm
import logging

logger = logging.getLogger(__name__)


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Apply masking to convert from noisy waveforms to enhanced signals.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            A dictionary with keys {"spec", "wav"} with predicted features.
        """

        # We first move the batch to the appropriate device, and
        # compute the features necessary for masking.
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.noisy_sig
        noisy_feats = self.compute_feats(noisy_wavs)

        # Masking is done here with the "signal approximation (SA)" algorithm.
        # The masked input is compared directly with clean speech targets.
        mask = self.modules.model(noisy_feats)
        predict_spec = torch.mul(mask, noisy_feats)

        # Also return predicted wav, for evaluation. Note that this could
        # also be used for a time-domain loss term.
        predict_wav = self.hparams.resynth(
            torch.expm1(predict_spec), noisy_wavs
        )

        # Return a dictionary so we don't have to remember the order
        return {"spec": predict_spec, "wav": predict_wav}

    def compute_feats(self, wavs):
        """Returns corresponding log-spectral features of the input waveforms.

        Arguments
        ---------
        wavs : torch.Tensor
            The batch of waveforms to convert to log-spectral features.
        """

        # Log-spectral features

        feats = self.hparams.compute_STFT(wavs)
        feats = sb.processing.features.spectral_magnitude(feats, power=0.5)

        # Log1p reduces the emphasis on small differences
        feats = torch.log1p(feats)

        return feats

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        # Prepare clean targets for comparison
        clean_wavs, lens = batch.clean_sig
        clean_spec = self.compute_feats(clean_wavs)

        # Directly compare the masked spectrograms with the clean targets
        loss = sb.nnet.losses.mse_loss(predictions["spec"], clean_spec, lens)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions["spec"], clean_spec, lens, reduction="batch"
        )

        # Some evaluations are slower, and we only want to perform them
        # on the validation set.
        if stage != sb.Stage.TRAIN:

            # sb.dataio.dataio.write_audio("out.wav",predictions["wav"][0].cpu(),hparams["sample_rate"])
            # Evaluate speech intelligibility as an additional metric
            self.stoi_metric.append(
                batch.id,
                predictions["wav"],
                clean_wavs,
                lens,
                reduction="batch",
            )

            self.snr_metric.append(
                batch.id,
                clean_wavs,
                predictions["wav"],
                lens,
                reduction="batch",
                fs=hparams["sample_rate"],
            )

            self.pesq_metric.append(
                batch.id,
                clean_wavs,
                predictions["wav"],
                lens,
                reduction="batch",
                fs=hparams["sample_rate"],
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.mse_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.stoi_metric = sb.utils.metric_stats.MetricStats(
                metric=sb.nnet.loss.stoi_loss.stoi_loss
            )
            # we set up a SNR metric using our `SNR_calc` defined below.
            self.snr_metric = sb.utils.metric_stats.MetricStats(metric=SNR_calc)
            self.pesq_metric = sb.utils.metric_stats.MetricStats(
                metric=PESQ_calc
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "stoi": -self.stoi_metric.summarize("average"),
                "snr": -self.snr_metric.summarize("average"),
                "pesq": self.pesq_metric.summarize("average"),
            }

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best STOI score.
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["stoi"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def beamformer(x, fs):
    """computes the delay sum beamformer output of an input multichannel signal

    Arguments
    ---------
    x :
        multichannel input audio signal
    fs : int
        sample rate of signal, used for Fourier transforms
    """
    stft = STFT(sample_rate=fs)
    cov = Covariance()
    gccplat = GccPhat()
    delaysum = DelaySum()
    istft = ISTFT(sample_rate=fs)

    X = stft(x)  # take the fourier transform
    # print(X)
    XX = cov(X)  # calculate the covarience matrix
    # print(XX)
    tdoas = gccplat(XX)  # delay estimation
    # print(tdoas)
    Y = delaysum(X, tdoas)  # delay sum beamformer
    y = istft(Y)  # inverse fourier transform
    return y


def SNR_calc(y_pred_batch, y_true_batch, lens, fs=16000, reduction="mean"):
    """
    Calculates the segmental SNR over the input batch using the `SNRseg` function
    from pysepm.
    """
    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    batch_size = y_pred_batch.shape[0]

    D = torch.zeros(batch_size)

    for i in range(0, batch_size):  # for each true/pred pair in the batch
        y_true = y_true_batch[i, 0 : int(lens[i] * y_pred_batch.shape[1])]
        y_pred = y_pred_batch[i, 0 : int(lens[i] * y_pred_batch.shape[1])]

        # sometimes the prediction can be a few frames longer, so we truncate them
        limit = min(len(y_pred), len(y_true))

        D[i] = pysepm.SNRseg(
            y_true.cpu().numpy()[:limit], y_pred.cpu().numpy()[:limit], fs
        )
    if reduction == "mean":
        return D.mean()
    return D


def PESQ_calc(y_pred_batch, y_true_batch, lens, fs=16000, reduction="mean"):
    """
    Calculates the PESQ score over the input batch using the `pesq` function
    from pysepm.
    """
    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    batch_size = y_pred_batch.shape[0]

    D = torch.zeros(batch_size)

    for i in range(0, batch_size):  # for each true/pred pair in the batch
        y_true = y_true_batch[i, 0 : int(lens[i] * y_pred_batch.shape[1])]
        y_pred = y_pred_batch[i, 0 : int(lens[i] * y_pred_batch.shape[1])]

        # sometimes the prediction can be a few frames longer, so we truncate them
        limit = min(len(y_pred), len(y_true))

        D[i] = pysepm.pesq(
            y_true.cpu().numpy()[:limit], y_pred.cpu().numpy()[:limit], fs
        )[1]
    if reduction == "mean":
        return D.mean()
    return D


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    We expect `prepare_chime3` to have been called before this,
    so that the `train.json` and `valid.json` manifest files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    def get_audio(wav_list):
        audio = read_audio_multichannel({"files": wav_list})
        if (
            audio.shape[1] > 1
        ):  # if the data is multi channel, perform the basic beamforming.
            # logger.info(audio.shape)

            # we don't want to beamform using the backward facing channel, so we remove it
            # TODO: this is pretty ugly, but there dosn't seem to be an easier way of removing
            # a colunm from a tensor by it's index?
            audio = audio.T
            audio = audio[
                torch.arange(audio.size(0)) != 1
            ]  # remove the backward facing channel
            audio = audio.T

            # logger.info(audio.shape)
            audio = beamformer(audio.unsqueeze(0), hparams["sample_rate"])
            # logger.info(audio.shape)

            # write_audio("out.wav",audio.squeeze(),16000)
            # input()
        return audio.flatten()

    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in hparams["sets_dict"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams["json_dir"] + "/" + dataset + ".json",
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[
                # you can also do this using the @overrides
                {
                    "func": lambda l: get_audio(l),
                    "takes": ["wavs_far"],
                    "provides": "noisy_sig",
                },
                {
                    "func": lambda l: get_audio(l),
                    "takes": ["wav_reference"],
                    "provides": "clean_sig",
                },
            ],
            output_keys=["id", "noisy_sig", "clean_sig"],
        ).filtered_sorted(sort_key="duration")

    return datasets


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_chime3,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["json_dir"],
            "sets_dict": hparams["sets_dict"],
            "skip": hparams["skip_prep"],
        },
    )

    # Create dataset objects

    datasets = dataio_prep(hparams)
    # Initialize the Brain object to prepare for mask training.
    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.

    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["val"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = se_brain.evaluate(
        test_set=datasets["test"],
        max_key="stoi",
        test_loader_kwargs=hparams["dataloader_options"],
    )
