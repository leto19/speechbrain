"""
Creates json files for the CHiME3 challenge
http://spandh.dcs.shef.ac.uk/chime_challenge/chime2015/index.html
for use with the speechbrain tool
Authors:
 * George Close,2021
"""

import os
import json
import logging
from speechbrain.dataio.dataio import read_audio
import re

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def get_duration(file):
    """ returns the duration of an audio file in seconds"""
    signal = read_audio(file)
    signal = signal.squeeze(0)
    duration = signal.shape[0] / SAMPLERATE
    return duration


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def normalise_transcript(transcript):
    """
    Removes the punctiuation characters + tags from the simulated transcripts.
    """
    regex = re.compile(r"[^A-Z^\s][^\s]+")
    # print(transcript)
    out = re.sub(regex, "", transcript)
    # print(out)
    return out


def prepare_chime3(
    data_folder,
    save_folder="out",
    sets_dict={
        "train": ["tr05_real"],
        "val": ["dt05_real"],
        "test": ["et05_real"],
    },
    skip=False,
):
    """
    Prepares the json files for the CHiME3 dataset.


    Arguments
    ---------
    data_folder : str
        Path to the folder where the CHIME3 dataset is stored.
    save_folder: str
        The folder to save the output json files to, defaults to 'out/' in the current directory
    sets: dict
        Dictionary refering to which `.trn_all` files you want to prepare for and which set that file belongs,
        defaults to `{'train':['tr05_real'],'val':['dt05_real'],'test':['et05_real']}`
    skip: boolean
        If True, skips the data prep step, defaults to false.
    Example
    -------
    >>> data_folder = '/path/to/CHiME3'
    >>> save_folder = 'chime3_json'
    >>> sets_dict = {'train':['tr05_real','tr05_simu'],'val':['dt05_real'],'test':['et05_real']}
    >>> prepare_chime3(data_folder, save_folder,sets_dict)
    """

    if skip:
        logger.info(f"Skipping data prep!")
        return

    transcript_path = (
        data_folder + "/data/transcriptions/"
    )  # location of the .trn_all files
    audio_path = data_folder + "/data/audio/16kHz/isolated"

    if not check_folders(save_folder):
        os.makedirs(save_folder)

    for set in sets_dict:  # for each set (train/val/test)
        logger.info(f"processing set: {set}")

        # set up the json file for this set
        json_save_folder = save_folder
        json_file = os.path.join(json_save_folder, set + ".json")
        if os.path.exists(json_file):
            logger.info("removing old version of %s" % json_file)
            os.remove(json_file)
        json_dict = {}

        for files in sets_dict[set]:  # for each part of the set
            logger.info(f"{files} being processed...")

            # get the transcripion files:
            filenames = [
                name
                for name in os.listdir(transcript_path)
                if (name.endswith("trn_all") and name.split(".")[0] in files)
            ]

            for file in filenames:  #
                logger.info(f"Processing file:{file}")

                dset = file.split("_")[0]  # set type - tr05, dt05 or et05

                dtype = file.split("_")[1]  # data type - real or simu
                with open(os.path.join(transcript_path, file), "r") as f:

                    for line in f.readlines():
                        id_string = line.split(" ")[0]
                        transcript = " ".join(line.split(" ")[1:]).strip(
                            "\n"
                        )  # gets the transcription text

                        speaker = id_string.split("_")[0]  # gets the speaker ID
                        loc = id_string.split("_")[
                            -1
                        ]  # gets the location (BUS,PED,CAF ect)
                        trans_id = id_string.split("_")[
                            1
                        ]  # gets the ID of the transcription

                        # get the path to the folder where the wav files are:
                        # NOTE the format for the wav directory names is slightly different to that of the transcripts,
                        # hence this line:
                        folder_path = (
                            dset + "_" + loc.lower() + "_" + dtype.split(".")[0]
                        )
                        folder_path = os.path.join(audio_path, folder_path)

                        # get the multi channel data - note that channel 2 (at index 1 here)
                        # is backwards facing so should not be used as input to a beamformer
                        channels = sorted(
                            [
                                os.path.join(folder_path, name)
                                for name in os.listdir(folder_path)
                                if (
                                    name.startswith(id_string)
                                    and not name.endswith(".CH0.wav")
                                )
                            ]
                        )

                        # get the close talking mic for the real data,
                        # for simulated data, we use the orignial recording
                        # NOTE we store this as a list so we can use the same get_audio function in the train.py for both the
                        # multichannel and the reference
                        if not dtype.startswith(
                            "s"
                        ):  # if the data is not simulated, get the close talking mic for the reference
                            reference = [
                                os.path.join(folder_path, name)
                                for name in os.listdir(folder_path)
                                if (
                                    name.startswith(id_string)
                                    and name.endswith(".CH0.wav")
                                )
                            ]
                        else:  # the data is simulated, get the orignal recording for the reference
                            orig_path = os.path.join(audio_path, dset + "_org")
                            # this line gets us to the orignal file name which we can use as a reference for the simulated data
                            orig_id = (
                                "_".join(id_string.split("_", 2)[:2])
                                + "_ORG.wav"
                            )
                            orig_file = os.path.join(orig_path, orig_id)
                            reference = [orig_file]

                        # calculate the duration in seconds of the file (maybe not needed?)
                        # but a nice way to check if the file exists otherwise :)
                        duration = get_duration(
                            os.path.join(folder_path, channels[0])
                        )

                        # the simulated transcripts include punctuation tags,
                        # which should (hopefully) be removed via this function:
                        transcript = normalise_transcript(transcript)

                        # set up the dictionary for the json object
                        json_dict[id_string] = {
                            "speaker": speaker,  # speaker ID
                            "loc": loc,  # location tag
                            "trans_id": trans_id,  # transcript ID
                            "transcript": transcript,  # text of transcript
                            "channels": len(
                                channels
                            ),  # number of channels (should always be 6)
                            "duration": duration,  # duration in seconds
                            "wavs_far": channels,  # list of file paths to the far field channels
                            "wav_reference": reference,  # file path to the close talking mic for real data or the original signal for simulated
                        }
        # write the set to it's file
        with open(json_file, "w") as j:
            json.dump(json_dict, j, indent=2)
        logger.info(f"{json_file} successfully created!")


if __name__ == "__main__":
    prepare_chime3("/fastdata/ac1sg/corpora/CHiME3")
# normalise_transcript("THE RATE FELL TO SIX PERCENT IN NOVEMBER NINETEEN EIGHTY SIX .PERIOD")
