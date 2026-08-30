import os
import pandas as pd
from argparse import Namespace
from myutils.datasets.base import AudioDataset
from myutils.tools import read_file_paths_from_folder

# Define global constants for Logical Access (LA) attack methods
VOCODERs = ["bonafide"] + ["A{:02d}".format(i) for i in range(7, 20)]

class ASV2021LA_AudioDs(AudioDataset):
    """
    Audio Dataset class for ASVspoof 2021 Logical Access (LA) track.
    Handles metadata parsing, split generation, and feature extraction setup.
    """

    def update_audio_path(self, data, root_path):
        """Updates the absolute audio path based on the relative path in metadata."""
        data["audio_path"] = data["relative_path"].apply(
            lambda x: os.path.join(root_path, x)
        )

    def postprocess(self):
        """Standard post-loading operations: path updates and vocoder list initialization."""
        self.update_audio_path(self.data, self.root_path)
        self.vocoders = VOCODERs

    def read_label_data(self, root_path):
        """
        Parses the official trial metadata files.
        Maps string labels to binary integers and normalizes split names.
        """
        label_files = ["keys/LA/CM/trial_metadata.txt"]

        label_data = pd.concat(
            [
                pd.read_csv(
                    os.path.join(root_path, _f),
                    delimiter=" ",
                    names=[
                        "speaker", "filename", "codec", "col1",
                        "method", "label", "trim", "split",
                    ],
                )
                for _f in label_files
            ],
            ignore_index=True,
        )

        # Convert labels: 'bonafide' -> 1 (True), 'spoof' -> 0 (False)
        label_data["label"] = (
            label_data["label"].replace("bonafide", 1).replace("spoof", 0)
        )
        
        # Normalize split names for consistency
        label_data["split"] = (
            label_data["split"].replace("progress", "train").replace("eval", "test")
        )

        # Randomly reserve 2000 samples from the training set for validation
        val_data = label_data.query("split == 'train'").sample(2000, random_state=42)
        label_data.loc[val_data.index, "split"] = "val"

        return label_data

    def _read_metadata(self, root_path, *args, **kwargs):
        """
        Scans the local filesystem for audio files and merges them with label data.
        """
        paths = read_file_paths_from_folder(root_path, exts="flac")
        data = pd.DataFrame(paths, columns=["path"])
        
        data["relative_path"] = data["path"].apply(
            lambda x: x.replace(root_path + "/", "")
        )
        data["filename"] = data["path"].apply(
            lambda x: os.path.split(x)[1].replace(".flac", "")
        )

        label_data = self.read_label_data(root_path)
        data = pd.merge(data, label_data)

        # Map attack methods to integer indices
        data["vocoder_label"] = data["method"].apply(lambda x: VOCODERs.index(x))

        self.update_audio_path(data, root_path)
        # Inherited method to read file properties (sample rate, length)
        data = self.read_audio_info(data)  
        return data

    def get_splits(self):
        """
        Returns the data divided into train, val, and test Namespace.
        """
        data = self.data
        sub_datas = []
        for split in ["train", "val", "test"]:
            _data = data.query(f'split == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)

        return Namespace(
            train=sub_datas[0],
            val=sub_datas[1],
            test=sub_datas[2],
        )
    
    def get_fake_train(self, data=None):
        """Retrieves only spoofed (fake) samples from the training split."""
        data = data if data is not None else self.data.query("split == 'train'")
        return data.query("label == 0").reset_index(drop=True)

    def get_true_train(self, data=None):
        """Retrieves only bonafide (real) samples from the training split."""
        data = data if data is not None else self.data.query("split == 'train'")
        return data.query("label == 1").reset_index(drop=True)

    def get_true_val(self, data=None):
        """Retrieves only bonafide samples from the validation split."""
        data = data if data is not None else self.data.query("split == 'val'")
        return data.query("label == 1").reset_index(drop=True)
    
    def get_test_splits_for_tsne(self, data=None):
        """
        Prepares data subsets for t-SNE visualization.
        Classifies samples into binary groups: Bonafide (0) and Spoof (1).
        """
        data = self.data if data is None else data
        
        bona_mask = data['label'] == 1
        fake_mask = data['label'] == 0
        
        bona_data = data[bona_mask].copy()
        fake_data = data[fake_mask].copy()
        
        # Internal labels for visualization consistency
        bona_data['vocoder_label'] = 0
        fake_data['vocoder_label'] = 1

        return [bona_data, fake_data]