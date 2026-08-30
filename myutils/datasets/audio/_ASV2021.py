import os
import pandas as pd
from argparse import Namespace
from myutils.datasets.base import AudioDataset
from myutils.tools import read_file_paths_from_folder

# Global categories for ASVspoof 2021 DF Track
VOCODERs = [
    "bonafide",
    "neural_vocoder_autoregressive",
    "neural_vocoder_nonautoregressive",
    "traditional_vocoder",
    "unknown",
    "waveform_concatenation",
]

COMPRESSIONs = [
    "high_m4a", "high_mp3", "high_ogg",
    "low_m4a", "low_mp3", "low_ogg",
    "mp3m4a", "nocodec", "oggm4a",
]

class ASV2021_AudioDs(AudioDataset):
    """
    Audio Dataset class for ASVspoof 2021 DeepFake (DF) track.
    Focuses on artifacts introduced by different vocoders and lossy compression codecs.
    """
    def postprocess(self):
        """Standard post-loading operations for paths and label indexing."""
        self.data["audio_path"] = self.data["file"].apply(
            lambda x: os.path.join(self.root_path, x)
        )
        self.data["compression_label"] = self.data["compression"].apply(
            lambda x: COMPRESSIONs.index(x)
        )
        self.vocoders = VOCODERs

    def _read_metadata_legacy(self, root_path, *args, **kwargs):
        """Legacy metadata reader for 2019-style folder structures."""
        paths = read_file_paths_from_folder(root_path, exts="flac")
        data = pd.DataFrame(paths, columns=["path"])
        
        data["relative_path"] = data["path"].apply(
            lambda x: x.replace(root_path + "/", "")
        )
        data["filename"] = data["path"].apply(
            lambda x: os.path.split(x)[1].replace(".flac", "")
        )
        data["split"] = data["path"].apply(
            lambda x: "train" if "_train" in x else ("val" if "_dev" in x else "test")
        )

        label_data = self.read_label_data(root_path)
        data = pd.merge(data, label_data)
        data["vocoder_label"] = data["method"].apply(lambda x: VOCODERs.index(x))

        return self.read_audio_info(data)

    def _read_metadata(self, root_path, *args, **kwargs):
        """Primary metadata reader for the 2021 DF used_metadata CSV format."""
        data = pd.read_csv(os.path.join(root_path, "used_metadata.csv"))
        
        # Map filenames and labels
        data["file"] = data["trial"].apply(lambda x: f"wav/{x}.wav")
        data["label"] = data["label"].apply(lambda x: 1 if x == "bonafide" else 0)
        data["vocoder_label"] = data["vocoder"].apply(lambda x: VOCODERs.index(x))
        data["audio_path"] = data["file"].apply(lambda x: os.path.join(root_path, x))

        # Split normalization
        data["split"] = data["subset"].apply(
            lambda x: "test" if x == "eval" else "train"
        )

        # Randomly sample 10,000 samples for validation from the progress subset
        val_samples = data.query("subset == 'progress'").sample(10000, random_state=42)
        data.loc[val_samples.index, "split"] = "val"

        return self.read_audio_info(data)

    def get_splits(self):
        """
        Generates standard dataset splits.
        Returns a Namespace containing train, val, and a list of test subsets categorized by vocoder.
        """
        data = self.data
        sub_datas = []
        for split in ["train", "val", "test"]:
            _data = data.query(f'split == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)

        # Break down the test set by vocoder for granular evaluation
        test_subsets = self.get_test_splits(sub_datas[2])

        return Namespace(
            train=sub_datas[0],
            val=sub_datas[1],
            test=test_subsets,
        )

    def get_test_splits(self, data=None):
        """
        Splits the test data into multiple subsets.
        Each subset contains one specific vocoder type plus all bonafide samples.
        """
        data = data if data is not None else self.data.query("split == 'test'")
        sub_datas = []
        for vocoder in VOCODERs[1:]:
            _subset = data.query(
                f"vocoder == '{vocoder}' or vocoder == '{VOCODERs[0]}'"
            ).reset_index(drop=True)
            sub_datas.append(_subset)
        return sub_datas

    def get_fake_train(self, data=None):
        """Retrieves all spoofed (fake) samples from the training set."""
        data = data if data is not None else self.data.query("split == 'train'")
        return data.query("label == 0").reset_index(drop=True)

    def get_true_train(self, data=None):
        """Retrieves all bonafide (real) samples from the training set."""
        data = data if data is not None else self.data.query("split == 'train'")
        return data.query("label == 1").reset_index(drop=True)

    def get_true_val(self, data=None):
        """Retrieves all bonafide samples from the validation set."""
        data = data if data is not None else self.data.query("split == 'val'")
        return data.query("label == 1").reset_index(drop=True)