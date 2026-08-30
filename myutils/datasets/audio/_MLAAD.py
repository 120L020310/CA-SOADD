import os
import torch
import pandas as pd
from argparse import Namespace
from myutils.datasets.base import AudioDataset
from myutils.tools import read_file_paths_from_folder

# Global categories for MLAAD track
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

def read_metadata_of_MLADD(root_path):
    """
    Parses metadata for the MLAAD dataset.
    Consolidates real and fake audio metadata into a unified DataFrame.
    """
    fake_folder = os.path.join(root_path, "fake")
    real_folder = os.path.join(root_path, "real")

    # Load all CSV metadata files from the fake folder
    files = read_file_paths_from_folder(fake_folder, exts=["csv"])
    datas = [pd.read_csv(csv, delimiter="|") for csv in files]
    data = pd.concat(datas, ignore_index=True)

    # Process Bonafide (Real) metadata
    real_data = data.copy().drop_duplicates("original_file").reset_index(drop=True)
    real_data["label"] = 1
    real_data["path"] = real_data["original_file"]
    real_data["audio_path"] = real_data["path"].apply(lambda x: os.path.join(real_folder, x))
    real_data["vocoder"] = "bonafide"
    # Extract language code (e.g., 'en_UK' -> 'en')
    real_data["language"] = real_data["path"].apply(lambda x: x.split("/")[0][:2])

    # Process Spoof (Fake) metadata
    fake_data = data.copy()
    fake_data["label"] = 0
    # Strip prefix to match directory structure
    fake_data["audio_path"] = fake_data["path"].apply(lambda x: os.path.join(fake_folder, x[7:]))
    fake_data["vocoder"] = fake_data["architecture"]

    # Combine subsets
    cols = ["audio_path", "label", "vocoder", "path", "language"]
    combined = pd.concat([real_data[cols], fake_data[cols]], ignore_index=True)
    combined["vocoder_label"] = combined["vocoder"].apply(
        lambda x: VOCODERs.index(x) if x in VOCODERs else VOCODERs.index("unknown")
    )
    return combined

class MLAAD_AudioDs(AudioDataset):
    """
    Audio Dataset class for the Multi-Language Audio Anti-Spoofing Dataset (MLAAD).
    Supports language-specific cross-domain split generation and few-shot sampling.
    """
    default_root_path = "/path/to/MLADD"

    def postprocess(self):
        """Standardizes audio paths based on labels (real vs fake)."""
        fake_folder = os.path.join(self.root_path, "fake")
        real_folder = os.path.join(self.root_path, "real")

        self.data["audio_path"] = self.data["path"].apply(
            lambda x: os.path.join(fake_folder, x[7:]) if x.startswith("./fake") 
            else os.path.join(real_folder, x)
        )

    def _read_metadata(self, root_path, *args, **kwargs):
        """Reads and initializes dataset metadata and audio info (fps, length)."""
        data = read_metadata_of_MLADD(root_path)
        return self.read_audio_info(data)

    def get_splits(self, language_list=["en", "de", "es"], test_lang_list=["en", "de", "es", "uk"]):
        """
        Generates train/val/test splits based on linguistic domains.
        Creates language-specific test sets for cross-lingual evaluation.
        """
        # Filter In-Domain data for training
        in_domain_data = self.data.query(f"language in {language_list}").reset_index(drop=True)
        
        if len(in_domain_data) > 10:
            train, val, _ = self.split_data(in_domain_data, splits=[0.8, 0.1, 0.1], return_list=True)
        else:
            train, val = None, None

        # Prepare evaluation subsets for each target language
        out_domain_data = self.data.query(f"language in {test_lang_list}").reset_index(drop=True)
        grouped = out_domain_data.groupby("language")
        test_subsets = [group.reset_index(drop=True) for _, group in grouped]
        
        # Add a concatenated 'full' test set
        test_subsets.insert(0, out_domain_data)

        return Namespace(
            train=train,
            val=val,
            test=test_subsets,
            test_keys=["full"] + list(grouped.groups.keys()),
        )

    def get_true_few_shot(self, language_list, samples_per_lang=200):
        """
        Samples a fixed number of bonafide (real) samples per language for few-shot adaptation.
        """
        in_domain_data = self.data.query(f"language in {language_list}").reset_index(drop=True)
        
        if len(in_domain_data) <= 10:
            return pd.DataFrame()

        # Perform a standard split to ensure few-shot samples are taken from the training partition
        train_data, _, _ = self.split_data(in_domain_data, splits=[0.8, 0.1, 0.1], return_list=True)
        real_data = train_data.query("label == 1")

        # Balanced sampling per language
        few_shot_data = real_data.groupby('language', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), samples_per_lang))
        )
        return few_shot_data

    def get_true_train(self, language_list):
        """Retrieves all bonafide (real) training samples for specific languages."""
        in_domain_data = self.data.query(f"language in {language_list}").reset_index(drop=True)
        if len(in_domain_data) <= 10:
            return pd.DataFrame()
            
        train_data, _, _ = self.split_data(in_domain_data, splits=[0.8, 0.1, 0.1], return_list=True)
        return train_data.query("label == 1").reset_index(drop=True)

    def get_true_val(self, language_list):
        """Retrieves all bonafide samples from the validation partition for specific languages."""
        in_domain_data = self.data.query(f"language in {language_list}").reset_index(drop=True)
        if len(in_domain_data) <= 10:
            return pd.DataFrame()
            
        _, val_data, _ = self.split_data(in_domain_data, splits=[0.8, 0.1, 0.1], return_list=True)
        return val_data.query("label == 1").reset_index(drop=True)