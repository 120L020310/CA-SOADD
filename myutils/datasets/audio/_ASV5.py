import os
import torch
import pandas as pd
from argparse import Namespace
from myutils.datasets.base import AudioDataset

# Define global constants for Vocoder and Compression types
VOCODERs = ["bonafide"] + ["A%02d" % i for i in range(1, 33)]

COMPRESSIONs = [
    "-", "C01", "C02", "C03", "C04", "C05", 
    "C06", "C07", "C08", "C09", "C10", "C11"
]

def init_track1_data(root_path):
    """
    Initializes and cleans metadata for ASVspoof5 Track 1.
    Handles file path mapping and label indexing.
    """
    data = pd.read_csv(os.path.join(root_path, "metadata/track1.csv"), sep=",", low_memory=False)
    
    # Map file names to their respective directory structures
    data["file"] = data["FLAC_FILE_NAME"].apply(
        lambda x: (
            f"flac_D/{x}.flac" if x.startswith("D")
            else (f"flac_E_eval/{x}.flac" if x.startswith("E") else f"flac_T/{x}.flac")
        )
    )
    
    # Label encoding for vocoders and compression codecs
    data["vocoder"] = data["ATTACK_LABEL"]
    data["vocoder_label"] = data["vocoder"].apply(lambda x: VOCODERs.index(x))
    data["audio_path"] = data["file"].apply(lambda x: os.path.join(root_path, x))
    data["compression"] = data["CODEC"]
    data["compression_label"] = data["compression"].apply(lambda x: COMPRESSIONs.index(x))
    data["compression_quality"] = data["CODEC_Q"].replace("-", 0)
    
    return data

class ASVSpoof5_AudioDs(AudioDataset):
    """
    Audio Dataset class for ASVspoof 2024 (ASVspoof5).
    Inherits from base AudioDataset to provide standardized access to audio files and metadata.
    """
    def postprocess(self):
        """Finalizes audio paths and metadata references."""
        self.data["audio_path"] = self.data["file"].apply(
            lambda x: os.path.join(self.root_path, x)
        )
        self.vocoders = VOCODERs
        self.compressions = COMPRESSIONs

    def read_metadata(self, root_path, *args, **kwargs):
        """Loads metadata from the root path."""
        return init_track1_data(root_path)

    def split_train_val_in_train_tsv(self, train_data=None, train_val_rate_in_train_tsv=0.8):
        """
        Splits the training set into training and validation subsets ensuring 
        speaker IDs are disjoint between the two.
        """
        from myutils.tools.pandas import DF_spliter

        if train_data is None:
            train_data = self.data[self.data['split'] == 'train']

        # Disjoint split based on Speaker ID
        train, val = DF_spliter.split_by_number_and_column(train_data, [0.8, 0.2], refer='SPEAKER_ID')

        ids_train = set(train['SPEAKER_ID'])
        ids_val = set(val['SPEAKER_ID'])
        
        print("-" * 30)
        print("ASVspoof5 Track 1 Split Summary:")
        print(f"Train subset: {len(ids_train)} speakers, {len(train)} audios")
        print(f"Val subset:   {len(ids_val)} speakers, {len(val)} audios")
        print(f"Disjoint Speakers Check: {ids_train.isdisjoint(ids_val)}")
        print("-" * 30)
        
        return train, val

    def get_true_train(self, data=None):
        """Retrieves only Bonafide samples from the training split."""
        data = data if data is not None else self.data.query("split == 'train'")
        return data.query("label == 1").reset_index(drop=True)

    def get_true_val(self, data=None):
        """Retrieves only Bonafide samples from the development (dev) split."""
        data = data if data is not None else self.data.query("split == 'dev'")
        return data.query("label == 1").reset_index(drop=True)

    def get_splits(self, train_val_rate_in_train_tsv=0.8, 
                   use_dev_as_test=False, 
                   use_both_dev_test_for_test=False,
                   only_test_vocoder=False):
        """
        Generates standard train/val/test splits.
        Supports various testing configurations (e.g., using dev as test).
        """
        data = self.data
        train_raw = data.query("split == 'train'").reset_index(drop=True)
        train, val = self.split_train_val_in_train_tsv(train_raw, train_val_rate_in_train_tsv)
        
        test = data.query("split == 'test'").reset_index(drop=True)
        dev = data.query("split == 'dev'").reset_index(drop=True)
        
        res = Namespace(train=train, val=val)
        
        if use_dev_as_test:
            res.test = dev
        elif use_both_dev_test_for_test:
            all_test_data = pd.concat([dev, test], ignore_index=True)
            # Standard test data + breakdowns by vocoder categories
            res.test = [dev, test, all_test_data] + self.get_test_splits_for_vocoder_types(all_test_data)
        elif only_test_vocoder:
            all_test_data = pd.concat([dev, test], ignore_index=True)
            sampled_test_data = all_test_data.sample(n=50000)
            res.test = [sampled_test_data] + self.get_test_splits_for_vocoder_types(all_test_data)
        else:
            res.test = test
        return res

    def get_test_splits_for_vocoder_types(self, data=None, add_bonafide=True):
        """
        Splits test data into three categories based on synthesis technology: 
        Voice Conversion (VC), Text-to-Speech (TTS), and Audio Tampering (AT).
        """
        data = self.data if data is None else data
        
        # ID lists based on ASVspoof5 technical specifications
        if add_bonafide:
            vc_list, tts_list, at_list = [0, 13, 15, 16, 24, 25, 26], \
                                         [0, 9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29], \
                                         [0, 18, 20, 23, 27, 30, 31, 32]
        else:
            vc_list, tts_list, at_list = [13, 15, 16, 24, 25, 26], \
                                         [9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29], \
                                         [18, 20, 23, 27, 30, 31, 32]

        sub_data = []
        for l in [vc_list, tts_list, at_list]:
            subset = data[data['vocoder_label'].isin(l)]
            # Uniform sampling for comparative evaluation
            sub_data.append(subset.sample(min(10000, len(subset))))
            
        return sub_data

    def get_test_splits_for_vocoder_types_T_SNE(self, data=None):
        """
        Prepares data specifically for t-SNE visualization.
        Groups samples into four classes: Bonafide, VC, TTS, and AT.
        """
        data = self.data if data is None else data
        
        # Category mappings for visualization
        bonafide_data = data[data['vocoder_label'] == 0].copy()
        bonafide_data['vocoder_label'] = 0
        
        vc_data = data[data['vocoder_label'].isin([13, 15, 16, 24, 25, 26])].copy()
        vc_data['vocoder_label'] = 1
        
        tts_data = data[data['vocoder_label'].isin([9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29])].copy()
        tts_data['vocoder_label'] = 2
        
        at_data = data[data['vocoder_label'].isin([18, 20, 23, 27, 30, 31, 32])].copy()
        at_data['vocoder_label'] = 3

        return [bonafide_data, vc_data, tts_data, at_data]