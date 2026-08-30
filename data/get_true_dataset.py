
from torch.utils.data import ConcatDataset, DataLoader

from data.my_dataset import get_dataset_and_loader_ASV2021_DF, get_dataset_and_loader_ASV2021_LA, get_dataset_and_loader_ASV5, get_dataset_and_loader_MLAAD

def get_true(tag,stage="val",bs=16,label = "train",language_list = None):
    datasets = []
    dataloaders = []
    for ds_name in tag:
        if ds_name=="ASV2021_LA":
            ds,dl = get_dataset_and_loader_ASV2021_LA(label,stage)
        if ds_name=="ASV2021_DF":
            ds,dl = get_dataset_and_loader_ASV2021_DF(label,stage)
        if ds_name=="ASV5":
            ds,dl = get_dataset_and_loader_ASV5(label,stage)
        if ds_name=="MLAAD":
            ds,dl = get_dataset_and_loader_MLAAD(label,stage,language_list)
        datasets.append(ds)
        dataloaders.append(dl)
    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(combined_dataset, batch_size=bs, shuffle=True, num_workers=4)
    print(f"Total data: {len(combined_dataset)}")
    return combined_dataset,loader

