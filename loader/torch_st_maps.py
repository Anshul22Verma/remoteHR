# import albumentations
import glob
import random

import numpy as np
import os.path
import pandas as pd
from PIL import ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose


ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLoaderRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
    """
    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, st_maps_path, target_signal_path):
        self.H = 25
        self.W = 300
        self.C = 3
        # self.video_path = data_path
        self.st_maps_path = st_maps_path
        # self.resize = resize
        self.target_path = target_signal_path

        self.maps = "YUV"
        self.dataset = "VIPL-V1"

        # Maybe add more augmentations
        # self.augmentation_pipeline = albumentations.Compose(
        #     [
        #         albumentations.Normalize(
        #             mean, std, max_pixel_value=255.0, always_apply=True
        #         )
        #     ]
        # )
        # self.transforms = Compose([ToTensor()])

    def __len__(self):
        return len(self.st_maps_path)

    def __getitem__(self, index):
        st_map = self.st_maps_path[index]
        target_csv = self.target_path[index]

        if self.dataset == "VIPL-V1":
            data = np.load(st_map)
            target = pd.read_csv(target_csv)["HR"].values.tolist()

            window = 300  # frames
            step = 12  # 0.5 seconds -> 25 fps (skip-12 frames)
            idx = 0
            # st_idx = int(np.random.randint(0, len(target)-10))
            maps = []
            targets = []

            if len(data) < 300:
                print(f"Less than 300 frames for video {st_map}, shape {data.shape}")
                data_slice = data[-1]
                data_stack = np.repeat(data_slice[np.newaxis, ...], 300 - len(data), axis=0)
                # print(data_stack.shape)
                data = np.concatenate((data, data_stack), axis=0)
                # print(data.shape)

            while idx < len(target) and (idx*step)+300 <= len(data):  # and idx < 10:
                targets.append(target[idx])
                sub_data = np.copy(data[idx*step: (idx*step) + 300])
                if random.random() > 0.5:
                    # augmentation with masking like defined
                    # randomly mask a few frames
                    mask = np.zeros((random.randint(10, 30), 25, 3))
                    # position to put the mask in
                    pos = random.randint(0, len(sub_data)-len(mask)-1)
                    sub_data[pos:pos+len(mask)] = mask
                maps.append(sub_data)
                idx += 1

            targets = np.array(targets)
            maps = np.array(maps)
            # For the temporal relationship modelling,
            # six adjacent estimated HRs are used to compute the L_smooth

            # rnadomly slice 6 consecutive maps
            # if len(maps) > 6:
            #     st_idx = np.random.randint(0, len(maps)-6)
            #     maps = maps[st_idx: st_idx+6]
            #     targets = targets[st_idx: st_idx+6]

            # print(st_map)
            # print(targets.shape)
            # print(maps.shape)

        map_shape_0 = min(len(targets), len(maps))
        # To check the fact that we don't have number of targets greater than the number of maps
        targets = targets[:map_shape_0]
        maps = maps[:map_shape_0, :, :, :]
        # make the channels as the dimension 1 to pass it through CNN
        maps = maps.swapaxes(1, 3)
        maps = maps.swapaxes(2, 3)

        return {
            "st_maps": torch.tensor(maps, dtype=torch.float),  # self.transforms(maps),
            "targets": torch.tensor(targets, dtype=torch.float),
            "target_HR": torch.tensor(sum(targets)/len(targets), dtype=torch.float),
            "st_map_path": st_map
        }


def collate_fn(batch):
    batched_st_map, batched_targets = [], []
    # for data in batch:
    #     batched_st_map.append(data["st_maps"])
    #     batched_targets.append(data["target"])
    # # torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
    return batch


if __name__ == "__main__":
    # VIPL-V1
    files = glob.glob("D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\**\\**\\**\\st_map_yuv.npy")
    files = [f for f in files if "source4" not in f]  # NIR videos to avoid
    target_files = []
    map_files = []
    for f in files:
        dir_name = os.path.dirname(f)
        if os.path.exists(os.path.join(dir_name, "gt_HR.csv")):
            map_files.append(f)
            target_files.append(os.path.join(dir_name, "gt_HR.csv"))
        else:
            raise Exception(f"{dir_name} does not have gt_HR.csv")

    ds = DataLoaderRhythmNet(st_maps_path=map_files,
                             target_signal_path=target_files)
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    sample = next(iter(dl))

    for sample in dl:
        for s in sample:
            if s['st_maps'].shape[0] == 0:
                print(s)
                # print(f"Feature batch shape: {sample[0]['st_maps'].shape}")
                # print(f"Labels batch shape: {sample[0]['targets'].shape}")
                # print(f"Target shape: {sample[0]['target_HR']}")
