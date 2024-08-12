import glob
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("C:\\Users\\transponster\\Documents\\anshul\\remoteHR")
from loader.torch_st_maps import DataLoaderRhythmNet


def get_train_test_split(fold: int):
    assert fold in {1, 2, 3, 4, 5}
    files = glob.glob("D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\**\\**\\**\\st_map_YUV_2.npy")
    files = [f for f in files if "source4" not in f]  # NIR videos to avoid

    # remove a few videos for which the number of frames is < 300
    # files = [f for f in files if "\\p22\\v3\\source1" not in f and "p2\\v5\\source3" not in f and
    #          "p46\\v2\\source3" not in f and "p84\\v2\\source2" not in f and
    #          "p10\\v9\\source2" not in f and "p28\\v2\\source2" not in f and
    #          "p54\\v5\\source3" not in f and "p46\\v2\\source1" not in f and
    #          "p40\\v7\\source2" not in f]
    target_files_f1 = []
    map_files_f1 = []

    target_files_f2 = []
    map_files_f2 = []
    for f in files:
        dir_name = os.path.dirname(f)
        p = int(dir_name.split("\\")[-3].replace("p", ""))

        if os.path.exists(os.path.join(dir_name, "gt_HR.csv")):
            if p % 5 == 5-fold:
                map_files_f1.append(f)
                target_files_f1.append(os.path.join(dir_name, "gt_HR.csv"))
            else:
                map_files_f2.append(f)
                target_files_f2.append(os.path.join(dir_name, "gt_HR.csv"))
        else:
            raise Exception(f"{dir_name} does not have gt_HR.csv")

    return map_files_f2, target_files_f2, map_files_f1, target_files_f1
    # For 2 fold
    # if fold == 1:
    #     return map_files_f1, target_files_f1, map_files_f2, target_files_f2
    # else:
    #     return map_files_f2, target_files_f2, map_files_f1, target_files_f1


def collate_fn(batch):
    batched_st_map, batched_targets = [], []
    # for data in batch:
    #     batched_st_map.append(data["st_maps"])
    #     batched_targets.append(data["target"])
    # # torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
    return batch


def get_dataloader(map_files: list, target_files: list, batch_size: int):
    ds = DataLoaderRhythmNet(st_maps_path=map_files,
                             target_signal_path=target_files)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # sample = next(iter(dl))
    #
    # print(sample)
    # print(f"Feature batch shape: {sample['st_maps'].shape}")
    # print(f"Labels batch shape: {sample['targets'].shape}")
    # print(f"Target shape: {sample['target_HR'].shape}")
    return dl


def rhythmnet_loss(output, targets, mva_targets, mean_target):
    lambda_ = 100
    loss_1 = torch.nn.L1Loss()
    loss_2 = torch.nn.L1Loss()
    return loss_1(output.mean(), mean_target) + loss_1(output, targets) + lambda_ * loss_2(output, mva_targets)


def train_fn(model, data_loader, optimizer, dev):
    model.train()
    fin_loss = 0.0

    target_hr_list = []
    predicted_hr_list = []
    files = []
    tk_iterator = tqdm(data_loader, total=len(data_loader))

    for batch in tk_iterator:
        for data in batch:
            # an item of the data is available as a dictionary
            for (key, value) in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(dev)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                _, gru_outputs = model(data["st_maps"])

                # w/o GRU
                # loss = loss_fn(outputs.squeeze(0), data["target"])
                # loss = rhythmnet_loss(outputs, data["targets"], data["target_HR"])

                sma = torch.nn.AvgPool1d(kernel_size=5, stride=1, padding=2, count_include_pad=False)
                gru_target_avg = sma(gru_outputs.unsqueeze(0)).squeeze(0)
                # print(f"target: {data['targets']}")
                # print(f"gru_target avg: {gru_target_avg}")
                # w/ GRU
                loss = rhythmnet_loss(gru_outputs, data["targets"], gru_target_avg, data["target_HR"])  # data["target_HR"])
                # loss /= len(batch)
                loss.backward()
                # no gradient accumulation throughout the batch
                optimizer.step()

            # "For each face video, the avg of all HR (bpm) of individual clips are computed as the final HR result
            # target_hr_batch = list(data["target"].mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            target_hr_list.append(data["target_HR"].item())
            # target_hr_list += data["targets"].cpu().detach().tolist()

            # predicted_hr_batch = list(outputs.squeeze(2).mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            predicted_hr_list.append(gru_outputs.mean().item())
            # predicted_hr_list += outputs.cpu().detach().tolist()

            files.append(data["st_map_path"])  # [data["st_map_path"] for _ in range(len(outputs))]
            fin_loss += loss.item()

    return target_hr_list, predicted_hr_list, fin_loss/len(data_loader), files


def eval_fn(model, data_loader, dev):
    model.eval()
    fin_loss = 0
    target_hr_list = []
    predicted_hr_list = []
    files = []
    with torch.no_grad():
        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for batch in tk_iterator:
            for data in batch:
                print(data)
                for (key, value) in data.items():
                    if torch.is_tensor(value):
                        data[key] = value.to(dev)

                # with torch.set_grad_enabled(False):
                _, gru_outputs = model(data["st_maps"])

                if gru_outputs.mean().item() > 200:
                    print(gru_outputs)

                sma = torch.nn.AvgPool1d(kernel_size=5, stride=1, padding=2, count_include_pad=False)
                gru_target_avg = sma(gru_outputs.unsqueeze(0)).squeeze(0)
                # loss with GRU
                loss = rhythmnet_loss(gru_outputs, data["targets"], gru_target_avg, data["target_HR"])
                # loss /= len(batch)
                fin_loss += loss.item()

                # target_hr_list += data["targets"].cpu().detach().tolist()
                target_hr_list.append(data["target_HR"].item())

                # predicted_hr_list += outputs.cpu().detach().tolist()
                predicted_hr_list.append(gru_outputs.mean().item())
                print(gru_outputs)
                files.append(data["st_map_path"])
                # files += [data["st_map_path"] for _ in range(len(outputs))]
            print(target_hr_list)
            print(predicted_hr_list)
        return target_hr_list, predicted_hr_list, fin_loss/len(data_loader), files
