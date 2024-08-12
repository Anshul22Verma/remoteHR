import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import PIL
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

sys.path.append("C:\\Users\\transponster\\Documents\\anshul\\remoteHR")
from models.rhythmnet import RhythmNet
from utils.helper import bland_altman_plot, gt_vs_est, load_model_if_checkpointed, save_model_checkpoint, \
    compute_criteria
from pipe.vipl_v1 import get_train_test_split, get_dataloader, train_fn, eval_fn


def create_plot_for_tensorboard(plot_name, data1, data2, tb: bool = True, plot_path: str = ""):
    if plot_name == "bland_altman":
        if tb:
            fig_buf = bland_altman_plot(data1, data2, tb=True)
        else:
            fig_buf = bland_altman_plot(data1, data2, plot_path=plot_path)
    if plot_name == "gt_vs_est":
        if tb:
            fig_buf = gt_vs_est(data1, data2, tb=True)
        else:
            fig_buf = gt_vs_est(data1, data2, plot_path=plot_path)

    if tb:
        image = PIL.Image.open(fig_buf)
        image = ToTensor()(image)

        return image
    else:
        return None


def run_training(config):
    # check path to checkpoint directory
    if config["CHECKPOINT_PATH"]:
        if not os.path.exists(config["CHECKPOINT_PATH"]):
            os.makedirs(config["CHECKPOINT_PATH"])
            print("Output directory is created")

    # --------------------------------------
    # Initialize Model
    # --------------------------------------

    model = RhythmNet()

    if torch.cuda.is_available():
        print('GPU available... using GPU')
        torch.cuda.manual_seed_all(42)
    else:
        print("GPU not available, using CPU")

    if config["CHECKPOINT_PATH"]:
        os.makedirs(config["CHECKPOINT_PATH"], exist_ok=True)
        print("Output directory is created")

    if config["DEVICE"] == "cpu":
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    # Initialize SummaryWriter object
    writer = SummaryWriter()

    # Read from a pre-made csv file that contains data divided into folds for cross validation
    FOLD = config["FOLD"]
    map_files_train, target_files_train, map_files_test, target_files_test = get_train_test_split(fold=FOLD)

    # Loop for enumerating through folds.
    print(f"Training for {config['EPOCHS']} Epochs (each video)")
    # --------------------------------  ------
    # Build Dataloaders
    # --------------------------------------

    train_loader = get_dataloader(map_files=map_files_train, target_files=target_files_train,
                                  batch_size=config["BATCH_SIZE"])
    test_loader = get_dataloader(map_files=map_files_test, target_files=target_files_test,
                                 batch_size=config["BATCH_SIZE"])
    print('\nTrain and Test DataLoader constructed successfully!')

    # Code to use multiple GPUs (if available)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # --------------------------------------
    # Load checkpointed model (if  present)
    # --------------------------------------
    if device == "cpu":
        load_on_cpu = True
    else:
        load_on_cpu = False
    model, optimizer, checkpointed_loss, \
        checkpoint_flag = load_model_if_checkpointed(model=model, optimizer=optimizer,
                                                     checkpoint_path=config["CHECKPOINT_PATH"],
                                                     load_on_cpu=load_on_cpu)
    if checkpoint_flag:
        print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
    else:
        print("Checkpoint Not Found! Training from beginning")

    # -----------------------------
    # Start training
    # -----------------------------

    train_loss_per_epoch = []
    for epoch in range(config["EPOCHS"]):
        target_hr_list, predicted_hr_list, train_loss, files = train_fn(model=model, data_loader=train_loader,
                                                                        optimizer=optimizer, dev=device)

        # Save model with final train loss (script to save the best weights?)
        if checkpointed_loss != 0.0:
            if train_loss < checkpointed_loss:
                save_model_checkpoint(model=model, optimizer=optimizer, loss=train_loss,
                                      checkpoint_path=config["CHECKPOINT_PATH"])
                checkpointed_loss = train_loss
            else:
                pass
        else:
            if len(train_loss_per_epoch) > 0:
                if train_loss < min(train_loss_per_epoch):
                    save_model_checkpoint(model=model, optimizer=optimizer, loss=train_loss,
                                          checkpoint_path=config["CHECKPOINT_PATH"])
            else:
                save_model_checkpoint(model=model, optimizer=optimizer, loss=train_loss,
                                      checkpoint_path=config["CHECKPOINT_PATH"])

        metrics = compute_criteria(target_hr_list, predicted_hr_list)

        for metric in metrics.keys():
            writer.add_scalar(f"Train/{metric}", metrics[metric], epoch)

        print(f"\nFinished [Epoch: {epoch + 1}/{config['EPOCHS']}]",
              "\nTraining Loss: {:.3f} |".format(train_loss),
              "HR_MAE : {:.3f} |".format(metrics["MAE"]),
              "HR_RMSE : {:.3f} |".format(metrics["RMSE"]),)
              # "Pearsonr : {:.3f} |".format(metrics["Pearson"]), )

        train_loss_per_epoch.append(train_loss)
        writer.add_scalar("Loss/train", train_loss, epoch+1)

        # Plots on tensorboard
        ba_plot_image = create_plot_for_tensorboard('bland_altman', target_hr_list, predicted_hr_list)
        gtvsest_plot_image = create_plot_for_tensorboard('gt_vs_est', target_hr_list, predicted_hr_list)
        writer.add_image('BA_plot', ba_plot_image, epoch)
        writer.add_image('gtvsest_plot', gtvsest_plot_image, epoch)

    mean_loss = np.mean(train_loss_per_epoch)
    # Save the mean_loss value for each video instance to the writer
    print(f"Avg Training Loss: {np.mean(mean_loss)} for {config['EPOCHS']} epochs")
    writer.flush()

    # --------------------------------------
    # Load checkpointed model (if  present)
    # --------------------------------------
    if config["DEVICE"] == "cpu":
        load_on_cpu = True
    else:
        load_on_cpu = False
    model, optimizer, checkpointed_loss, \
        checkpoint_flag = load_model_if_checkpointed(model=model, optimizer=optimizer,
                                                     checkpoint_path=config["CHECKPOINT_PATH"],
                                                     load_on_cpu=load_on_cpu)
    if checkpoint_flag:
        print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
    else:
        print("Checkpoint Not Found! Training from beginning")

    # -----------------------------
    # Start Validation
    # -----------------------------
    # model = torch.load("D:\\anshul\\rPPG\\experiments\\vipl-fold-1-GRU\\running_model.pt")

    print(f"Finished Training, Validating {len(test_loader)} video files")
    # validation
    target_hr_list, predicted_hr_list, test_loss, files = eval_fn(model=model, data_loader=test_loader,
                                                                  dev=device)
    df = pd.DataFrame()
    df["target"] = target_hr_list
    df["predicted"] = predicted_hr_list
    df["file"] = files
    df.to_csv(os.path.join(config["CHECKPOINT_PATH"], "predictions.csv"), index=False)
    # truth_hr_list.append(target)
    # estimated_hr_list.append(predicted)
    metrics = compute_criteria(target_hr_list, predicted_hr_list)
    for metric in metrics.keys():
        writer.add_scalar(f"Test/{metric}", metrics[metric], 1)

    print(f"\nFinished Test",
          "\nTest Loss: {:.3f} |".format(test_loss),
          "HR_MAE : {:.3f} |".format(metrics["MAE"]),
          "HR_RMSE : {:.3f} |".format(metrics["RMSE"]),)

    writer.add_scalar("Loss/test", test_loss, 1)

    # Plots on tensorboard
    ba_plot_image = create_plot_for_tensorboard('bland_altman', target_hr_list, predicted_hr_list)
    gtvsest_plot_image = create_plot_for_tensorboard('gt_vs_est', target_hr_list, predicted_hr_list)
    writer.add_image('BA_plot', ba_plot_image, 1)
    writer.add_image('gtvsest_plot', gtvsest_plot_image, 1)
    gt_vs_est(target_hr_list, predicted_hr_list, plot_path=config["PLOT_PATH"])
    bland_altman_plot(target_hr_list, predicted_hr_list, plot_path=config["PLOT_PATH"])

    # checking the performance on training set as well for plotting
    print(f"Finished Testing, Checking performance on train {len(train_loader)} video files")
    target_hr_list, predicted_hr_list, test_loss, files = eval_fn(model=model, data_loader=train_loader,
                                                                  dev=device)

    # truth_hr_list.append(target)
    # estimated_hr_list.append(predicted)
    metrics = compute_criteria(target_hr_list, predicted_hr_list)
    for metric in metrics.keys():
        writer.add_scalar(f"Evaluation Train/{metric}", metrics[metric], 1)

    print(f"\nFinished Train evaluation",
          "\nTest Loss: {:.3f} |".format(test_loss),
          "HR_MAE : {:.3f} |".format(metrics["MAE"]),
          "HR_RMSE : {:.3f} |".format(metrics["RMSE"]), )

    writer.add_scalar("Loss/test", test_loss, 1)

    # Plots on tensorboard
    os.makedirs(os.path.join(config["CHECKPOINT_PATH"], "train"), exist_ok=True)
    ba_plot_image = create_plot_for_tensorboard('bland_altman', target_hr_list, predicted_hr_list,
                                                tb=False, plot_path=os.path.join(config["CHECKPOINT_PATH"], "train"))
    gtvsest_plot_image = create_plot_for_tensorboard('gt_vs_est', target_hr_list, predicted_hr_list,
                                                     tb=False, plot_path=os.path.join(config["CHECKPOINT_PATH"], "train"))
    # # writer.add_image('BA_plot', ba_plot_image, epoch)
    # # writer.add_image('gtvsest_plot', gtvsest_plot_image, epoch)

    writer.flush()
    # plot_train_test_curves(train_loss_data, test_loss_data, plot_path=config["PLOT_PATH"], fold_tag=k)
    # Plots on the local storage.
    writer.close()
    print("done")


if __name__ == '__main__':
    config = {
        "CHECKPOINT_PATH": "D:\\anshul\\rPPG\\experiments\\vipl-fold-1-GRU",
        "EPOCHS": 0,
        "lr": 0.001,
        "PLOT_PATH": "D:\\anshul\\rPPG\\experiments\\vipl-fold-1-GRU\\test-plots",
        "FOLD": 1,
        "DEVICE": "gpu",
        "BATCH_SIZE": 16,
    }
    run_training(config)
