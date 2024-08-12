import io
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import r2_score
import torch


def gt_vs_est(data1, data2, plot_path=None, tb=False):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    # mean = np.mean([data1, data2], axis=0)
    # diff = data1 - data2                   # Difference between data1 and data2
    # md = np.mean(diff)                   # Mean of the difference
    # sd = np.std(diff, axis=0)            # Standard deviation of the difference

    fig = plt.figure()
    plt.scatter(data1, data2)
    plt.title('true labels vs estimated')
    plt.ylabel('estimated HR')
    plt.xlabel('true HR')
    # plt.axhline(md,           color='gray', linestyle='--')
    # plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    # plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlim(min(min(data1), min(data2)) - 1, max(max(data1), max(data2)) + 1)
    plt.xlim(min(min(data1), min(data2)) - 1, max(max(data1), max(data2)) + 1)
    plt.plot([min(min(data1), min(data2)) - 1, max(max(data1), max(data2)) + 1],
             [min(min(data1), min(data2)) - 1, max(max(data1), max(data2)) + 1], 'k--',
             label=f'$R^2$: {round(r2_score(data1, data2), 4)}')
    plt.legend()
    # plt.show()
    fig.savefig(os.path.join(plot_path), dpi=fig.dpi)
    if tb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf
    plt.close(fig)


def bland_altman_plot(data1, data2, plot_path=None, tb=False):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    fig = plt.figure()
    plt.scatter(mean, diff)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.xlabel("($HR_{gt}$ + $HR_{est}$)/2")
    plt.ylabel("$HR_{gt}$ - $HR_{est}$")
    plt.title("Bland-Altman plot")
    # plt.show()
    fig.savefig(os.path.join(plot_path), dpi=fig.dpi)

    if tb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf
    plt.close(fig)


def load_model_if_checkpointed(model, optimizer, checkpoint_path, load_on_cpu=False):
    loss = 0.0
    checkpoint_flag = False

    # check if checkpoint exists
    if os.path.exists(os.path.join(checkpoint_path, "running_model.pt")):
        checkpoint_flag = True
        if load_on_cpu:
            checkpoint = torch.load(os.path.join(checkpoint_path, "running_model.pt"), map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(os.path.join(checkpoint_path, "running_model.pt"))

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    return model, optimizer, loss, checkpoint_flag


def save_model_checkpoint(model, optimizer, loss, checkpoint_path):
    save_filename = "running_model.pt"
    # checkpoint_path = os.path.join(checkpoint_path, save_filename)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save({
        # 'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(checkpoint_path, save_filename))
    print('Saved!')


def rmse(l1, l2):

    return np.sqrt(np.mean((l1-l2)**2))


def mae(l1, l2):

    return np.mean([abs(item1-item2)for item1, item2 in zip(l1, l2)])


def compute_criteria(target_hr_list, predicted_hr_list):
    pearson_per_signal = []
    HR_MAE = mae(np.array(predicted_hr_list), np.array(target_hr_list))
    HR_RMSE = rmse(np.array(predicted_hr_list), np.array(target_hr_list))

    # for (gt_signal, predicted_signal) in zip(target_hr_list, predicted_hr_list):
    #     r, p_value = pearsonr(predicted_signal, gt_signal)
    #     pearson_per_signal.append(r)

    # return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE, "Pearson": np.mean(pearson_per_signal)}
    return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE}
