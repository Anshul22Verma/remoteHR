import cv2
import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os


def segVid2stmaps(file_path: str):
    dir_name = os.path.dirname(file_path)
    # load the segmented image
    capture = cv2.VideoCapture(file_path)
    # 'D:\\anshul\\remoteHR\\mahnob\\Sessions\\103\\P1-Rec4-2009.07.09.18.57.48_C1 trigger _C_Section_3.avi')
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = 1
    frames_yuv = []
    frames_rgb = []
    frames_lab = []
    while capture.isOpened():
        try:
            ret, frame = capture.read()
            frame = cv2.resize(frame, (256, 256))
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_lab.append(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
            frames_yuv.append(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV))
            frame_count += 1
        except Exception as e:
            print(f"Coudnt process frame {frame_count} with error {e}")
            frame_count += 1
            break

    blocks = 5
    len = int(256/5)

    maps_rgb = []
    maps_lab = []
    maps_yuv = []
    for frame_yuv, frame_lab, frame_rgb in zip(frames_yuv, frames_lab, frames_rgb):
        maps_frame_rgb = []
        maps_frame_lab = []
        maps_frame_yuv = []
        for r_idx in range(blocks):
            for c_idx in range(blocks):
                row_min, row_max = r_idx*len, min(256, (r_idx+1)*len)
                col_min, col_max = c_idx*len, min(256, (c_idx+1)*len)

                # rgb
                frames_roi = frame_rgb[row_min:row_max, col_min:col_max, :]
                frames_roi = np.sum(frames_roi, axis=(0, 1)) / (np.shape(frames_roi)[0] * np.shape(frames_roi)[1])
                maps_frame_rgb.append(frames_roi)

                # lab
                frames_roi = frame_lab[row_min:row_max, col_min:col_max, :]
                frames_roi = np.sum(frames_roi, axis=(0, 1)) / (np.shape(frames_roi)[0] * np.shape(frames_roi)[1])
                maps_frame_lab.append(frames_roi)

                # yuv
                frames_roi = frame_yuv[row_min:row_max, col_min:col_max, :]
                frames_roi = np.sum(frames_roi, axis=(0, 1)) / (np.shape(frames_roi)[0] * np.shape(frames_roi)[1])
                maps_frame_yuv.append(frames_roi)
        # RGB ST-Maps
        maps_rgb.append(np.array(maps_frame_rgb))
        maps_yuv.append(np.array(maps_frame_yuv))
        maps_lab.append(np.array(maps_frame_lab))

    # frames_roi = np.average(frames_roi[np.nonzero(frames_roi)], axis=(1, 2))
    # maps.append(frames_roi)
    maps_rgb = np.array(maps_rgb)
    maps_yuv = np.array(maps_yuv)
    maps_lab = np.array(maps_lab)

    for s in range(maps_rgb.shape[1]):
        for c in range(maps_rgb.shape[2]):
            # RGB
            max_v = max(maps_rgb[:, s, c])
            min_v = min(maps_rgb[:, s, c])
            if max_v != min_v:
                maps_rgb[:, s, c] = (maps_rgb[:, s, c] - min_v) / (max_v - min_v)

            # YUV
            max_v = max(maps_yuv[:, s, c])
            min_v = min(maps_yuv[:, s, c])
            if max_v != min_v:
                maps_yuv[:, s, c] = (maps_yuv[:, s, c] - min_v) / (max_v - min_v)

            # LAB
            max_v = max(maps_lab[:, s, c])
            min_v = min(maps_lab[:, s, c])
            if max_v != min_v:
                maps_lab[:, s, c] = (maps_lab[:, s, c] - min_v) / (max_v - min_v)

    # maps = maps.swapaxes(0, 1)
    np.save(os.path.join(dir_name, "st_map_RGB.npy"), maps_rgb)
    np.save(os.path.join(dir_name, "st_map_yuv.npy"), maps_yuv)
    np.save(os.path.join(dir_name, "st_map_lab.npy"), maps_lab)

    # print(np.array(maps_rgb))
    # plt.imshow(np.array(maps_rgb).astype(np.uint8))
    # plt.waitforbuttonpress()
    # print(np.array(maps_rgb).shape)
    #
    # print(np.array(maps_lab))
    # plt.imshow(np.array(maps_lab).astype(np.uint8))
    # plt.waitforbuttonpress()
    # print(np.array(maps_lab).shape)
    #
    # print(np.array(maps_yuv))
    # plt.imshow(np.array(maps_yuv).astype(np.uint8))
    # plt.waitforbuttonpress()
    # print(np.array(maps_yuv).shape)
    print(dir_name)


if __name__ == "__main__":
    # # COHFACE
    # files = glob.glob("D:\\anshul\\remoteHR\\4081054\\cohface\\**\\**\\DnS_v.avi")
    # # segVid2stmaps(files[0])
    # pool = multiprocessing.Pool(8)
    # pool.map(segVid2stmaps, files)

    # VIPL-V1
    files = glob.glob("D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\**\\**\\**\\DnS_v.avi")
    # segVid2stmaps(file_path="D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\p22\\v3\\source1\\DnS_v.avi")
    pool = multiprocessing.Pool(10)
    pool.map(segVid2stmaps, files)

    # # MANHOB
    # files = glob.glob("D:\\anshul\\remoteHR\\mahnob\\Sessions\\**\\DnS_v.avi")
    # pool = multiprocessing.Pool(8)
    # pool.map(segVid2stmaps, files)

    # engagement-test
    # files = glob.glob("D:\\anshul\\engagement_test\\**\\DnS_v.avi")
    # print(files)
    # pool = multiprocessing.Pool(4)
    # pool.map(segVid2stmaps, files)
