from collections import defaultdict
import cv2
import glob
import math
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from PIL import Image
import time
from typing import Tuple, Union

segmenter_model_path = "D:\\anshul\\remoteHR\\mediapipe\\selfie_multiclass_256x256.tflite"


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")
    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def resize_w_padding(img, max_h, max_w):
    h, w, _ = img.shape
    if h < max_h or w < max_w:
        delta_w = max_w - w
        delta_h = max_h - h

        top, bottom = delta_h//2, delta_h - (delta_h//2)
        left, right = delta_w//2, delta_w - (delta_w//2)
        color = [0, 0, 0]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


base_options = python.BaseOptions(model_asset_path='D:\\anshul\\remoteHR\\mediapipe\\face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)
# base_options = python.BaseOptions(model_asset_path='D:\\anshul\\remoteHR\\mediapipe\\blaze_face_short_range.tflite')
# options = vision.FaceDetectorOptions(base_options=base_options)
# detector = vision.FaceDetector.create_from_options(options)


def process_video(video_path: str):
    dir_path = os.path.dirname(video_path)
    capture = cv2.VideoCapture(video_path)

    # segmented_images = []
    regions_avg_px_R = defaultdict(list)
    regions_avg_px_G = defaultdict(list)
    regions_avg_px_B = defaultdict(list)

    regions_avg_px_L = defaultdict(list)
    regions_avg_px_A = defaultdict(list)
    regions_avg_px_B_ = defaultdict(list)

    regions_avg_px_Y = defaultdict(list)
    regions_avg_px_U = defaultdict(list)
    regions_avg_px_V = defaultdict(list)
    frame_count = 0
    max_h, max_w = 0, 0
    previous_face_landmarks = None

    segmented_aligned_frames = []
    while capture.isOpened():
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        try:
            # capture frame by frame
            ret, frame = capture.read()
            frame_count += 1
            # Converting from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            image_ = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            detection_result = detector.detect(image_)
            # use the image_ to align using eyes and segment the skin only region

            # mask the eyes
            LE_points = []
            for p in [133, 173, 157, 158, 159, 160, 161, 246,
                      33, 7, 163, 144, 145, 153, 154, 155]:
                LE_points.append([detection_result.face_landmarks[0][p].x * w,
                                  detection_result.face_landmarks[0][p].y * h])

            LE_points = np.array(LE_points, np.int32)
            image = cv2.fillPoly(image, [LE_points], 0)

            RE_points = []
            for p in [362, 398, 384, 385, 386, 387, 388, 466, 263,
                      249, 390, 373, 374, 380, 381, 382]:
                RE_points.append([detection_result.face_landmarks[0][p].x * w,
                                  detection_result.face_landmarks[0][p].y * h])

            RE_points = np.array(RE_points, np.int32)
            image = cv2.fillPoly(image, [RE_points], 0)

            # get the left-eye and right-eye points in the space for facial alignment
            if len(detection_result.face_landmarks) < 1:
                detection_result.face_landmarks = previous_face_landmarks
            le1, le2 = detection_result.face_landmarks[0][33], detection_result.face_landmarks[0][133]
            re1, re2 = detection_result.face_landmarks[0][362], detection_result.face_landmarks[0][263]

            le = ((le1.x + le2.x) / 2, (le1.y + le2.y) / 2)
            re = ((re1.x + re2.x) / 2, (re1.y + re2.y) / 2)

            le = (int(le[0] * image.shape[1]), int(le[1] * image.shape[0]))  # image_f
            re = (int(re[0] * image.shape[1]), int(re[1] * image.shape[0]))  # image_f

            # face alignment with le and re
            delta_x = re[0] - le[0]
            delta_y = re[1] - le[1]
            angle = np.arctan(delta_y / delta_x)
            angle = (angle * 180) / np.pi
            h, w = image.shape[:2]
            # Calculating a center point of the image
            center = (w // 2, h // 2)
            # Defining a matrix M and calling
            # cv2.getRotationMatrix2D method
            M = cv2.getRotationMatrix2D(center, (angle), 1.0)
            # Applying the rotation to our image using the
            # cv2.warpAffine method
            rotated = cv2.warpAffine(image, M, (w, h))
            # rotate the landmark key points
            lc, rc = detection_result.face_landmarks[0][93], detection_result.face_landmarks[0][323]
            ec, ch = detection_result.face_landmarks[0][8], detection_result.face_landmarks[0][152]
            # print(F"Frame Count: {frame_count}")

            lc, rc = np.array([lc.x*w, lc.y*h]), np.array([rc.x*w, rc.y*h])
            ec, ch = np.array([ec.x*w, ec.y*h]), np.array([ch.x*w, ch.y*h])

            points = np.array([lc, rc, ec, ch])
            rotated_points = cv2.transform(points.reshape(-1, 1, 2), M).reshape(-1, 2)
            lc_r, rc_r, ec_r, ch_r = rotated_points[0], rotated_points[1], rotated_points[2], rotated_points[3]

            # now segment the skin pixels
            cropped_image = rotated[int(ec_r[1] - 0.2*h):int(ch_r[1]), int(lc_r[0]):int(rc_r[0])]
            image_ = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.array(cropped_image))

            base_options = python.BaseOptions(model_asset_path=segmenter_model_path)
            options = vision.ImageSegmenterOptions(base_options=base_options,
                                                   output_category_mask=True)
            BG_COLOR = (0, 0, 0)  # black
            MASK_COLOR = (255, 255, 255)  # white

            # Create the image segmenter
            segmenter = vision.ImageSegmenter.create_from_options(options)
            segmentation_result = segmenter.segment(image_)
            category_mask = segmentation_result.category_mask
            # Generate solid color images for showing the output segmentation mask.
            image_data = image_.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

            condition = np.stack(((category_mask.numpy_view() == 3),), axis=-1)
            if sum(sum(condition)).item() < 300:
                raise Exception("Error in segmentation model")
            segmented_image = cropped_image * condition
            segmented_aligned_frames.append(segmented_image)
        except Exception as e:
            print(f"Couldn't process frame {frame_count-1} with error {e}, file {video_path}")
            if frame is None:
                break
            else:
                # if we cant detect and align face then we replace it with a black image
                segmented_image = np.zeros([25, 25, 3], dtype=np.uint8)
                segmented_aligned_frames.append(segmented_image)

    for segmented_image in segmented_aligned_frames:
        # segmented_images.append(segmented_image)
        h, w, _ = segmented_image.shape
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w


        segmented_image_LAB = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2LAB)
        segmented_image_YUV = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2LUV)
        h_, w_ = h // 5, w // 5
        for sub_w in range(0, 5):
            for sub_h in range(0, 5):
                img_sub = segmented_image[sub_h*h_:(sub_h + 1)*h_, sub_w*w_:(sub_w+1)*w_, :]
                s_r, s_g, s_b = np.sum(img_sub[:, :, 0]), np.sum(img_sub[:, :, 1]), np.sum(img_sub[:, :, 2])
                if s_r > 0:
                    regions_avg_px_R[sub_h + sub_w*5].append(s_r / np.sum(img_sub[:, :, 0] > 0))
                else:
                    regions_avg_px_R[sub_h + sub_w * 5].append(0)
                if s_g > 0:
                    regions_avg_px_G[sub_h + sub_w * 5].append(s_g / np.sum(img_sub[:, :, 1] > 0))
                else:
                    regions_avg_px_G[sub_h + sub_w * 5].append(0)
                if s_b > 0:
                    regions_avg_px_B[sub_h + sub_w * 5].append(s_b / np.sum(img_sub[:, :, 2] > 0))
                else:
                    regions_avg_px_B[sub_h + sub_w * 5].append(0)

                img_sub = segmented_image_LAB[sub_h*h_:(sub_h + 1)*h_, sub_w*w_:(sub_w+1)*w_, :]
                s_l, s_a, s_b_ = np.sum(img_sub[:, :, 0]), np.sum(img_sub[:, :, 1]), np.sum(img_sub[:, :, 2])
                if s_l > 0:
                    regions_avg_px_L[sub_h + sub_w*5].append(s_l / np.sum(img_sub[:, :, 0] > 0))
                else:
                    regions_avg_px_L[sub_h + sub_w * 5].append(0)
                if s_a > 0:
                    regions_avg_px_A[sub_h + sub_w * 5].append(s_a / np.sum(img_sub[:, :, 1] > 0))
                else:
                    regions_avg_px_A[sub_h + sub_w * 5].append(0)
                if s_b_ > 0:
                    regions_avg_px_B_[sub_h + sub_w * 5].append(s_b_ / np.sum(img_sub[:, :, 2] > 0))
                else:
                    regions_avg_px_B_[sub_h + sub_w * 5].append(0)

                img_sub = segmented_image_YUV[sub_h*h_:(sub_h + 1)*h_, sub_w*w_:(sub_w+1)*w_, :]
                s_y, s_u, s_v = np.sum(img_sub[:, :, 0]), np.sum(img_sub[:, :, 1]), np.sum(img_sub[:, :, 2])
                if s_y > 0:
                    regions_avg_px_Y[sub_h + sub_w*5].append(s_y / np.sum(img_sub[:, :, 0] > 0))
                else:
                    regions_avg_px_Y[sub_h + sub_w * 5].append(0)
                if s_u > 0:
                    regions_avg_px_U[sub_h + sub_w * 5].append(s_u / np.sum(img_sub[:, :, 1] > 0))
                else:
                    regions_avg_px_U[sub_h + sub_w * 5].append(0)
                if s_v > 0:
                    regions_avg_px_V[sub_h + sub_w * 5].append(s_v / np.sum(img_sub[:, :, 2] > 0))
                else:
                    regions_avg_px_V[sub_h + sub_w * 5].append(0)

        # cv2.imshow("Segmented Image", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
        # cv2.imshow("Frame", frame)
        # # Enter key 'q' to break the loop
        # if cv2.waitKey(2) & 0xFF == ord('q'):
        #     break
    eps = 1e-10
    for sig in [regions_avg_px_R, regions_avg_px_G, regions_avg_px_B,
                regions_avg_px_Y, regions_avg_px_U, regions_avg_px_V,
                regions_avg_px_L, regions_avg_px_A, regions_avg_px_B_]:
        for k in sig:
            signal_ = np.array(sig[k])

            if len(signal_[signal_ > 0]) > 0:
                max_, min_ = max(signal_[signal_ > 0]), min(signal_[signal_ > 0])
                if max_ == min_:
                    signal_ = (signal_ - min_) / (max_ - min_ + eps)
                else:
                    signal_ = (signal_ - min_) / (max_ - min_)
                signal_[signal_ < 0] = 0
                sig[k] = list(signal_)
            else:
                sig[k] = signal_

    st_map_R = np.vstack([np.array(regions_avg_px_R[i]) for i in regions_avg_px_R])
    st_map_G = np.vstack([np.array(regions_avg_px_G[i]) for i in regions_avg_px_G])
    st_map_B = np.vstack([np.array(regions_avg_px_B[i]) for i in regions_avg_px_B])

    st_map_RGB = np.vstack((st_map_R[np.newaxis, :], st_map_G[np.newaxis, :], st_map_B[np.newaxis, :])).T
    np.save(os.path.join(dir_path, 'st_map_RGB_2.npy'), st_map_RGB)
    st_map_RGB = (st_map_RGB * 255).astype(np.uint8)
    img = Image.fromarray(st_map_RGB)
    img.save(os.path.join(dir_path, 'st_map_RGB_2.png'))


    st_map_Y = np.vstack([np.array(regions_avg_px_Y[i]) for i in regions_avg_px_Y])
    st_map_U = np.vstack([np.array(regions_avg_px_U[i]) for i in regions_avg_px_U])
    st_map_V = np.vstack([np.array(regions_avg_px_V[i]) for i in regions_avg_px_V])
    st_map_YUV = np.vstack((st_map_Y[np.newaxis, :], st_map_U[np.newaxis, :], st_map_V[np.newaxis, :])).T
    np.save(os.path.join(dir_path, 'st_map_YUV_2.npy'), st_map_YUV)
    st_map_YUV = (st_map_YUV * 255).astype(np.uint8)
    img = Image.fromarray(st_map_YUV)
    img.save(os.path.join(dir_path, 'st_map_YUV_2.png'))

    st_map_L = np.vstack([np.array(regions_avg_px_L[i]) for i in regions_avg_px_L])
    st_map_A = np.vstack([np.array(regions_avg_px_A[i]) for i in regions_avg_px_A])
    st_map_B_ = np.vstack([np.array(regions_avg_px_B_[i]) for i in regions_avg_px_B_])
    st_map_LAB = np.vstack((st_map_L[np.newaxis, :], st_map_A[np.newaxis, :], st_map_B_[np.newaxis, :])).T
    np.save(os.path.join(dir_path, 'st_map_LAB_2.npy'), st_map_LAB)
    st_map_LAB = (st_map_LAB * 255).astype(np.uint8)
    img = Image.fromarray(st_map_LAB)
    img.save(os.path.join(dir_path, 'st_map_LAB_2.png'))

    print(f"{video_path}, number of frames {frame_count-1}")
    # saving intermediate video requires padding which results in noise

    # video_name = "DnS_v.avi"
    # height, width, layers = max_h, max_w, 3
    #
    # video = cv2.VideoWriter(video_name, 0, fps, (width, height))
    # for frame in segmented_images:
    #     frame = resize_w_padding(frame)
    #     video.write(frame)
    # cv2.destroyAllWindows()
    # video.release()
    # print(video_name)


if __name__ == "__main__":
    # # COHFACE
    # files = glob.glob("D:\\anshul\\remoteHR\\4081054\\cohface\\**\\**\\*.avi")
    # pool = multiprocessing.Pool(4)
    # pool.map(process_video, files)

    # VIPL-V1
    files = glob.glob("D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\**\\**\\**\\*.avi")
    files = [f for f in files if "DnS_v" not in f]
    print(len(files))
    files_un_processed = []
    for f in files:
        dir_f = os.path.dirname(f)
        if not os.path.exists(os.path.join(dir_f, 'st_map_LAB_2.png')) and "source4" not in f:
            files_un_processed.append(f)
    # main('D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\p10\\v9\\source2\\video.avi')
    print(len(files_un_processed))
    pool = multiprocessing.Pool(5)
    pool.map(process_video, files_un_processed)
    # process_video("D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\p19\\v7\\source1\\video.avi")
