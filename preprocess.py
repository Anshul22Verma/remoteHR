import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time

# Initializing the Model
model_path = "D:\\anshul\\rPPG\\mediapipe\\selfie_multiclass_256x256.tflite"

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.png")

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)
'''
Input size 256 x 256

0 - background
1 - hair
2 - body-skin
3 - face-skin
4 - clothes
5 - others (accessories)
'''
DESIRED_WIDTH = 256
DESIRED_HEIGHT = 256


# Performs resizing and showing the image
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow("segmented image", img)
    cv2.waitKey(5)


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax  # y1, y2, x1, x2


# detect and segment for saving
def detect_and_segment(og_image, mask):
    try:
        y_min, y_max, x_min, x_max = get_bbox_from_mask(mask)

        cropped_segmented_image = og_image[y_min:y_max, x_min:x_max] * mask[y_min:y_max, x_min:x_max]
        h, w = cropped_segmented_image.shape[:2]
        # if h < w:
        #     img = cv2.resize(cropped_segmented_image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
        # else:
        #     img = cv2.resize(cropped_segmented_image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
        img = cv2.resize(cropped_segmented_image, (DESIRED_WIDTH, DESIRED_HEIGHT))
        print(img.shape)
    except:
        img = np.zeros((DESIRED_WIDTH, DESIRED_HEIGHT, 3), dtype="uint8")
    # cv2.imshow("segmented image", img)
    # cv2.waitKey(5)
    return img


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)
BG_COLOR = (0, 0, 0)  # gray
MASK_COLOR = (255, 255, 255)  # white

# Create the image segmenter
with vision.ImageSegmenter.create_from_options(options) as segmenter:
    capture = cv2.VideoCapture('D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\p17\\v1\\source3\\video.avi')
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = 1
    cleaned_frames = []
    while capture.isOpened():
        try:
            ret, frame = capture.read()
            frame = cv2.resize(frame, (256, 256))
            image_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(image)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask
            # Generate solid color images for showing the output segmentation mask.
            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

            # print(segmentation_result)
            # print(np.unique(category_mask.numpy_view()))

            condition = np.stack(((category_mask.numpy_view() == 3),), axis=-1)

            # np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)

            print(f'Segmentation mask of frame {frame_count}:')
            # resize_and_show(output_image)
            cleaned_frames.append(detect_and_segment(frame, condition))
            frame_count += 1
        except Exception as e:
            print(f"Couldn't process frame {frame_count} with error {e}")
            frame_count += 1
            break

    video_name = 'video.avi'
    height, width, layers = cleaned_frames[0].shape

    video = cv2.VideoWriter(video_name, 0, fps, (width, height))
    for frame in cleaned_frames:
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()



# import cv2
# import numpy as np
#
# # Create the image segmenter
# capture = cv2.VideoCapture(
#     'D:\\anshul\\remoteHR\\4081054\\cohface\\2\\0\\DnS_v.avi')
# fps = int(capture.get(cv2.CAP_PROP_FPS))
# frame_count = 1
# frames = []
# while capture.isOpened():
#     try:
#         ret, frame = capture.read()
#         frame = cv2.resize(frame, (256, 256))
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
#         frames.append(frame)
#         frame_count += 1
#     except Exception as e:
#         print(f"Coudnt process frame {frame_count} with error {e}")
#         frame_count += 1
#         break
#
# blocks = 5
# len = int(256/5)
#
# frames = np.array(frames)
# print(frames.shape)
#
# maps = []
# for frame in frames[:100]:
#     maps_frame = []
#     for r_idx in range(blocks):
#         for c_idx in range(blocks):
#             row_min, row_max = r_idx*len, min(256, (r_idx+1)*len)
#             col_min, col_max = c_idx*len, min(256, (c_idx+1)*len)
#             frames_roi = frame[row_min:row_max, col_min:col_max, :]
#             frames_roi = np.sum(frames_roi, axis=(0, 1)) / (np.shape(frames_roi)[0] * np.shape(frames_roi)[1])
#             maps_frame.append(frames_roi)
#     # # RGB ST-Maps
#     print(np.array(maps_frame).shape)
#     maps.append(np.array(maps_frame))
#     # frames_roi = np.average(frames_roi[np.nonzero(frames_roi)], axis=(1, 2))
#     # maps.append(frames_roi)
# maps = np.array(maps)
# print(maps.shape)
#
# # maps = maps.swapaxes(0, 1)
# import matplotlib.pyplot as plt
# plt.hist(maps.flatten())
# plt.waitforbuttonpress()
# plt.close()
#
# print(maps)
# plt.imshow(maps.astype(np.uint8))
# plt.waitforbuttonpress()
# print(maps.shape)

