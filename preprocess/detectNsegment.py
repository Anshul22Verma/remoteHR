import cv2
import glob
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import multiprocessing
import numpy as np
import os


# Initializing the Model
model_path = "D:\\anshul\\rPPG\\mediapipe\\selfie_multiclass_256x256.tflite"
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
        try:
            y_min, y_max, x_min, x_max = get_bbox_from_mask(mask)
            if x_min == x_max or y_min == y_max:
                y_min, y_max, x_min, x_max = 0, og_image.shape[1], 0, og_image.shape[0]
        except:
            y_min, y_max, x_min, x_max = 0, og_image.shape[1], 0, og_image.shape[0]
        # y_min, y_max, x_min, x_max = get_bbox_from_mask(mask)

        cropped_segmented_image = og_image[y_min:y_max, x_min:x_max] * mask[y_min:y_max, x_min:x_max]
        h, w = cropped_segmented_image.shape[:2]
        img = cv2.resize(cropped_segmented_image, (DESIRED_WIDTH, DESIRED_HEIGHT))
        # print(img.shape)
    except:
        img = np.zeros((DESIRED_WIDTH, DESIRED_HEIGHT, 3), dtype="uint8")
    return img


def main(video_file: str):
    dir_name = os.path.dirname(video_file)
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

    if os.path.exists(os.path.join(dir_name, "DnS_v.avi")):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(base_options=base_options,
                                               output_category_mask=True)
        BG_COLOR = (0, 0, 0)  # black
        MASK_COLOR = (255, 255, 255)  # white

        # Create the image segmenter
        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            capture = cv2.VideoCapture(video_file)
            # 'D:\\anshul\\remoteHR\\mahnob\\Sessions\\103\\P1-Rec4-2009.07.09.18.57.48_C1 trigger _C_Section_3.avi'
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
                    try:
                        condition = np.stack(((category_mask.numpy_view() == 3),), axis=-1)
                        # print(condition.shape)
                    except:
                        condition = np.zeros(frame.shape)

                    # np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
                    # output_image = np.where(condition, fg_image, bg_image)

                    # print(f'Segmentation mask of frame {frame_count}:')
                    # resize_and_show(output_image)
                    cleaned_frames.append(detect_and_segment(frame, condition))
                    frame_count += 1
                except Exception as e:
                    print(f"Couldn't process frame {frame_count} with error {e}")
                    frame_count += 1
                    break

            if len(cleaned_frames) > 0:
                video_name = os.path.join(dir_name, "DnS_v.avi")
                height, width, layers = cleaned_frames[0].shape

                video = cv2.VideoWriter(video_name, 0, fps, (width, height))
                for frame in cleaned_frames:
                    video.write(frame)
                cv2.destroyAllWindows()
                video.release()
                print(video_name)
            else:
                print(f"Error with file {video_file}")


# mp_holistic = mp.solutions.holistic
# holistic_model = mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
#
# # Initializing the drawing utils for drawing the facial landmarks on image
# mp_drawing = mp.solutions.drawing_utils
# # (0) in VideoCapture is used to connect to your computer's default camera
# capture = cv2.VideoCapture(
# 'D:\\anshul\\remoteHR\\mahnob\\Sessions\\103\\P1-Rec4-2009.07.09.18.57.48_C1 trigger _C_Section_3.avi')
#
# # Initializing current time and precious time for calculating the FPS
# previousTime = 0
# currentTime = 0
#
# while capture.isOpened():
#     # capture frame by frame
#     ret, frame = capture.read()
#
#     # resizing the frame for better view
#     frame = cv2.resize(frame, (800, 600))
#
#     # Converting the from BGR to RGB
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Making predictions using holistic model
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = holistic_model.process(image)
#     image.flags.writeable = True
#
#     # Converting back the RGB image to BGR
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     # Drawing the Facial Landmarks
#     mp_drawing.draw_landmarks(
#         image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_CONTOURS,
#         mp_drawing.DrawingSpec(
#             color=(255, 0, 255),
#             thickness=1,
#             circle_radius=1
#         ),
#         mp_drawing.DrawingSpec(
#             color=(0, 255, 255),
#             thickness=1,
#             circle_radius=1
#         )
#     )
#
#     # Drawing Right hand Land Marks
#     mp_drawing.draw_landmarks(
#         image,
#         results.right_hand_landmarks,
#         mp_holistic.HAND_CONNECTIONS
#     )
#
#     # Drawing Left hand Land Marks
#     mp_drawing.draw_landmarks(
#         image,
#         results.left_hand_landmarks,
#         mp_holistic.HAND_CONNECTIONS
#     )
#
#     # Calculating the FPS
#     currentTime = time.time()
#     fps = 1 / (currentTime - previousTime)
#     previousTime = currentTime
#
#     # Displaying FPS on the image
#     cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#
#     # Display the resulting image
#     cv2.imshow("Facial and Hand Landmarks", image)
#
#     # Enter key 'q' to break the loop
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
#
# # When all the process is done
# # Release the capture and destroy all windows
# capture.release()
# cv2.destroyAllWindows()
#
# # Code to access landmarks
# for landmark in mp_holistic.HandLandmark:
#     print(landmark, landmark.value)
#
# print(mp_holistic.HandLandmark.WRIST.value)


if __name__ == "__main__":
    # # COHFACE
    # files = glob.glob("D:\\anshul\\remoteHR\\4081054\\cohface\\**\\**\\*.avi")
    # pool = multiprocessing.Pool(4)
    # pool.map(main, files)

    # VIPL-V1
    files = glob.glob("D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\**\\**\\**\\*.avi")
    files = [f for f in files if "DnS_v" not in f]
    # main('D:\\anshul\\remoteHR\\VIPL-HR-V1\\data\\p10\\v9\\source2\\video.avi')
    pool = multiprocessing.Pool(8)
    pool.map(main, files)

    # # MANHOB
    # files = glob.glob("D:\\anshul\\remoteHR\\mahnob\\Sessions\\*\\*.avi")
    # pool = multiprocessing.Pool(8)
    # pool.map(main, files)

    # # engagement-test
    # files = glob.glob("D:\\anshul\\engagement_test\\*\\*.mp4")
    # main("D:\\anshul\\engagement_test\\7\\subject_7_as2uk9lhe2_vid_0_18.mp4")
    # # pool = multiprocessing.Pool(4)
    # # pool.map(main, files)
