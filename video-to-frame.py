import cv2
import os
from os.path import join, exists
import handsegment as hs  # Make sure this module is available
from tqdm import tqdm

# Source and output folders
gesture_folder = "alltrain_videos"      # Source folder containing gesture videos
target_folder = "alltrain_frames"   # Output folder where extracted frames will be saved

hc = []  # List to hold frame metadata (optional)


def convert(gesture_folder, target_folder):
    rootPath = os.getcwd()
    majorData = os.path.abspath(target_folder)

    if not exists(majorData):
        os.makedirs(majorData)

    gesture_folder = os.path.abspath(gesture_folder)

    os.chdir(gesture_folder)
    gestures = os.listdir(os.getcwd())

    print("Source Directory containing gestures: %s" % (gesture_folder))
    print("Destination Directory containing frames: %s\n" % (majorData))

    for gesture in tqdm(gestures, desc="Processing Gestures", unit='gesture', ascii=True):
        gesture_path = os.path.join(gesture_folder, gesture)
        if not os.path.isdir(gesture_path):
            continue  # Skip files

        os.chdir(gesture_path)

        gesture_frames_path = os.path.join(majorData, gesture)
        if not os.path.exists(gesture_frames_path):
            os.makedirs(gesture_frames_path)

        videos = os.listdir(os.getcwd())
        videos = [video for video in videos if os.path.isfile(video)]

        for video in tqdm(videos, desc=f"Extracting {gesture}", unit='video', ascii=True):
            name = os.path.abspath(video)
            cap = cv2.VideoCapture(name)  # Load video
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            lastFrame = None

            os.chdir(gesture_frames_path)
            count = 0

            while count < 201:
                ret, frame = cap.read()
                if ret is False:
                    break

                framename = os.path.splitext(video)[0] + f"_frame_{count}.jpeg"
                hc.append([join(gesture_frames_path, framename), gesture, frameCount])

                if not os.path.exists(framename):
                    frame = hs.handsegment(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    lastFrame = frame
                    cv2.imwrite(framename, frame)

                count += 1

            # Pad with last frame if needed
            while count < 201:
                framename = os.path.splitext(video)[0] + f"_frame_{count}.jpeg"
                hc.append([join(gesture_frames_path, framename), gesture, frameCount])
                if not os.path.exists(framename) and lastFrame is not None:
                    cv2.imwrite(framename, lastFrame)
                count += 1

            os.chdir(gesture_path)
            cap.release()
            cv2.destroyAllWindows()

    os.chdir(rootPath)
    print("✅ Frame extraction complete.")


# Call the function with your source and output folders
if __name__ == '__main__':
    convert(gesture_folder, target_folder)