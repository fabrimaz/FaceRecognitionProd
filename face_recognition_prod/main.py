import cv2

from src.utils import compress
from src.encodings import FaceEncoder
from src.authenticator import FaceAuthenticator

WINDOW_NAME = "Face Authentication"

PRECOMPUTED_ENCS = False


def run():
    # init video feed
    # cv2.namedWindow(WINDOW_NAME)
    # capture the video from the web-cam
    video_capture = cv2.VideoCapture(0)

    # or like this if we want use a saved video
    # video_capture = cv2.VideoCapture(video path)

    # ---------------------------------------------------------#
    ref_path = 'path of the reference encodings'
    input_path = "path of the of the input encodings if PRECOMPUTED is True"
    face_encoder = FaceEncoder(PRECOMPUTED_ENCS, input_path)
    face_authenticator = FaceAuthenticator(ref_path)
    # ---------------------------------------------------------#

    while True:
        if not PRECOMPUTED_ENCS:
            ret, frame = video_capture.read(0)
            frame = compress(frame, 2)  # to make it run faster
        else:
            ret = True
            frame = None

        if not ret:
            break

        # run the face tracker
        face_detected, encoding = face_encoder.run(frame)
        # if no encodings are detected use the next frame
        if face_detected:
            # run face authenticator
            stop_recognition, final_answer = face_authenticator.run(encoding)

            # if a final answer is given stop the recognition
            if stop_recognition:
                # True or False on the fact that the person is recognized or not
                print(final_answer)
                break

        # show frames
        if not PRECOMPUTED_ENCS:
            cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
