import cv2
import numpy as np
import face_recognition
# from imutils import face_utils
# import dlib
# from src.config import get_algorithm_params
# from src.utils import find_best_bounding_box
from src.utils import encodingsRead

METHOD = 'cnn'


class FaceEncoder:

    def __init__(self, precomputed, input_path):
        # defines the method used
        self.detection = METHOD
        # this is true is encodings already exist
        self.precomputed_encs = precomputed
        if self.precomputed_encs:
            # initialise the counter
            self.counter = 0
            # set the path where to take the encoding
            self.input_path = input_path
            # reads the encoding
            self.encodings = encodingsRead(self.input_path)

    def assign_landmark_to_box(self, landmarks_pos, boxes):
        """
        Matches the box and landmark, analysing if
        the key face features are all positioned inside the box
        """
        for box in boxes:
            # extract box corners
            bottom, right, top, left = box
            # initialise the counter for each box
            confirmed_box = 0
            for feature_pos in landmarks_pos:
                # extract the feature averaged position
                mean_w, mean_h = feature_pos
                # states if the feature position is inside the face box corners
                if (bottom < mean_h < top
                        and left < mean_w < right):
                    # if the feature is inside we increae the counter
                    confirmed_box = confirmed_box + 1

            # if every feature is inside the box there is a match
            if confirmed_box == len(landmarks_pos):
                print("0k")
                return box

        return None

    def filter_front_facing(self, frame, boxes, landmarks):
        # defining array where to store boxes where we had front-facing pictures
        new_boxes = []

        print(len(boxes), len(landmarks))

        # landmark and box are independent each other
        for landmark in landmarks:
            landmark_avg = []

            # Taking important key-feature in the person's face
            # We chose these ones because they represent upper, center and lower part
            nose_bridge = landmark['nose_bridge']
            bottom_lip = landmark['bottom_lip']
            right_eyebrow = landmark['right_eyebrow']

            # Computing the average of taken points relative to the part of the face
            nose_bridge = [sum(y) / len(y) for y in zip(*nose_bridge)]
            bottom_lip = [sum(y) / len(y) for y in zip(*bottom_lip)]
            right_eyebrow = [sum(y) / len(y) for y in zip(*right_eyebrow)]

            # Concatenating average points in the averaged array
            landmark_avg.append(nose_bridge)
            landmark_avg.append(bottom_lip)
            landmark_avg.append(right_eyebrow)

            # check the compatibility between the box and the landmark
            box = self.assign_landmark_to_box(landmark_avg, boxes)
            # every confirmed box is added to the nex_box array
            if box is not None:
                new_boxes.append(box)

        return new_boxes

    def find_biggest_box(self, frame, boxes):
        # define ratios array 
        ratios = []
        # define new box array
        main_box_list = []

        # compute frame area
        area_image = frame.shape[0] * frame.shape[1]

        for face_location in boxes:
            # extract coordinates from the box
            top, right, bottom, left = face_location
            # compute face area
            area_face = (bottom - top) * (right - left)
            # compute ratio wrt the frame
            ratio = area_face / area_image * 100
            # add value to the array
            ratios.append(ratio)

        # selection of the biggest face in the picture
        if ratios:
            # select the biggest ratio (biggest face)
            main_box_index = np.argmax(ratios)
            # select the corresponding box
            main_box = boxes[main_box_index]
            # append this box to the new box list
            main_box_list.append(main_box)

        return main_box_list

    def filter_found_faces(self, frame, boxes, landmarks):
        # initialise result array
        main_box = []

        print(len(boxes), len(landmarks))

        if boxes and landmarks:
            # finds the biggest face box in the picture
            main_box = self.find_biggest_box(frame, boxes)
            # matches the given face box with the landmarks detected
            main_box = self.filter_front_facing(frame, main_box, landmarks)

        # returns the main face box in the frame or an empty list
        return main_box

    def create_and_format_encoding(self, frame, main_box):
        # initialise the false feedback
        feedback = False

        if main_box:
            # if there is a box and a related landmark we will have an encoding
            feedback = True
            # takes the first enconding from the list (we always have one box here)
            encoding = face_recognition.face_encodings(frame, main_box)[0]
            # encoding is formatted as a dictionary
            enc_dict = {"encodings": encoding}
            # returns the feedback and the dictionary
            return feedback, enc_dict

        return feedback, None

    def run(self, frame):
        # if we are working with already create encodings
        if self.precomputed_encs:
            feedback, best_encoding = self.encoding_fetcher()

        elif not self.precomputed_encs:
            # convert the frame in the RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # estimate landmark to understand face position
            landmarks = face_recognition.face_landmarks(frame)
            # detect the (x, y)-coordinates of the bounding boxes corresponding to each face
            boxes = face_recognition.face_locations(frame, model=self.detection)
            # look for main face in the picture that is front-facing
            main_box = self.filter_found_faces(frame, boxes, landmarks)
            # compute the facial embedding on the main face box found
            feedback, best_encoding = self.create_and_format_encoding(frame, main_box)

        return feedback, best_encoding

    def encoding_fetcher(self):
        # initialise return values
        feedback = False
        enc_dict = None

        # if there are still enough encodings
        if self.counter < len(self.encodings['encodings']):
            feedback = True
            # takes the already filtered encoding and increases the counter
            encoding = self.encodings['encodings'][self.counter]
            self.counter += 1
            # encoding is formatted as a dictionary
            enc_dict = {"encodings": encoding}

        return feedback, enc_dict
