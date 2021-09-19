import numpy as np
import face_recognition
from src.config import get_algorithm_params
from src.utils import encodingsRead

FACE_AUTHENTICATOR = "FACE_AUTHENTICATOR"


class FaceAuthenticator:

    def __init__(self, encoding_path):

        # load the parameters
        self.params = get_algorithm_params(FACE_AUTHENTICATOR.lower())
        self.distances = []
        self.analysed_frames = 0
        self.unk_frames = 0
        self.saved_encodings = encodingsRead(encoding_path)

    def run(self, encoding):
        # compare the saved encoding and the new one, return the minimus distance between them
        recognised, dist = self.compareEncodings(self.saved_encodings, encoding['encodings'])

        feedback, final_decision = self.coreDecision(recognised, dist, len(self.saved_encodings))

        return feedback, final_decision

    def compareEncodings(self, saved_enc, unk_enc):
        """
        - unk_enc: SINGLE encoding of the person we want to recognize
        - saved_enc: dict list of encodings and name of the person of which we know the name-->data on our server
        - tolerance = max eculidean distance between two encoding to be recognised as the same

        Returns:
          name: True if recognized False otherwise
          dist:the min distance between the new face and the one saved if recognized, 1 otherwise
        """

        # initialize the list of names for each face detected
        dist = 1.0  # distance assigned to non recognized encodings

        # attempt to match each encoding to our known encodings global_number_enc : global parameter that allow us to
        # chose the number of known encoding on the server to consider in the comparison if it is <0 all the
        # encodings are taken
        ret = False

        data = saved_enc['encodings']
        # Compare a list of face encodings against a candidate encoding to see if they match
        # based on the tolerance parameter
        # Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        # for each comparison face.
        # The distance tells you how similar the faces are.
        face_distances = face_recognition.face_distance(data, unk_enc)

        if len(face_distances) > 0:  # if we don't have an empty result
            # Choose the known face with the smallest distance to the new face
            min_match_distance = np.amin(face_distances)

            if min_match_distance <= self.params['tolerance']:
                ret = True
                dist = min_match_distance

        return ret, dist

    def makeDecision(self, distances, n_frames, mod='avg'):
        """
        - distances: list of all the face distances so far
        - threshold: percentage of frames to be recognised to say True when distances don't works
        - n_frames: current frame number
        - mod : avg or min , takes the average or the minimum of the distances in order to give the answer
        - Return: True/false to say if the two people are the same
        """

        if len(distances) > 0:
            if mod == 'avg':
                min_d = np.mean(distances)
            else:
                if mod == 'min':
                    min_d = np.amin(distances)
                else:
                    print('Only avg or min supported the default will be used')
                    min_d = np.mean(distances)
            up = self.params['tolerance'] * (9 / 10)
            dists = np.linspace(0.2, up, 50)
            frames = np.linspace(int(self.params['min_frames_to_compare'] / 5),
                                 self.params['min_frames_to_compare'] * 2, 50)
            '''
            HAND-MADE VARIANT to explain how it ideally works
            if min_d <= 0.2:
              return True
            if min_d <= 0.25 and len(distances)>=(2/3)*min_frames_to_compare:
              return True
            if min_d <= 0.3 and len(distances)>=(3/4)*min_frames_to_compare:
              return True
            if min_d <= 0.35 and len(distances)>=(4/5)*min_frames_to_compare:
              return True
            if min_d <= 0.40 and len(distances)>=(5/6)*min_frames_to_compare:
              return True   
            if min_d <= 0.45 and len(distances)>=(6/7)*min_frames_to_compare:
              return True
            if min_d <= 0.50 and len(distances)>=(7/8)*min_frames_to_compare:
              return True
            '''
            # compact version of the above explanation
            for i in range(0, len(dists)):
                if min_d <= dists[i] and len(distances) >= frames[i]:
                    return True

        # we use again a threshold based decision when the previous logig doesn't work
        if len(distances) / n_frames > self.params['threshold']:
            return True

        return False

    def negativeEarlyStopping(self):
        # early stopping for negative result
        # After tot frames we try to see if we can give a negative answer before the end of the video
        # in particular we check if more than half of the frames weren't
        # recognised at all and if that is the case we can say that
        # the two person are different

        is_final_decision = False

        if self.analysed_frames >= self.params['min_frames_to_compare']:  # 45 frames = 1.5 second  -- early stopping
            if self.unk_frames / self.analysed_frames > 0.5:  # early stopping for the negative
                is_final_decision = True

        return is_final_decision

    def positiveEarlyStopping(self):
        # early stopping for positive result
        # starting from min_frames/5 every time we try to give an answer using the function make decision
        is_final_decision = False

        if self.analysed_frames >= self.params['min_frames_to_compare'] / 5:
            # ex 30fps --> 15 frames = 0.5 second  ---early stopping
            if self.makeDecision(self.distances, self.analysed_frames, mod=self.params['mod']):
                is_final_decision = True

        return is_final_decision

    def coreDecision(self, recognised, dist, length):

        is_final_decision = False
        final_answer = False
        # we had some cases where no faces where recognised
        if length <= 0:
            print('NO ENCODINGS SAVED , identification is not possible')
            return is_final_decision, final_answer

        self.analysed_frames += 1

        if recognised:
            self.distances.append(dist)
        else:
            self.unk_frames += 1

        if self.negativeEarlyStopping():
            is_final_decision = True
            final_answer = False

        elif self.positiveEarlyStopping():
            is_final_decision = True
            final_answer = True

        return is_final_decision, final_answer
