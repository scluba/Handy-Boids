import cv2, random
import numpy as np
#import mediapipe as mp
from boids import Boid
from scipy.spatial.transform import Rotation

DEBUG = False
WIDTH, HEIGHT = None, None

class Hands:

    def __init__(self, max_num_hands, min_detection_confidence, smoothing_alpha):
        #self.mp_hands = mp.solutions.hands
        #self.mp_draw = mp.solutions.drawing_utils
        #self.hands = self.mp_hands.Hands(max_num_hands = max_num_hands, min_detection_confidence = min_detection_confidence)
        self.finger_names = {'Thumb': 1, 'Index': 2, 'Middle': 3, 'Ring': 4, 'Pinky': 5}
        self.finger_inds = {name: list(range(4 * i - 3, 4 * i + 1)) for name, i in self.finger_names.items()}
        self.smoothed_angles, self.rot_offset = np.zeros(3), np.zeros(3)
        self.smoothing_alpha = smoothing_alpha
        self.boids = [Boid(random.randint(0, 1920), random.randint(0, 1080), 1920, 1080) for _ in range(100)]
        self.img, self.landmarks = None, None

    # Return the appropriate coordinate of node
    def _node_point(self, node: int, frmt: str):
        match frmt:
            case 'x':
                return self.landmarks[node].x
            case 'y':
                return self.landmarks[node].y
            case 'z':
                return self.landmarks[node].z
            case 'xy':
                return np.array([self.landmarks[node].x, self.landmarks[node].y])
            case 'xyz':
                return np.array([self.landmarks[node].x, self.landmarks[node].y, self.landmarks[node].z])
            case _:
                raise LookupError('Requested node format unavailable: {frmt}')

    # Return the angle of the finger as formed by its nodes
    def calculate_finger_angle(self, finger: str):
        nodes = [self._node_point(i, 'xy') for i in self.finger_inds[finger]]
        tot_angle = 0

        for i in range(2):
            r_vec, t_vec = nodes[i] - nodes[i + 1], nodes[i + 2] - nodes[i + 1]
            angle = np.arccos(np.dot(r_vec, t_vec) / (np.linalg.norm(r_vec) * np.linalg.norm(t_vec)))
            tot_angle += np.degrees(angle)

        return round(tot_angle / 2, 3)
    
    # Return whether each finger is pointed up or down
    def finger_direction(self):
        fingers = []

        for nodes, ind in self.finger_inds.items():
            root, tip = nodes[1], nodes[3]
            fingers.append('Down' if self._node_point(root, 'y') < self._node_point(tip, 'y') else 'Up')
        
        return fingers
    
    # Return the yaw, pitch, and roll of the hand
    def hand_rotation(self):
        wrist = self._node_point(0, 'xyz')
        root_index = self._node_point(5, 'xyz')
        root_pinky = self._node_point(17, 'xyz')

        x_axis = root_index - wrist
        x_axis /= np.linalg.norm(x_axis)

        z_axis = np.cross(x_axis, root_pinky - wrist)
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
    
        rot_mat = np.vstack([x_axis, y_axis, z_axis]).T
        r = Rotation.from_matrix(rot_mat)
        return r.as_euler('xyz', degrees = True) - self.rot_offset
        
    # Draw a 2D visualization of cube rotated by the magnitude and direction of the wrist
    def rotating_cube(self, angles, center = (1600, 150), size = 100):
        cube = np.float32([[-1, -1, -1],
                           [-1, -1,  1],
                           [-1,  1, -1],
                           [-1,  1,  1],
                           [ 1, -1, -1],
                           [ 1, -1,  1],
                           [ 1,  1, -1],
                           [ 1,  1,  1]]) * (size / 2)
        

        mod_angles = [val if i != 2 else val * 2 for i, val in enumerate(angles)]
        r = Rotation.from_euler('xyz', mod_angles, degrees = True)
        rot_cube = r.apply(cube)

        points_2d = rot_cube[:, :2] + np.array(center)
        points_2d = points_2d.astype(int)

        edges = [(0, 1), (0, 2), (1, 3), (2, 3),
                 (4, 5), (4, 6), (5, 7), (6, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        
        for start, end in edges:
            cv2.line(self.img, tuple(points_2d[start]), tuple(points_2d[end]), (0, 255, 0), 2)

    # Return the distance between two fingertips
    def inter_finger_dist(self, finger_1: str, finger_2: str):
        root, tip = self.finger_inds[finger_1][-1], self.finger_inds[finger_2][-1]

        root *= [WIDTH, HEIGHT]
        tip *= [WIDTH, HEIGHT]
        dist = np.linalg.norm(tip - root)

        all_points = np.array([self._node_point(i, 'xy') * [WIDTH, HEIGHT] for name, i in self.finger_inds])
        x_min, y_min = np.min(all_points, axis = 0)
        x_max, y_max = np.max(all_points, axis = 0)

        scale = np.hypot(x_max - x_min, y_max - y_min)
        norm_dist = dist / scale

        return norm_dist, dist, tuple(root), tuple(tip)
    
    def run(self):
        cap = cv2.VideoCapture()
        global WIDTH, HEIGHT
        WIDTH, HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        perception, norm_pinch, speed = 0, 0, 0
        center = np.array([0, 0])

        while True:
            ret, self.img = cap.read()
            if not ret: break

            self.img = cv2.flip(self.img, 1)
            img_rbg = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rbg)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(self.img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.landmarks = hand_landmarks.landmarks

                    finger_states = self.finger_direction()
                    curl_states = {name: self.calculate_finger_angle(name) for name, i in self.finger_names.items()}
                    hand_rotation = self.hand_rotation()




h = Hands(1, 2, 3)
h.calculate_finger_angle('Thumb')