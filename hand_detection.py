import cv2, random
import numpy as np
import mediapipe as mp
from boids import Boid
from scipy.spatial.transform import Rotation

DEBUG = False

class Hands:

    def __init__ (self, max_num_hands, min_detection_confidence, smoothing_alpha):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands = max_num_hands, min_detection_confidence = min_detection_confidence)
        self.finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        self.finger_inds = {name: list(range(4 * i + 1, 4 * i + 5)) for i, name in enumerate(self.finger_names)}
        self.img = None
        self.smoothed_angles, self.rot_offset = np.zeros(3), np.zeros(3)
        self.smoothed_alpha = smoothing_alpha
        self.landmarks = None
        self.boids = [Boid(random.randint(0, 1920), random.randint(0, 1080), 1920, 1080) for _ in range(100)]
    
    # Returns angle formed by 3 finger landmarks
    def calculate_finger_angle(self, lm1, lm2, lm3):
        lm1, lm2, lm3 = np.array(lm1), np.array(lm2), np.array(lm3)
        vec_21, vec_23 = lm1 - lm2, lm3 - lm2

        cos_angle = np.dot(vec_21, vec_23) / (np.linalg.norm(vec_21) * np.linalg.norm(vec_23))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return np.degrees(angle)
    
    # Returns whether each finger is up / down
    def fingers_up(self):
        fingers, landmarks = [], self.landmarks

        # Thumb logic
        fingers.append(int(landmarks[4].y < landmarks[3].y))

        # Other fingers
        for tip_id in [8, 12, 16, 20]:
            pip_id = tip_id - 2
            fingers.append(1 if landmarks[tip_id].y < landmarks[pip_id].y else 0)

        return fingers

    # Returns the angle formed by each finger
    def finger_curl_states(self):
        landmarks, statuses = self.landmarks, {}

        for name, ids in self.finger_inds.items():
            mcp = [landmarks[ids[0]].x, landmarks[ids[0]].y]
            pip = [landmarks[ids[1]].x, landmarks[ids[1]].y]
            dip = [landmarks[ids[2]].x, landmarks[ids[2]].y]
            tip = [landmarks[ids[3]].x, landmarks[ids[3]].y]

            angle1 = self.calculate_finger_angle(mcp, pip, dip)
            angle2 = self.calculate_finger_angle(pip, dip, tip)

            statuses[name] = (round((angle1 + angle2) / 2, 3))

        return statuses
    
    # Provide a smoothing buffer for the angles of the hands' movements
    def smooth_angles(self, angles):
        angles = np.array(angles)
        self.smoothed_angles = self.smoothed_alpha * angles + (1 - self.smoothed_alpha) * self.smoothed_angles
        return self.smoothed_angles

    # Returns the yaw, pitch, and roll of the hand
    def hand_rotation(self):
        lm = self.landmarks
        wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
        index_mcp = np.array([lm[5].x, lm[5].y, lm[5].z])
        pinky_mcp = np.array([lm[17].x, lm[17].y, lm[17].z])

        x_axis = index_mcp - wrist
        x_axis /= np.linalg.norm(x_axis)

        z_axis = np.cross(x_axis, pinky_mcp - wrist)
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        rot_mat = np.vstack([x_axis, y_axis, z_axis]).T
        r = Rotation.from_matrix(rot_mat)
        return self.smooth_angles(r.as_euler('xyz', degrees=True)) - self.rot_offset

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

    # Return the distnace between the thumb and index finger
    def thumb_index_dist(self):
        landmarks = self.landmarks
        h, w = self.img.shape[:2]

        thumb_px = np.array([int(landmarks[4].x * w), int(landmarks[4].y * h)])
        index_px = np.array([int(landmarks[8].x * w), int(landmarks[8].y * h)])
        pixel_dist = np.linalg.norm(index_px - thumb_px)

        all_points = np.array([[int(pt.x * w), int(pt.y * h)] for pt in landmarks])
        x_min, y_min = np.min(all_points, axis = 0)
        x_max, y_max = np.max(all_points, axis = 0)

        scale = np.hypot(x_max - x_min, y_max - y_min)
        norm_dist = pixel_dist / scale

        return norm_dist, pixel_dist, tuple(thumb_px), tuple(index_px)

    # Debug function to show additional data
    def debug(self, finger_states, curl_states, hand_rotation):
        for idx, name in enumerate(self.finger_names):
            up_down = 'Up' if finger_states[idx] else 'Down'
            curl = curl_states.get(name, "N/A")
            text = f'{name}: {up_down}, {curl}'
            cv2.putText(self.img, text, (10, 30 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            yaw, pitch, roll = hand_rotation
            rot_texts = [f"Yaw: {yaw:.2f}", f"Pitch: {pitch:.2f}", f"Roll: {2 * roll:.2f}"]
            for i, txt in enumerate(rot_texts):
                cv2.putText(self.img, txt, (10, 150 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            self.rotating_cube(hand_rotation)

            norm_dist, px_dist, thumb, index = self.thumb_index_dist()
            cv2.putText(self.img, f'Distance: {px_dist:.2f}px', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(self.img, f'Norm Distance: {norm_dist:.2f}px', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(self.img, f'Activation: {"Y" if norm_dist > 0.1 else "F"}', (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # Optional: draw a line between tips
            cv2.line(self.img, thumb, index, (255, 0, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(1)
        perception, norm_pinch, speed = 0, 0, 0
        center = np.array([0, 0])

        while True:
            ret, self.img = cap.read()
            if not ret: break

            self.img = cv2.flip(self.img, 1)
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(self.img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.landmarks = hand_landmarks.landmark

                    # Check fingers and joints
                    finger_states = self.fingers_up()
                    curl_states = self.finger_curl_states()
                    hand_rotation = self.hand_rotation()
                    
                    roll = hand_rotation[2]
                    norm_dist, _, _, _ = self.thumb_index_dist()
                    center = np.array([self.landmarks[9].x * int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), self.landmarks[9].y * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
                    speed = np.clip(np.interp(curl_states['Ring'], [90, 180], [1.0, 4.0]), 1.0, 4.0)
                    perception = np.clip(np.interp(roll, [-90, -90], [30, 150]), 30, 150)
                    norm_pinch = np.clip(np.interp(norm_dist, [0.05, 0.7], [0.0, 2.0]), 0.0, 2.0)

                    cv2.putText(self.img, f'Perception: {perception}', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 255), 2)
                    cv2.putText(self.img, f'Norm_P: {norm_pinch}', (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 255), 2)
                    cv2.putText(self.img, f'Speed: {speed}', (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 255), 2)

                    # Display debug text -> finger orientation and curl angle
                    global DEBUG
                    if DEBUG: self.debug(finger_states, curl_states, hand_rotation)

            for boid in self.boids:
                boid.perception = perception
                boid.max_speed = speed
                boid.flock(self.boids, align_w = norm_pinch)
                boid.attract(center, strength = 3)
                boid.update()
                boid.draw(self.img)

            cv2.imshow("Hand Finger Detection", self.img)

            # Press 'q' to quit, 'c' to recalibrate angle smoothing
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('d'): DEBUG = not DEBUG
            elif key == ord('c') and DEBUG: self.smoothed_angles = np.zeros(3)
            elif key == ord('r'):
                self.rot_offset = self.smoothed_angles.copy()
                print('Rotation zero point recalibrated')

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

h = Hands(1, 0.7, 0.2)
h.run()