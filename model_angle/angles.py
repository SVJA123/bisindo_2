import numpy as np

def calculate_adjacent_angles(landmarks):
    hand1 = landmarks[:21] 
    hand2 = landmarks[21:]

    def calculate_adjacent_angles_for_hand(hand_landmarks):
        wrist = hand_landmarks[0]
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        pinky_tip = hand_landmarks[20]

        if np.all(wrist == 0) or np.all(thumb_tip == 0) or np.all(index_tip == 0) or np.all(middle_tip == 0) or np.all(ring_tip == 0) or np.all(pinky_tip == 0):
            return np.zeros(4) 

        v_thumb = thumb_tip - wrist
        v_index = index_tip - wrist
        v_middle = middle_tip - wrist
        v_ring = ring_tip - wrist
        v_pinky = pinky_tip - wrist

        vectors = [v_thumb, v_index, v_middle, v_ring, v_pinky]

        angles = []
        for i in range(len(vectors) - 1):
            dot_product = np.dot(vectors[i], vectors[i + 1])
            norm_i = np.linalg.norm(vectors[i])
            norm_j = np.linalg.norm(vectors[i + 1])

            if norm_i == 0 or norm_j == 0:
                angles.append(0.0) 
            else:
                cos_theta = dot_product / (norm_i * norm_j)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta)  
                angles.append(np.degrees(angle)/ 180) 
        return np.array(angles)

    angles_hand1 = calculate_adjacent_angles_for_hand(hand1)
    angles_hand2 = calculate_adjacent_angles_for_hand(hand2)

    angles = np.concatenate([angles_hand1, angles_hand2])

    return angles