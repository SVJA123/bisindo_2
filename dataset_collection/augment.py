import os
import cv2
import numpy as np
import random

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def add_noise(image, intensity=0.1):
    row, col, ch = image.shape
    mean = 0
    sigma = intensity**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def augment_image(image, num_new_sequences=1):
    augmented_images = []
    for aug_idx in range(num_new_sequences):
        angle = random.uniform(-45, 45)
        rotated_image = rotate_image(image, angle)

        noisy_image = add_noise(rotated_image, intensity=0.15)

        augmented_images.append(noisy_image)
    return augmented_images

def duplicate_sequences_with_augmentation(DATA_DIR, number_of_classes, num_new_sequences=1):
    for i in range(number_of_classes):
        label = chr(65 + i)
        class_dir = os.path.join(DATA_DIR, label)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            original_image = cv2.imread(img_path)
            
            augmented_images = augment_image(original_image, num_new_sequences)
            for aug_idx, aug_image in enumerate(augmented_images):
                new_file_name = img_file.replace('.', f'_aug_{aug_idx}.')
                save_path = os.path.join(class_dir, new_file_name)
                cv2.imwrite(save_path, aug_image)

def duplicate_single_subfolder(DATA_DIR, label, num_new_sequences=1):
    class_dir = os.path.join(DATA_DIR, label)
    images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in images:
        img_path = os.path.join(class_dir, img_file)
        original_image = cv2.imread(img_path)

        augmented_images = augment_image(original_image, num_new_sequences)
        for aug_idx, aug_image in enumerate(augmented_images):
            new_file_name = img_file.replace('.', f'_aug_{aug_idx}.')
            save_path = os.path.join(class_dir, new_file_name)
            cv2.imwrite(save_path, aug_image)
