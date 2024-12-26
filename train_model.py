import os
import cv2
import numpy as np
import face_recognition
import pickle
from sklearn.datasets import fetch_lfw_people

# Directories for local images and LFW dataset
LOCAL_IMAGES_DIR = "data/faces/"
LFW_DATASET_DIR = "data/dataset/"

# Model file path
MODEL_FILE = "data/trained_model.pkl"

def load_lfw_dataset():
    """
    Load the Labeled Faces in the Wild (LFW) dataset.
    """
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people.images, lfw_people.target, lfw_people.target_names

def process_local_images():
    """
    Process the local images in the 'data/faces' directory.
    """
    known_encodings = []
    known_names = []

    # Loop through each person in the dataset folder
    for person_name in os.listdir(LOCAL_IMAGES_DIR):
        person_path = os.path.join(LOCAL_IMAGES_DIR, person_name)

        if not os.path.isdir(person_path):
            continue

        # Loop through each image of the person
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)

            # Load the image and get face encodings
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) > 0:
                # Use the first encoding (if more than one face is detected)
                known_encodings.append(face_encodings[0])
                known_names.append(person_name)

    return known_encodings, known_names

def train_face_recognition_model():
    """
    Train the face recognition model using both local images and the LFW dataset.
    """
    known_encodings = []
    known_names = []

    # Process local images
    local_encodings, local_names = process_local_images()
    known_encodings.extend(local_encodings)
    known_names.extend(local_names)

    # Load the LFW dataset
    print("Loading LFW dataset...")
    lfw_images, lfw_target, lfw_target_names = load_lfw_dataset()

    # Process each image in the LFW dataset
    for i in range(len(lfw_images)):
        image = lfw_images[i]
        encoding = face_recognition.face_encodings(image)

        if len(encoding) > 0:
            known_encodings.append(encoding[0])
            known_names.append(lfw_target_names[lfw_target[i]])

    # Save the encodings and names to a file using pickle
    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump({"encodings": known_encodings, "names": known_names}, model_file)

    print(f"Model trained and saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_face_recognition_model()
