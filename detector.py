from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import pyautogui
import numpy as np
from keras_cv.models import YOLOV8Detector
from keras_cv import bounding_box
from keras_cv import visualization
import os

def get_test_data(directory="./test_data"):
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Filter out only .jpg images
    image_paths = [os.path.join(directory, f) for f in files if f.endswith(".jpg")]

    # Convert list to tf tensor
    image_paths_tensor = tf.convert_to_tensor(image_paths)

    # Create tf dataset
    data = tf.data.Dataset.from_tensor_slices(image_paths_tensor)

    return data

def visualize_detections(model, dataset, bounding_box_format):
    images = next(iter(dataset.take(1)))  # Only getting image paths, no y_true
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    
    # Using an empty array as placeholder for y_true
    y_true = [[] for _ in range(len(images))]
    
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7
    )


SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0

# Check TensorFlow Version
print("TensorFlow Version:", tf.__version__)

# Check for GPU availability
print("GPU Available:", tf.test.is_gpu_available())

# List the names of available GPUs
print("GPU Device Names:", tf.config.list_physical_devices('GPU'))


def start_detecting():
    # Load the model
    try:

# Save only architecture and weights

        # Load the model
        custom_objects = {'YOLOV8Detector': YOLOV8Detector}
        loaded_yolo = load_model('./goblin-detector-model_v1', custom_objects=custom_objects, compile=False)

        # Manually compile the model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            global_clipnorm=GLOBAL_CLIPNORM,
        )

        loaded_yolo.compile(
            optimizer=optimizer,
            classification_loss="binary_crossentropy",
            box_loss="ciou"
        )
        test_data = get_test_data()

        # Visualize detections on the test dataset
        visualize_detections(loaded_yolo, dataset=test_data, bounding_box_format="xyxy")
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Model could not be loaded.")
        return

   

    # Yu can use pyautogui.size() to get the screen size if needed
    # screen_width, screen_height = pyautogui.size()

    # while True:
    #     # Capture the screen
    #     screenshot = pyautogui.screenshot(region=(0, 0, screen_width, screen_height))
    #     frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    #     # Prepare the frame for the model
    #     input_frame = cv2.resize(frame, (640, 640))  # Assuming your YOLO model takes 416x416 images
    #     input_frame = np.expand_dims(input_frame, axis=0)

    #     # Predict
    #     y_pred_dict = loaded_yolo.predict(input_frame)

    #     # Get boxes, confidence, and classes from prediction
    #     boxes = y_pred_dict['boxes'][0]  # First index 0 assumes a batch size of 1
    #     confidence = y_pred_dict['confidence'][0]
    #     classes = y_pred_dict['classes'][0]

    #     # Filter out detections with confidence -1 or classes -1
    #     for box, conf, cls in zip(boxes, confidence, classes):
    #         if conf == -1 or cls == -1:
    #             continue

    #         # Draw rectangle
    #         x1, y1, x2, y2 = map(int, box)  # Convert to integers
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    #         # Add label
    #         cv2.putText(frame, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #     # Show the frame
    #     cv2.imshow('Object Detection', frame)

    #     # Break the loop if 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cv2.destroyAllWindows()



gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu}")
    start_detecting()
else:
    print("No GPUs are available!")

