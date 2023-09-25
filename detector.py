from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import pyautogui
import numpy as np
from keras_cv.models import YOLOV8Detector


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
        loaded_yolo = load_model('./goblin-detector-modelv1', custom_objects=custom_objects, compile=False)

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

    except KeyError as e:
        print(f"KeyError: {e}")
        print("Model could not be loaded.")
        return

    # You can use pyautogui.size() to get the screen size if needed
    screen_width, screen_height = pyautogui.size()

    while True:
        # Capture the screen
        screenshot = pyautogui.screenshot(region=(0, 0, screen_width, screen_height))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Prepare the frame for the model
        input_frame = cv2.resize(frame, (640, 640))  # Assuming your YOLO model takes 416x416 images
        input_frame = np.expand_dims(input_frame, axis=0)

        # Predict
        y_pred_dict = loaded_yolo.predict(input_frame)

        # Get boxes, confidence, and classes from prediction
        boxes = y_pred_dict['boxes'][0]  # First index 0 assumes a batch size of 1
        confidence = y_pred_dict['confidence'][0]
        classes = y_pred_dict['classes'][0]

        # Filter out detections with confidence -1 or classes -1
        for box, conf, cls in zip(boxes, confidence, classes):
            if conf == -1 or cls == -1:
                continue

            # Draw rectangle
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(frame, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show the frame
        cv2.imshow('Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print(f"Num GPUs Available: {len(gpus)}")
#     for gpu in gpus:
#         print(f"GPU: {gpu}")
#     start_detecting()
# else:
#     print("No GPUs are available!")

start_detecting()