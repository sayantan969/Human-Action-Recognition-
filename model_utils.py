import os
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model , regularizers ,  Input,models
from tensorflow.keras.layers import Conv2D , MaxPooling2D, Flatten, Dense, Dropout , Input, TimeDistributed, LSTM, Dropout , Bidirectional,Concatenate, BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2 , ResNet50 , EfficientNetV2S ,EfficientNetV2B0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision
from glob import glob


FRAME_SIZE     = (224, 224)
K              = 80        # number of frames per clip
BATCH_SIZE     = 64

# 50 selected classes
SELECTED_CLASSES = [
    'ApplyEyeMakeup','ApplyLipstick','Archery','BabyCrawling','BalanceBeam',
    'BandMarching','Basketball','BasketballDunk','BenchPress','Biking',
    'Billiards','BlowDryHair','BlowingCandles','BodyWeightSquats','Bowling',
    'BoxingPunchingBag','BoxingSpeedBag','BreastStroke','BrushingTeeth','CleanAndJerk',
    'CliffDiving','CricketBowling','CricketShot','CuttingInKitchen','Diving',
    'Drumming','Fencing','FieldHockeyPenalty','FloorGymnastics','FrisbeeCatch',
    'FrontCrawl','GolfSwing','Haircut','Hammering','HandstandPushups',
    'HandstandWalking','HeadMassage','HighJump','HulaHoop','IceDancing',
    'JavelinThrow','JumpRope','JumpingJack','Kayaking','Knitting',
    'LongJump','Lunges','MilitaryParade','Mixing','MoppingFloor'
]
NUM_CLASSES = len(SELECTED_CLASSES)
selected_label2index = {c: i for i, c in enumerate(SELECTED_CLASSES)}






def build_deep_cnn_lstm_model(
    input_shape=(K, FRAME_SIZE[0], FRAME_SIZE[1], 3),
    num_classes=NUM_CLASSES,
    base_model_trainable=False,
    dropout_rate=0.5,
    lstm_dropout=0.4,
    l2_strength=5e-4,
    learning_rate=1e-4
):
    frames_input = Input(shape=input_shape, name="input_frames")


    base_cnn = MobileNetV2(
        input_shape=input_shape[1:], # (H, W, C)
        include_top=False,
        weights="imagenet",
        pooling=None
    )
    base_cnn.trainable = base_model_trainable


    x = TimeDistributed(base_cnn, name="time_distributed_cnn")(frames_input)

    x = TimeDistributed(GlobalAveragePooling2D(), name="time_distributed_pooling")(x)
    #if base_model_trainable:
    x = TimeDistributed(BatchNormalization(), name="time_distributed_bn")(x)
    x = TimeDistributed(Dropout(dropout_rate), name="time_distributed_dropout")(x)


    x = Bidirectional(LSTM(512, return_sequences=True, dropout=lstm_dropout), name="bilstm_1")(x)
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=lstm_dropout), name="bilstm_2")(x)
    x = Bidirectional(LSTM(128, return_sequences=False, dropout=lstm_dropout), name="bilstm_3")(x)


    x = Dense(512, activation="relu",kernel_regularizer=regularizers.l2(l2_strength), name="dense_1")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(dropout_rate, name="dropout_1")(x)

    x = Dense(256, activation="relu",kernel_regularizer=regularizers.l2(l2_strength), name="dense_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)

    outputs = Dense(num_classes, activation="softmax", dtype="float32",kernel_regularizer=regularizers.l2(l2_strength), name="output_softmax")(x)


    model = models.Model(inputs=frames_input, outputs=outputs)


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model

print("\n--- Building Model ---")
model = build_deep_cnn_lstm_model(
    input_shape=(K, FRAME_SIZE[0], FRAME_SIZE[1], 3),
    num_classes=NUM_CLASSES,
    base_model_trainable=False,
)



MODEL_PATH = 'best_model_combined (4).keras'

def predict_action_from_video(video_path):
    # 1. Read all frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        print(f"❌ No frames extracted from video: {video_path}")
        return None

    # 2. Compute motion between frames
    motion = [0.0]
    for i in range(1, len(frames)):
        prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        diff = np.sum(np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32)))
        motion.append(diff)

    # 3. Select top-k motion window
    if len(frames) < K:
        frames += [frames[-1]] * (K - len(frames))  # Pad
    else:
        motion_sum = np.convolve(motion, np.ones(K, dtype=np.float32), mode='valid')
        start_idx = int(np.argmax(motion_sum))
        frames = frames[start_idx:start_idx + K]

    # 4. Normalize and stack
    video_tensor = np.array(frames, dtype=np.float32) / 255.0  # (K, H, W, 3)
    video_tensor = np.expand_dims(video_tensor, axis=0)  # (1, K, H, W, 3)

    # 5. Load model and predict
    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict(video_tensor)
    class_id = np.argmax(preds[0])
    confidence = np.max(preds[0])

    num_frames_to_display = 5  # Adjust as needed
    interval = len(frames) // num_frames_to_display

    # plt.figure(figsize=(15, 5))
    # for i in range(num_frames_to_display):
    #     plt.subplot(1, num_frames_to_display, i + 1)
    #     plt.imshow(frames[i * interval])
    #     plt.axis('off')
    # plt.suptitle(f"Predicted: {SELECTED_CLASSES[class_id]} (confidence: {confidence:.4f})", fontsize=16)
    # plt.show()

    print(f"🎯 Predicted: {SELECTED_CLASSES[class_id]} (confidence: {confidence:.4f})")
    return SELECTED_CLASSES[class_id], confidence


