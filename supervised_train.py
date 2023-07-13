import os
from io import BufferedReader
import pickle

import numpy as  np
import tensorflow as tf
from keras import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, MSE

from prob_reversi import Position, DiscColor
from dualnet import position_to_input, dual_network, DN_INPUT_SHAPE, DN_OUTPUT_SIZE

BOARD_SIZE = 6
CHANNEL_NUM = DN_INPUT_SHAPE[2]
TRANS_PROB = []

t = 5
for coord in range(BOARD_SIZE ** 2):
    TRANS_PROB.append(1.0 - t * 0.01 * (coord % (BOARD_SIZE + 1) + 3))

EPOCH = 1
BATCH_SIZE = 512
NUM_CACHED_BATCH = 1000

MODEL_PATH = "model_6x6.h5"
TRAIN_DATA_PATH = "train_data_6x6.pickle"
LOSS_HISTORY_PATH = "pv_loss.txt"

loss_history = []

def load_batches(file: BufferedReader) -> list[list[(Position, int, float)]]:
    batches = []
    for i in range(NUM_CACHED_BATCH):
        batch = []
        for j in range(BATCH_SIZE):
            try:
                bb, coord, reward = pickle.load(file)
                pos = Position(BOARD_SIZE, TRANS_PROB)
                pos.set_state(bb[0], bb[1], DiscColor.BLACK)
                batch.append((pos, coord, reward))
            except EOFError:
                break
        
        if len(batch) != 0:
            batches.append(batch)

    return batches

if os.path.exists(MODEL_PATH):
    model: Model = load_model(MODEL_PATH)
else:
    model: Model = dual_network()
    model.compile(optimizer=Adam(learning_rate=0.01), loss=[CategoricalCrossentropy(), MSE])
    model.save(MODEL_PATH)

for epoch in range(EPOCH):
    with open(TRAIN_DATA_PATH, mode="rb") as file:
        num_batches = 0
        x = np.empty(shape=(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, CHANNEL_NUM)).astype(np.float32)
        value_target = np.empty(shape=(BATCH_SIZE, 1)).astype(np.float32)

        while True:
            batches = load_batches(file)
            if len(batches) == 0:
                break

            while len(batches) != 0:
                print(f"batch_id: {num_batches}")
                batch = batches.pop()
                batch_size = len(batch)

                if batch_size != BATCH_SIZE:
                    x.fill(0.0)
                    value_target.fill(0.0)

                policy_traget = tf.one_hot(list(map(lambda x: x[1], batch)), DN_OUTPUT_SIZE)
                
                for i, (pos, _, reward) in enumerate(batch):
                    position_to_input(pos, x[i])
                    value_target[i] = reward
                
                loss = model.train_on_batch(x, y=[policy_traget, value_target])
                loss_history.append(str(loss))
                print(f"epoch = {epoch + 1}")
                print(f"policy_loss: {loss[0]}, value_loss: {loss[1]}")
                num_batches += 1

            tf.keras.backend.clear_session()

    model.save(MODEL_PATH)

    with open(LOSS_HISTORY_PATH, mode="w") as file:
        file.write(str(loss_history))
    
            
            



