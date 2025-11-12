# cs767_cheque_mlp_app_tf.py
# MET CS 767 – Assignment 2 – Section 2.x
# Notebook 2 (converted to .py): Cheque-Digit Application Demo (FNN-only, TensorFlow)
# Author: <YOUR NAME>
# Usage: python cs767_cheque_mlp_app_tf.py
# Notes:
#   - Simulates a cheque workflow using per-digit MLP classification (no CNN/RNN).
#   - Trains an MLP on MNIST, applies light cheque-like augmentations during training.
#   - Demonstrates simple sequence assembly and confidence-based manual-review flag.

import numpy as np, tensorflow as tf, random, argparse
from tensorflow.keras import layers, models

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

@tf.function
def augment_batch(x):
    std = tf.random.uniform([], 0.0, 0.2)
    x = tf.clip_by_value(x + tf.random.normal(tf.shape(x), stddev=std), 0., 1.)
    c = tf.random.uniform([tf.shape(x)[0],1,1,1], 0.85, 1.15)
    x = tf.clip_by_value((x - 0.5)*c + 0.5, 0., 1.)
    return x

def ds(x, y, batch=128, train=True, aug=False):
    d = tf.data.Dataset.from_tensor_slices((x,y))
    if train: d = d.shuffle(4096, seed=SEED)
    d = d.batch(batch)
    if train and aug:
        d = d.map(lambda xi, yi: (augment_batch(xi), yi), num_parallel_calls=tf.data.AUTOTUNE)
    return d.prefetch(tf.data.AUTOTUNE)

def build_cheque_mlp():
    inputs = tf.keras.Input(shape=(28,28,1))
    z = layers.Flatten()(inputs)
    for units in (512, 256):
        z = layers.Dense(units, use_bias=False)(z)
        z = layers.BatchNormalization()(z)
        z = layers.Activation("relu")(z)
        z = layers.Dropout(0.3)(z)
    outputs = layers.Dense(10, activation="softmax")(z)
    return models.Model(inputs, outputs, name="cheque_mlp")

def predict_digits(model, crops):
    probs = model.predict(crops, verbose=0)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return pred, conf

def needs_review(confs, thr=0.90):
    return bool((confs < thr).any())

def make_sequence(x_pool, y_pool, n_digits=5, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_pool), size=n_digits, replace=False)
    imgs = x_pool[idx]
    labels = y_pool[idx]
    return imgs, labels

def main():
    print("TensorFlow version:", tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    # Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32")/255.0, -1)
    x_test  = np.expand_dims(x_test.astype("float32")/255.0, -1)

    from sklearn.model_selection import train_test_split
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )
    y_tr_oh  = tf.keras.utils.to_categorical(y_tr, 10)
    y_val_oh = tf.keras.utils.to_categorical(y_val, 10)
    y_test_oh= tf.keras.utils.to_categorical(y_test, 10)

    # Datasets
    train_ds = ds(x_tr, y_tr_oh, train=True, aug=True, batch=args.batch)
    val_ds   = ds(x_val, y_val_oh, train=False, aug=False, batch=args.batch)
    test_ds  = ds(x_test, y_test_oh, train=False, aug=False, batch=256)

    # Model
    model = build_cheque_mlp()
    steps = int(np.ceil(len(x_tr)/args.batch))
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-3, decay_steps=steps*args.epochs, alpha=0.1
    )
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs, verbose=2)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print({"test_acc": float(test_acc), "params": int(model.count_params())})

    # Demo: assemble 3 sequences
    for n in [3,4,6]:
        imgs, labels = make_sequence(x_test, y_test, n_digits=n, seed=SEED + n)
        preds, confs = predict_digits(model, imgs)
        gt = "".join(str(d) for d in labels)
        pr = "".join(str(d) for d in preds)
        print(f"GT={gt}  PRED={pr}  conf(min/avg/max)=({confs.min():.3f}/{confs.mean():.3f}/{confs.max():.3f})",
              " REVIEW" if needs_review(confs, 0.90) else "")

    # Single example flag
    imgs, labels = make_sequence(x_test, y_test, n_digits=5, seed=SEED + 99)
    preds, confs = predict_digits(model, imgs)
    print("Needs manual review:", needs_review(confs, 0.90))

if __name__ == "__main__":
    main()
