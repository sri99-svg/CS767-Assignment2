# cs767_mlp_tuning_tf.py
# MET CS 767 – Assignment 2 – Section 1.x
# Notebook 1 (converted to .py): MLP on MNIST + Controlled Tuning (TensorFlow, FNN-only)
# Author: <YOUR NAME>
# Usage (Colab/Local): python cs767_mlp_tuning_tf.py
# Notes:
#   - Strictly feed-forward (Dense) layers; no CNN/RNN/etc.
#   - Enables controlled experiments: baseline vs. regularized vs. efficient + augmentation.
#   - Prints a compact summary table at the end.

import math, os, random, argparse
import numpy as np
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def stratified_subsample(x, y, per_class=1000, seed=SEED):
    """Pick per_class examples from each class for headroom in improvements."""
    idxs = []
    rng = np.random.default_rng(seed)
    for c in range(10):
        class_idx = np.where(y==c)[0]
        pick = rng.choice(class_idx, size=per_class, replace=False)
        idxs.append(pick)
    idxs = np.concatenate(idxs)
    return x[idxs], y[idxs]

@tf.function
def augment_batch(x):
    """Cheque-like noise/contrast jitter while keeping FNN legality."""
    std = tf.random.uniform([], 0.0, 0.25)
    x = tf.clip_by_value(x + tf.random.normal(tf.shape(x), stddev=std), 0., 1.)
    c = tf.random.uniform([tf.shape(x)[0],1,1,1], 0.8, 1.2)
    x = tf.clip_by_value((x - 0.5)*c + 0.5, 0., 1.)
    return x

def ds_pipeline(x, y, batch=128, train=True, aug=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch)
    if train and aug:
        ds = ds.map(lambda xi, yi: (augment_batch(xi), yi), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

from tensorflow.keras import layers, models

def build_baseline_mlp():
    inputs = tf.keras.Input(shape=(28,28,1))
    z = layers.Flatten()(inputs)
    z = layers.Dense(256, activation="relu")(z)
    outputs = layers.Dense(10, activation="softmax")(z)
    return models.Model(inputs, outputs, name="baseline_mlp")

def build_regularized_mlp():
    inputs = tf.keras.Input(shape=(28,28,1))
    z = layers.Flatten()(inputs)
    for units in (512, 256):
        z = layers.Dense(units, use_bias=False)(z)
        z = layers.BatchNormalization()(z)
        z = layers.Activation("relu")(z)
        z = layers.Dropout(0.3)(z)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(z)
    return models.Model(inputs, outputs, name="mlp_bn_dropout")

def build_efficient_mlp():
    inputs = tf.keras.Input(shape=(28,28,1))
    z = layers.Flatten()(inputs)
    z = layers.Dense(256, activation="gelu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(128, activation="gelu")(z)
    z = layers.Dropout(0.2)(z)
    outputs = layers.Dense(10, activation="softmax")(z)
    return models.Model(inputs, outputs, name="mlp_efficient_gelu")

def compile_and_fit(model, x_tr, y_tr_oh, x_val, y_val_oh, epochs=30, use_adamw=True, cosine=True, label_smooth=0.0, batch_size=128, aug=False):
    train_ds = ds_pipeline(x_tr, y_tr_oh, batch=batch_size, train=True, aug=aug)
    val_ds   = ds_pipeline(x_val, y_val_oh, batch=256, train=False, aug=False)

    if use_adamw:
        if cosine:
            steps_per_epoch = int(np.ceil(len(x_tr)/batch_size))
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=3e-3, decay_steps=steps_per_epoch*epochs, alpha=0.1
            )
            opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
        else:
            opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2)
    return hist

def evaluate_model(model, x_test, y_test_oh):
    test_ds = ds_pipeline(x_test, y_test_oh, batch=256, train=False, aug=False)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    params = model.count_params()
    return {"test_acc": float(test_acc), "params": int(params), "name": model.name}

def main():
    print("TensorFlow version:", tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=1000, help="Training examples per class (headroom).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")/255.0
    x_test  = x_test.astype("float32")/255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test,  -1)

    # Subsample for headroom
    x_train_sub, y_train_sub = stratified_subsample(x_train, y_train, per_class=args.per_class, seed=SEED)

    # Train/val split
    from sklearn.model_selection import train_test_split
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train_sub, y_train_sub, test_size=0.1, random_state=SEED, stratify=y_train_sub
    )
    y_tr_oh  = tf.keras.utils.to_categorical(y_tr, 10)
    y_val_oh = tf.keras.utils.to_categorical(y_val, 10)
    y_test_oh= tf.keras.utils.to_categorical(y_test, 10)

    # Baseline
    baseline = build_baseline_mlp()
    compile_and_fit(baseline, x_tr, y_tr_oh, x_val, y_val_oh, epochs=args.epochs, use_adamw=False, cosine=False, label_smooth=0.0, batch_size=args.batch, aug=False)
    res_base = evaluate_model(baseline, x_test, y_test_oh)
    print("Baseline:", res_base)

    # Regularized MLP (BN+Dropout, AdamW + cosine, label smoothing)
    mlpA = build_regularized_mlp()
    compile_and_fit(mlpA, x_tr, y_tr_oh, x_val, y_val_oh, epochs=args.epochs, use_adamw=True, cosine=True, label_smooth=0.05, batch_size=args.batch, aug=False)
    res_A = evaluate_model(mlpA, x_test, y_test_oh)
    print("MLP A (BN+Dropout):", res_A)

    # Efficient MLP (smaller, GELU)
    mlpB = build_efficient_mlp()
    compile_and_fit(mlpB, x_tr, y_tr_oh, x_val, y_val_oh, epochs=args.epochs, use_adamw=True, cosine=True, label_smooth=0.0, batch_size=args.batch, aug=False)
    res_B = evaluate_model(mlpB, x_test, y_test_oh)
    print("MLP B (efficient):", res_B)

    # Regularized + augmentation
    mlpA_aug = build_regularized_mlp()
    compile_and_fit(mlpA_aug, x_tr, y_tr_oh, x_val, y_val_oh, epochs=args.epochs, use_adamw=True, cosine=True, label_smooth=0.05, batch_size=args.batch, aug=True)
    res_A_aug = evaluate_model(mlpA_aug, x_test, y_test_oh)
    print("MLP A + Aug:", res_A_aug)

    # Summary table
    results = [res_base, res_A, res_B, res_A_aug]
    results_sorted = sorted(results, key=lambda r: r["test_acc"], reverse=True)

    print("\n=== Summary (sorted by test_acc) ===")
    print("{:<20} {:>10} {:>12}".format("Model", "Test Acc", "Params"))
    for r in results_sorted:
        print("{:<20} {:>10.4f} {:>12}".format(r["name"], r["test_acc"], r["params"]))

if __name__ == "__main__":
    main()
