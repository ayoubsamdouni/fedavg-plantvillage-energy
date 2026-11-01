#!/usr/bin/env python3
"""
legacy_tools.py — checks and add-on experiments that reuse the ORIGINAL pipeline
(same image listing, same split seed 42, same partition code, same 3,000-image
client subsets, same model, same preprocessing) so they are directly comparable
to the results already in the paper. Nothing here retrains an FL scenario.

Subcommands
-----------
  params    : exact parameter count and per-client upload size
  freeze    : save the current Drive listing order + test-set hash (do this FIRST)
  verify    : re-evaluate saved final models on today's X_test and compare with the
              accuracy logged in each results CSV (=> proves the test set is unchanged)
  bncheck   : is the early "accuracy freeze" a BatchNorm-statistics artefact?
              evaluates a round-5 snapshot before/after re-estimating BN statistics
  timing    : per-epoch timing of one client update (E=3) — where do the ~120 s go?
  central   : centralised baseline on exactly the data the FL clients used

Examples (Colab)
----------------
  python legacy_tools.py params
  python legacy_tools.py freeze  --data /content/gdrive/MyDrive/color --out /content/gdrive/MyDrive/legacy
  python legacy_tools.py verify  --data ... --out ... \
        --pair /content/gdrive/MyDrive/fl_models_dirichlet_a05/global_model_round_30.keras:/content/gdrive/MyDrive/fl_results_dirichlet_a05.csv
  python legacy_tools.py bncheck --data ... --out ... \
        --model /content/gdrive/MyDrive/fl_models_dirichlet_a05_epoch1/global_model_round_5.keras
  python legacy_tools.py timing  --data ... --out ... --alpha iid --seed 42
  python legacy_tools.py central --data ... --out ... --epochs 90 --eval_every 3
"""
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SPLIT_SEED = 42
NUM_CLIENTS = 5
MAX_IMAGES_PER_CLIENT = 3000


# ------------------------------------------------ original data logic (verbatim)
def load_image_paths_and_labels(data_path):
    paths, labels = [], []
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                paths.append(os.path.join(class_path, img_file))
                labels.append(class_name)
    return np.array(paths), np.array(labels)


def partition_iid(paths, labels, num_clients, seed=42):
    rng = np.random.default_rng(seed)
    indices_by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    client_indices = [[] for _ in range(num_clients)]
    for c, idxs in indices_by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        for i, split in enumerate(np.array_split(idxs, num_clients)):
            client_indices[i].extend(split.tolist())
    out = []
    for idxs in client_indices:
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        out.append((paths[idxs], labels[idxs]))
    return out


def partition_dirichlet(paths, labels, num_clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        cutoffs = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        for i, split in enumerate(np.split(idxs, cutoffs)):
            client_indices[i].extend(split.tolist())
    out = []
    for idxs in client_indices:
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        out.append((paths[idxs], labels[idxs]))
    return out


def client_subset(paths, labels, cid, seed):
    """Exactly what _client_target did: the SAME subset every round."""
    if len(paths) > MAX_IMAGES_PER_CLIENT:
        rng = np.random.default_rng(seed + cid)
        idx = rng.choice(len(paths), size=MAX_IMAGES_PER_CLIENT, replace=False)
        return paths[idx], labels[idx]
    return paths, labels


def get_split(data_path, order_file=None):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    if order_file and os.path.exists(order_file):
        d = np.load(order_file, allow_pickle=False)
        image_paths, labels_raw = d["paths"], d["labels"]
        if data_path:   # allow moving the dataset (e.g. Drive -> local SSD)
            image_paths = np.array([os.path.join(data_path, os.path.relpath(p, str(d["root"])))
                                    for p in image_paths])
    else:
        image_paths, labels_raw = load_image_paths_and_labels(data_path)
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    Xtr, Xte, ytr, yte = train_test_split(image_paths, y, test_size=0.15,
                                          random_state=SPLIT_SEED, stratify=y)
    return Xtr, Xte, ytr, yte, le.classes_, image_paths, labels_raw


def test_hash(Xte, data_path):
    rel = [os.path.relpath(p, data_path) if data_path else p for p in Xte]
    return hashlib.sha1("\n".join(rel).encode()).hexdigest()[:12]


# ----------------------------------------------- original model + pipeline
def build_model(num_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    from tensorflow.keras.optimizers import Adam

    def CBAMLiteBlock(x, ratio=8):
        filters = x.shape[-1]
        avg_pool = layers.GlobalAveragePooling2D()(x)
        dense = layers.Dense(filters // ratio, activation='relu')(avg_pool)
        dense = layers.Dense(filters, activation='sigmoid')(dense)
        channel_att = layers.Multiply()([x, layers.Reshape([1, 1, filters])(dense)])
        spatial_avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(channel_att)
        spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_avg)
        return layers.Multiply()([channel_att, spatial])

    def InvertedResidualBlock(x, filters, expansion=4, strides=1, use_cbam=True):
        in_channels = x.shape[-1]
        expanded = layers.Conv2D(in_channels * expansion, 1, padding='same', use_bias=False)(x)
        expanded = layers.BatchNormalization()(expanded)
        expanded = layers.Activation('swish')(expanded)
        dw = layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False)(expanded)
        projected = layers.Conv2D(filters, 1, padding='same', use_bias=False)(dw)
        projected = layers.BatchNormalization()(projected)
        if use_cbam:
            projected = CBAMLiteBlock(projected)
        if strides == 1 and in_channels == filters:
            return layers.Add()([x, projected])
        return projected

    inputs = Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = InvertedResidualBlock(x, 32)
    x = InvertedResidualBlock(x, 64, strides=2)
    x = InvertedResidualBlock(x, 64)
    x = InvertedResidualBlock(x, 128, strides=2)
    x = InvertedResidualBlock(x, 128)
    x = InvertedResidualBlock(x, 256, strides=2)
    x = InvertedResidualBlock(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='swish')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    m = models.Model(inputs, outputs)
    m.compile(optimizer=Adam(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])
    return m


def make_dataset(paths, labels, num_classes, batch_size, shuffle=False, augment=False,
                 seed=SPLIT_SEED, cache=None):
    import tensorflow as tf

    def decode(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return tf.cast(img, tf.float32) / 255.0, label

    def aug(img, label):
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if cache is not None:
        # same numbers as the original pipeline, decoded once; shuffle after the cache
        ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).cache(cache)
        if shuffle:
            ds = ds.shuffle(buffer_size=min(1000, len(paths)), seed=seed)
    else:
        if shuffle:
            ds = ds.shuffle(buffer_size=min(1000, len(paths)), seed=seed)
        ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_into(model, path):
    """Load weights saved by the original scripts (.npz with w0..wN, or full .keras)."""
    if path.endswith(".npz"):
        d = np.load(path)
        keys = sorted(d.files, key=lambda k: int(k[1:]))
        model.set_weights([d[k] for k in keys])
        return
    try:
        model.load_weights(path)
        return
    except Exception:
        pass
    try:   # a .keras file is a zip that contains model.weights.h5
        tmp = path + ".weights.h5"
        with zipfile.ZipFile(path) as z, open(tmp, "wb") as f:
            f.write(z.read("model.weights.h5"))
        model.load_weights(tmp)
        os.remove(tmp)
        return
    except Exception:
        pass
    import tensorflow as tf
    import keras
    m = keras.models.load_model(path, safe_mode=False, compile=False, custom_objects={"tf": tf})
    model.set_weights(m.get_weights())


def eval_model(model, Xte, yte, num_classes):
    from sklearn.metrics import f1_score
    ds = make_dataset(Xte, yte, num_classes, 64)
    P = model.predict(ds, verbose=0)
    pred = P.argmax(1)
    return float((pred == yte).mean()), float(f1_score(yte, pred, average="macro")), pred


def _gpu():
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


# ------------------------------------------------------------- subcommands
def cmd_params(a):
    m = build_model(a.num_classes)
    ws = m.get_weights()
    tr = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    b = sum(w.nbytes for w in ws)
    print(json.dumps(dict(total_params=int(m.count_params()), trainable=tr,
                          non_trainable=int(m.count_params()) - tr,
                          upload_bytes_per_client=int(b),
                          upload_MB_per_client=round(b / 2 ** 20, 2),
                          upload_bytes_5_clients=int(5 * b)), indent=2))


def cmd_freeze(a):
    os.makedirs(a.out, exist_ok=True)
    Xtr, Xte, ytr, yte, classes, paths, labels_raw = get_split(a.data)
    f = os.path.join(a.out, "listing_order.npz")
    np.savez(f, paths=paths, labels=labels_raw, root=np.array(a.data))
    h = test_hash(Xte, a.data)
    sorted_same = bool(np.all(paths == np.array(sorted(paths, key=lambda p: (os.path.dirname(p), p)))))
    info = dict(n_images=int(len(paths)), n_train=int(len(Xtr)), n_test=int(len(Xte)),
                test_hash=h, listing_is_sorted=sorted_same, order_file=f)
    json.dump(info, open(os.path.join(a.out, "listing_info.json"), "w"), indent=2)
    print(json.dumps(info, indent=2))
    print("Run 'freeze' again in a NEW Colab session: if test_hash is identical, the Drive "
          "listing order is stable across sessions.")


def cmd_verify(a):
    import pandas as pd
    _gpu()
    Xtr, Xte, ytr, yte, classes, *_ = get_split(a.data, a.order)
    nc = len(classes)
    rows = []
    for pair in a.pair:
        mpath, cpath = pair.rsplit(":", 1)
        df = pd.read_csv(cpath)
        logged = float(df["accuracy"].iloc[-1])
        m = build_model(nc)
        load_into(m, mpath)
        acc, f1, pred = eval_model(m, Xte, yte, nc)
        np.save(os.path.join(a.out, os.path.basename(os.path.dirname(mpath)) + "_" +
                             os.path.basename(mpath) + ".preds.npy"), pred)
        ok = abs(acc - logged) < 1.5 / len(yte)      # within one image
        rows.append(dict(model=mpath, logged_round=int(df["round"].iloc[-1]),
                         logged_acc=logged, recomputed_acc=acc, macro_f1=f1, match=ok))
        print(f"{'OK  ' if ok else 'DIFF'} logged={logged:.4f} now={acc:.4f}  {mpath}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "verify_results.csv"), index=False)
    print(f"test_hash today = {test_hash(Xte, a.data)}")


def cmd_bncheck(a):
    import tensorflow as tf
    _gpu()
    Xtr, Xte, ytr, yte, classes, *_ = get_split(a.data, a.order)
    nc = len(classes)
    m = build_model(nc)
    load_into(m, a.model)
    acc0, f10, _ = eval_model(m, Xte, yte, nc)
    bns = [l for l in m.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    mov0 = [(float(np.mean(np.abs(l.moving_mean.numpy()))), float(np.mean(l.moving_variance.numpy())))
            for l in bns]
    # re-estimate BN running statistics from training images (no gradient step, weights unchanged)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), size=min(a.n_calib, len(Xtr)), replace=False)
    ds = make_dataset(Xtr[idx], ytr[idx], nc, BATCH_SIZE)
    for l in bns:
        l.momentum = 0.9
    for _ in range(a.passes):
        for xb, _ in ds:
            m(xb, training=True)
    acc1, f11, _ = eval_model(m, Xte, yte, nc)
    res = dict(model=a.model, acc_before=acc0, macro_f1_before=f10,
               acc_after_bn_recalibration=acc1, macro_f1_after=f11,
               first_bn_moving_mean_abs=mov0[0][0], first_bn_moving_var=mov0[0][1])
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(a.out, "bncheck_" + os.path.basename(os.path.dirname(os.path.abspath(a.model)))
                           + "_" + os.path.basename(a.model) + ".json"), "w"), indent=2)
    print("Interpretation: if accuracy jumps far above ~10% after recalibration, the "
          "'freeze' is a BatchNorm running-statistics effect, not a lack of learning signal.")


def cmd_timing(a):
    import tensorflow as tf
    _gpu()
    t_all = time.time()
    Xtr, Xte, ytr, yte, classes, *_ = get_split(a.data, a.order)
    t_list = time.time() - t_all
    nc = len(classes)
    parts = (partition_iid(Xtr, ytr, NUM_CLIENTS, seed=a.seed) if a.alpha == "iid" else
             partition_dirichlet(Xtr, ytr, NUM_CLIENTS, alpha=float(a.alpha), seed=a.seed))
    p, l = client_subset(*parts[a.client], a.client, a.seed)
    t0 = time.time()
    m = build_model(nc)
    t_build = time.time() - t0

    class EpochTimer(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.t = []
        def on_epoch_begin(self, e, logs=None):
            self.s = time.time()
        def on_epoch_end(self, e, logs=None):
            self.t.append(time.time() - self.s)

    cb = EpochTimer()
    ds = make_dataset(p, l, nc, BATCH_SIZE, shuffle=True, augment=True)
    t0 = time.time()
    m.fit(ds, epochs=a.epochs, verbose=0, callbacks=[cb])
    t_fit = time.time() - t0
    t1 = time.time()
    m.fit(ds, epochs=1, verbose=0)       # same data again: is it faster when files are cached?
    t_extra = time.time() - t1
    res = dict(n_images=int(len(p)), listing_and_split_sec=t_list, model_build_sec=t_build,
               fit_total_sec=t_fit, epoch_sec=cb.t, one_more_epoch_sec=t_extra)
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(a.out, "timing.json"), "w"), indent=2)
    print("If epoch 1 >> epochs 2-3, the fixed cost is first-epoch file I/O + graph tracing "
          "inside model.fit, not process spawning.")


def cmd_central(a):
    import pandas as pd
    _gpu()
    Xtr, Xte, ytr, yte, classes, *_ = get_split(a.data, a.order)
    nc = len(classes)
    if a.full:
        P, L = Xtr, ytr
        tag = "central_full"
    else:   # union of exactly the 5 IID client subsets used in S1 (same data budget as FL)
        parts = partition_iid(Xtr, ytr, NUM_CLIENTS, seed=SPLIT_SEED)
        subs = [client_subset(*parts[c], c, SPLIT_SEED) for c in range(NUM_CLIENTS)]
        P = np.concatenate([s[0] for s in subs]); L = np.concatenate([s[1] for s in subs])
        tag = "central_fl_subset"
    import tensorflow as tf
    tf.random.set_seed(a.seed)
    m = build_model(nc)
    cache = os.path.join(a.cache_dir, tag) if a.cache_dir else None
    if cache:
        os.makedirs(a.cache_dir, exist_ok=True)
    ds = make_dataset(P, L, nc, BATCH_SIZE, shuffle=True, augment=True, seed=a.seed, cache=cache)
    rows, done = [], 0
    csv = os.path.join(a.out, f"{tag}_s{a.seed}.csv")
    acc, f1, _ = eval_model(m, Xte, yte, nc)
    rows.append(dict(epoch=0, accuracy=acc, macro_f1=f1, train_sec=0.0))
    while done < a.epochs:
        k = min(a.eval_every, a.epochs - done)
        t0 = time.time()
        m.fit(ds, epochs=k, verbose=0)
        done += k
        acc, f1, pred = eval_model(m, Xte, yte, nc)
        rows.append(dict(epoch=done, accuracy=acc, macro_f1=f1, train_sec=time.time() - t0))
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"[{tag}] epoch {done}: acc={acc:.4f} macroF1={f1:.4f}", flush=True)
    np.save(csv.replace(".csv", "_preds.npy"), pred)
    m.save_weights(csv.replace(".csv", ".weights.h5"))
    print(f"n_train_images={len(P)}  saved {csv}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, data=True):
        if data:
            p.add_argument("--data", required=True)
            p.add_argument("--order", default="", help="listing_order.npz from 'freeze' (recommended)")
        p.add_argument("--out", default=".")
        p.add_argument("--num_classes", type=int, default=38)

    common(sub.add_parser("params"), data=False)
    common(sub.add_parser("freeze"))
    p = sub.add_parser("verify"); common(p)
    p.add_argument("--pair", action="append", required=True, help="model_path:results_csv")
    p = sub.add_parser("bncheck"); common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--n_calib", type=int, default=3200)
    p.add_argument("--passes", type=int, default=2)
    p = sub.add_parser("timing"); common(p)
    p.add_argument("--alpha", default="iid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--client", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p = sub.add_parser("central"); common(p)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--eval_every", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full", action="store_true", help="use the whole training split")
    p.add_argument("--cache_dir", default="/content/tfcache", help="'' to disable")
    a = ap.parse_args()
    if getattr(a, "order", None) == "":
        a.order = None
    if a.cmd != "params":
        os.makedirs(a.out, exist_ok=True)
    {"params": cmd_params, "freeze": cmd_freeze, "verify": cmd_verify, "bncheck": cmd_bncheck,
     "timing": cmd_timing, "central": cmd_central}[a.cmd](a)


if __name__ == "__main__":
    main()
