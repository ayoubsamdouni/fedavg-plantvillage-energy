#!/usr/bin/env python3
"""
edge_profile.py — deployment-cost profile of the paper's model (same architecture as the original scripts).

  python edge_profile.py profile --out OUT [--weights global_weights.npz]
      parameters, FLOPs per image, Keras/TFLite file sizes, CPU latency (batch 1, 1 thread and all threads)
  python edge_profile.py tflite_acc --out OUT --weights W --data DATA --order listing_order.npz
      accuracy of the float32 and int8 (dynamic-range) TFLite models on the full held-out test set

Run on a CPU-only Colab runtime (no compute units needed).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _model(weights, n_classes=38):
    import legacy_tools as L
    m = L.build_model(n_classes)
    if weights:
        L.load_into(m, weights)
    return m


def flops_per_image(m):
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    f = tf.function(lambda x: m(x, training=False))
    cf = f.get_concrete_function(tf.TensorSpec([1, 224, 224, 3], tf.float32))
    frozen = convert_variables_to_constants_v2(cf)
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts['output'] = 'none'
    info = tf.compat.v1.profiler.profile(graph=frozen.graph, run_meta=tf.compat.v1.RunMetadata(), cmd='op', options=opts)
    return int(info.total_float_ops)          # counts multiply and add separately (FLOPs = 2 x MACs)


def to_tflite(m, path, int8=False):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]          # dynamic-range int8 weights
    open(path, 'wb').write(conv.convert())
    return os.path.getsize(path)


def interpreter(path, threads=None):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    return it


def _latency_worker(kind, path, threads, n_warm, n_run, weights):
    """runs in a fresh process so the thread setting takes effect"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    if threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    x = np.random.default_rng(0).random((1, 224, 224, 3), dtype=np.float32)
    if kind == 'keras':
        m = _model(weights)
        f = tf.function(lambda t: m(t, training=False))
        call = lambda: f(x).numpy()
    else:
        it = interpreter(path, threads if threads > 0 else None)
        i, o = it.get_input_details()[0]['index'], it.get_output_details()[0]['index']
        def call():
            it.set_tensor(i, x); it.invoke(); return it.get_tensor(o)
    for _ in range(n_warm):
        call()
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter(); call(); ts.append((time.perf_counter() - t0) * 1000)
    print(json.dumps(dict(median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)),
                          mean_ms=float(np.mean(ts)))))


def latency(kind, path, threads, weights, n_warm=20, n_run=200):
    code = (f"import sys; sys.path.insert(0, {HERE!r}); import edge_profile as E; "
            f"E._latency_worker({kind!r}, {path!r}, {threads}, {n_warm}, {n_run}, {weights!r})")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith('{')]
    if not line:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return json.loads(line[-1])


def cmd_profile(a):
    os.makedirs(a.out, exist_ok=True)
    import tensorflow as tf
    m = _model(a.weights)
    tr = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    res = dict(params_total=int(m.count_params()), params_trainable=tr,
               fp32_state_MB=round(sum(w.nbytes for w in m.get_weights()) / 2 ** 20, 3))
    fl = flops_per_image(m)
    res.update(flops_per_image=fl, gflops_per_image=round(fl / 1e9, 3), gmacs_per_image=round(fl / 2e9, 3))
    p32, p8 = os.path.join(a.out, 'model_fp32.tflite'), os.path.join(a.out, 'model_int8_dynamic.tflite')
    res['tflite_fp32_MB'] = round(to_tflite(m, p32) / 2 ** 20, 3)
    res['tflite_int8_MB'] = round(to_tflite(m, p8, int8=True) / 2 ** 20, 3)
    res['cpu'] = (subprocess.run("lscpu | grep 'Model name' | head -1", shell=True, capture_output=True,
                                 text=True).stdout.split(':')[-1].strip() or 'unknown')
    res['cpu_cores'] = os.cpu_count()
    lat = {}
    for kind, path in (('keras', ''), ('tflite_fp32', p32), ('tflite_int8', p8)):
        for th in (1, 0):
            key = f"{kind}_{'1thread' if th == 1 else 'allthreads'}"
            lat[key] = latency('keras' if kind == 'keras' else 'tflite', path, th, a.weights)
            print(key, lat[key], flush=True)
    res['latency_batch1_ms'] = lat
    json.dump(res, open(os.path.join(a.out, 'edge_profile.json'), 'w'), indent=2)
    print(json.dumps(res, indent=2))


def cmd_tflite_acc(a):
    import tensorflow as tf
    import legacy_tools as L
    from sklearn.metrics import f1_score
    Xtr, Xte, ytr, yte, classes, *_ = L.get_split(a.data, a.order)
    m = _model(a.weights)
    p32, p8 = os.path.join(a.out, 'acc_fp32.tflite'), os.path.join(a.out, 'acc_int8.tflite')
    to_tflite(m, p32); to_tflite(m, p8, int8=True)
    ds = L.make_dataset(Xte, yte, len(classes), 1)
    preds = {'keras': [], 'tflite_fp32': [], 'tflite_int8': []}
    its = {}
    for k, p in (('tflite_fp32', p32), ('tflite_int8', p8)):
        its[k] = interpreter(p)
    f = tf.function(lambda t: m(t, training=False))
    t0 = time.time()
    for n, (xb, _) in enumerate(ds):
        x = xb.numpy()
        preds['keras'].append(int(np.argmax(f(x).numpy())))
        for k, it in its.items():
            it.set_tensor(it.get_input_details()[0]['index'], x); it.invoke()
            preds[k].append(int(np.argmax(it.get_tensor(it.get_output_details()[0]['index']))))
        if n % 1000 == 0:
            print(f'  {n}/{len(yte)} ({time.time()-t0:.0f}s)', flush=True)
    res = {k: dict(accuracy=float(np.mean(np.array(v) == yte)),
                   macro_f1=float(f1_score(yte, v, average='macro')),
                   agreement_with_keras=float(np.mean(np.array(v) == np.array(preds['keras']))))
           for k, v in preds.items()}
    res['weights'] = a.weights
    json.dump(res, open(os.path.join(a.out, 'tflite_accuracy.json'), 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('profile'); p.add_argument('--out', required=True); p.add_argument('--weights', default='')
    q = sub.add_parser('tflite_acc'); q.add_argument('--out', required=True); q.add_argument('--weights', required=True)
    q.add_argument('--data', required=True); q.add_argument('--order', default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {'profile': cmd_profile, 'tflite_acc': cmd_tflite_acc}[a.cmd](a)
