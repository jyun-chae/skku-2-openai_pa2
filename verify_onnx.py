import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("generator_1024_step20000.onnx", providers=["CPUExecutionProvider"])

for inp in sess.get_inputs():
    print(f"Input : {inp.name}  shape={inp.shape}  type={inp.type}")
for out in sess.get_outputs():
    print(f"Output: {out.name}  shape={out.shape}  type={out.type}")
print()

for b in (1, 2):
    z = np.random.randn(b, 512).astype(np.float32)
    img = sess.run(None, {"z": z})[0]
    ok = img.shape == (b, 3, 1024, 1024) and -1.0 <= img.min() and img.max() <= 1.0
    status = "PASS" if ok else "FAIL"
    print(f"b={b}: shape={img.shape}  range=[{img.min():.4f}, {img.max():.4f}]  {status}")
