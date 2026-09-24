# Third-party components

## SUTrack

This archive includes an unmodified, inference-oriented subset of [SUTrack](https://github.com/chenxin-dlut/SUTrack), commit `d65052d1ba3fcf55010e1fb3665ee6616c139a2c`, under `third_party/SUTrack`. Its MIT license is reproduced verbatim in `third_party/SUTrack/LICENSE.txt`. Original per-file copyright notices (including Meta/Facebook notices where present) are preserved. Training, dataset downloads, and weights are not bundled.

The wrapper constructs the checkpoint's CLIP architecture without downloading redundant pretrained weights, then loads the final SUTrack state dictionary strictly. It does not change the vendored neural tracker.

PyTorch, torchvision, OpenAI CLIP, timm, NumPy, OpenCV and other dependencies are installed separately and retain their own licenses. Consult their distributions for notices. Dataset and model-weight access and use must comply with the respective owners' terms.
