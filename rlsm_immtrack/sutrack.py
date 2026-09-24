"""Load the pinned SUTrack-B224 checkpoint without redundant downloads."""
from pathlib import Path
import sys
import torch

def configure_sutrack(repo: Path, checkpoint: Path):
    repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError('Inference requires a CUDA GPU; scoring works on CPU.')
    sys.path.insert(0, str(repo))

    import clip
    from clip.model import CLIP

    def checkpoint_only_clip_load(name, device="cpu", jit=False, download_root=None):
        del name, jit, download_root
        model = CLIP(
            embed_dim=768,
            image_resolution=224,
            vision_layers=24,
            vision_width=1024,
            vision_patch_size=14,
            context_length=77,
            vocab_size=49408,
            transformer_width=768,
            transformer_heads=12,
            transformer_layers=12,
        )
        return model.to(device), None

    # The final SUTrack checkpoint contains the encoder and CLIP weights.  These
    # two patches prevent the builders from downloading redundant pretraining
    # files while preserving the official final state_dict exactly.
    original_clip_load = clip.load
    clip.load = checkpoint_only_clip_load
    import lib.models.sutrack.encoder as encoder_module

    original_main_process = encoder_module.is_main_process
    encoder_module.is_main_process = lambda: False

    from lib.config.sutrack.config import cfg, update_config_from_file
    from lib.test.tracker.sutrack import SUTRACK
    from lib.test.utils import TrackerParams

    update_config_from_file(str(repo / "experiments" / "sutrack" / "sutrack_b224.yaml"))
    params = TrackerParams()
    params.cfg = cfg
    params.yaml_name = "sutrack_b224"
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = str(checkpoint)
    params.save_all_boxes = False
    params.debug = 0
    try:
        return SUTRACK(params, "UAV2UAV2")
    finally:
        clip.load = original_clip_load
        encoder_module.is_main_process = original_main_process
