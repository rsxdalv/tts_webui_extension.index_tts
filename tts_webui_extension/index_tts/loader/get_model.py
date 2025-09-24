import os
from tts_webui.utils.manage_model_state import manage_model_state
from .download_indextts_checkpoint import download_indextts_checkpoint, required_files

MODEL_DIR = os.path.join("data", "models", "index-tts")

@manage_model_state("index-tts")
def get_model(model_name="IndexTeam/IndexTTS-2", model_dir=MODEL_DIR, use_fp16=None, use_deepspeed=None, use_cuda_kernel=None):
    cfg_path = os.path.join(model_dir, "config.yaml")

    key = (os.path.abspath(model_dir), bool(use_fp16), bool(use_deepspeed), bool(use_cuda_kernel))

    if not check_required_files(model_dir=model_dir):
        download_indextts_checkpoint(dest_dir=model_dir)

    from indextts.infer_v2 import IndexTTS2

    return IndexTTS2(model_dir=model_dir,
                     cfg_path=cfg_path,
                     use_fp16=use_fp16,
                     use_deepspeed=use_deepspeed,
                     use_cuda_kernel=use_cuda_kernel,
                     )

def check_required_files(model_dir=MODEL_DIR):
    if not os.path.exists(model_dir):
        return False
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            return False
        
    return True
