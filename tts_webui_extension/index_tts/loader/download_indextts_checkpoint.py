import os
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download

required_files = [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
    "qwen0.6bemo4-merge/Modelfile",
    "qwen0.6bemo4-merge/added_tokens.json",
    "qwen0.6bemo4-merge/chat_template.jinja",
    "qwen0.6bemo4-merge/config.json",
    "qwen0.6bemo4-merge/generation_config.json",
    "qwen0.6bemo4-merge/merges.txt",
    "qwen0.6bemo4-merge/model.safetensors",
    "qwen0.6bemo4-merge/special_tokens_map.json",
    "qwen0.6bemo4-merge/tokenizer.json",
    "qwen0.6bemo4-merge/tokenizer_config.json",
    "qwen0.6bemo4-merge/vocab.json",
]


def download_indextts_checkpoint(
    repo_id: str = "IndexTeam/IndexTTS-2",
    revision: str = "main",
    dest_dir: str = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(dest_dir, exist_ok=True)
    for rel_path in required_files:
        out_path = os.path.join(dest_dir, rel_path)
        if os.path.exists(out_path):
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            local_path = hf_hub_download(
                repo_id=repo_id, filename=rel_path, revision=revision, token=token
            )
            with open(local_path, "rb") as src, open(out_path, "wb") as dst:
                dst.write(src.read())
        except Exception:
            # silent on errors per request to remove failed/missing; caller can inspect filesystem if needed
            pass
