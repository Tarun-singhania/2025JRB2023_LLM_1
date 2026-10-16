from pathlib import Path
from huggingface_hub import snapshot_download

HF_USER = "TarunExhaustice"
HF_REPOS = {
    "A1": f"{HF_USER}/llm-a1-a1",
    "A2": f"{HF_USER}/llm-a1-a2",
    "A3": f"{HF_USER}/llm-a1-a3",
    "A4": f"{HF_USER}/llm-a1-a4",
    "A5": f"{HF_USER}/llm-a1-a5",
    "A6": f"{HF_USER}/llm-a1-a6",
    "best_sanitizer": f"{HF_USER}/llm-a1-best-sanitizer",
    "victim_raw": f"{HF_USER}/llm-a1-victim-raw",
    "victim_san": f"{HF_USER}/llm-a1-victim-san",
}


def resolve_adapter(path):
    """Local folder ho to wahi; warna naam (adapters/<name>) se HF Hub se download."""
    if path is None or str(path).lower() == "none":
        return path
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return str(p)
    name = p.name
    if name in HF_REPOS:
        print(f"[hf_assets] '{path}' not found locally, downloading {HF_REPOS[name]} ...")
        return snapshot_download(repo_id=HF_REPOS[name])
    return str(path)