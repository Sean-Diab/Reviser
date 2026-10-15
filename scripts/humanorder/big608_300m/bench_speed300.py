"""Speed rows for the new models (helper L, 300M fill; notes/big608-300m-2026-10-09.md).

Runs helper K's scripts/speed/bench_speed.py unchanged (sha256 checked; notes/speed-2026-10-09.md): it is imported as a
module, three entries are added to its MODELS table, and its own main() runs with the same arguments K used. Nothing
else changes: same decoder (E's decode_batch), same fp16 swap, same prompts, protocol, GPU-wait rule and outputs.

Added models (each with its own locked decode as the fp16 replay check):
  jd608_300m_rl_s3  runs/big608_300m/models/jd608_300m_g8m2_s3.model_step000200.pt  (L's 300M + RL seed 3)
  jd608             runs/big608/models/jd608_final_model.pt                         (B's 100M, no RL)
  july608           runs/july_control/ckpt/july_runjump_step84488_model.pt          (July recipe 100M, 608M tokens)

  /opt/venv/bin/python scripts/humanorder/big608_300m/bench_speed300.py --models jd608_300m_rl_s3,ar300 \
      --batches 32,8,1 --n 64 --repeats 3 --amp fp16 --out_dir runs/speed/<ts>
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BS = ROOT / "scripts" / "speed" / "bench_speed.py"
BS_SHA = "c24480991ec5dbca2476f2d089a723bee246e476372f5d7d38dda7f60ed177e2"

h = hashlib.sha256(BS.read_bytes()).hexdigest()
if h != BS_SHA:
    raise SystemExit(f"bench_speed.py changed (sha256 {h}); re-check before using this wrapper")
sys.path.insert(0, str(BS.parent))
import bench_speed as B  # noqa: E402

B.MODELS.update({
    "jd608_300m_rl_s3": dict(path="runs/big608_300m/models/jd608_300m_g8m2_s3.model_step000200.pt", kind="reviser",
                             size="300M", label="jd608_300m + RL seed 3 (L)",
                             sha256="ae0cedc8cbb162e4acda1d80aecb00731585938706a894386e74310c9627f392",
                             ref="runs/big608_300m/dec/c4paper3k.lock.jd608_300m_g8m2_s3.jsonl"),
    "jd608": dict(path="runs/big608/models/jd608_final_model.pt", kind="reviser", size="100M",
                  label="jd608, no RL (B)",
                  sha256="641db09e231108be002461d41d914fa91ca567937e6f58e3224ac71ec6ab9342",
                  ref="runs/papereval/dec/c4paper3k.lock.jd608.jsonl"),
    "july608": dict(path="runs/july_control/ckpt/july_runjump_step84488_model.pt", kind="reviser", size="100M",
                    label="july608 (July recipe, 608M tokens)",
                    sha256="0256464d733f524c1b3a73c2aac7631624d9305ab0f9237204852d23b63bd9a7",
                    ref="runs/papereval/dec/c4paper3k.lock.july608.jsonl"),
})

if __name__ == "__main__":
    B.main()
