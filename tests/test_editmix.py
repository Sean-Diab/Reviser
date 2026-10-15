"""Reviser-Edit v2: task tokens (tasktokens.py), ModelConfig.n_out, the editmix task (tasks/editmix.py), the mix builder's
pure functions (scripts/editdata/build_mix_v2.py) and, when the built mix exists, its split / eval separation."""
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from reviser_v2 import train
from reviser_v2.model import ActionLM, ModelConfig
from reviser_v2.realdata.editscript import gpt2_action_space
from reviser_v2.tasktokens import (GPT2_NAME_IDS, N_TASKS, TASKS, apply_task_tokens, default_init_rows, eval_category,
                                   expand_embedding, resolve_task_tokens, task_id)
from reviser_v2.trajectories import episode_to_sequence

ROOT = Path(__file__).resolve().parents[1]
SPACE = gpt2_action_space()


def _builder():
    spec = importlib.util.spec_from_file_location("build_mix_v2", ROOT / "scripts" / "editdata" / "build_mix_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tok():
    tdir = ROOT / "data" / "tokenizers" / "gpt2"
    if not tdir.exists():
        pytest.skip("local GPT-2 tokenizer files not present")
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained(str(tdir))


# ------------------------------------------------------------------ task tokens
def test_task_ids_follow_action_vocab():
    assert SPACE.vocab_size == 50281
    assert [task_id(SPACE.vocab_size, t) for t in TASKS] == list(range(50281, 50281 + N_TASKS))


def test_name_ids_match_tokenizer():
    tok = _tok()
    for t in TASKS:
        assert tok.encode(" " + t, add_special_tokens=False) == GPT2_NAME_IDS[t], t


def test_eval_category_mapping():
    assert eval_category("gec", "coedit") == "grammar"
    assert eval_category("simplification", "coedit") == "clarity"
    assert eval_category("meaning-changed", "iterater_human_sent") == "polish"
    assert eval_category("fluency", "iterater_v2_sent") == "fluency"
    assert eval_category("synth2", "synth") == "grammar"
    assert eval_category("clean", "synth_real") == "grammar"
    assert eval_category("synth1", "synth_forced") is None
    recs = [dict(intent="gec", instruction_ids=[1, 2]), dict(intent="synth0", instruction_ids=[3])]
    out = apply_task_tokens(recs, SPACE.vocab_size, "coedit")
    assert out[0]["instruction_ids"] == [task_id(SPACE.vocab_size, "grammar"), 1, 2]
    assert recs[0]["instruction_ids"] == [1, 2]  # originals untouched
    assert apply_task_tokens(recs, SPACE.vocab_size, "synth_forced")[1]["instruction_ids"] == [3]


def test_expand_embedding_rows_are_means():
    w = torch.randn(10, 4)
    sd = {"embed.weight": w, "x": torch.ones(2)}
    out = expand_embedding(sd, 2, [[1, 2], [5]])
    assert out["embed.weight"].shape == (12, 4)
    assert torch.allclose(out["embed.weight"][10], w[[1, 2]].mean(0))
    assert torch.equal(out["embed.weight"][11], w[5])
    assert torch.equal(out["embed.weight"][:10], w) and out["x"] is sd["x"] and sd["embed.weight"].shape == (10, 4)
    assert len(default_init_rows()) == N_TASKS


def test_n_out_limits_logits_but_not_inputs():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=30, n_out=25, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=64)
    m = ActionLM(cfg).eval()
    ids = torch.tensor([[27, 3, 4, 29, 5]])  # task-token ids (>= n_out) are valid inputs
    logits, _ = m(ids)
    assert logits.shape == (1, 5, 25)
    # same as the full tied projection restricted to the first n_out rows
    _, _, h = m(ids, return_hidden=True, skip_logits=True)
    full = torch.nn.functional.linear(m.ln_f(h), m.embed.weight)
    assert torch.allclose(logits, full[..., :25], atol=1e-6)
    with pytest.raises(AssertionError):
        ActionLM(ModelConfig(vocab_size=30, n_out=25, tie_embeddings=False, d_model=16, n_layers=1, n_heads=2, d_ff=32))


# ------------------------------------------------------------------ editmix source on a tiny synthetic mix
def _write_mix(d: Path):
    rng = random.Random(0)
    (d / "pools").mkdir(parents=True)
    words = list(range(300, 400))

    def rec(n):
        return [rng.choice(words) for _ in range(n)]

    pools = {}
    specs = {"grammar_sent_x": ("edit", "grammar", "sent", True), "coherence_para_x": ("edit", "coherence", "para", False),
             "nochange_sent": ("nochange", None, "sent", False), "restoration": ("restoration", None, None, False)}
    for name, (kind, cat, level, has_instr) in specs.items():
        meta = dict(kind=kind, cat=cat, level=level, splits={}, instr_vocab=[[11, 12]] if has_instr else [])
        for split, n in (("train", 7), ("dev", 3), ("test", 3)):
            a = [rec(rng.randint(5, 20)) for _ in range(n)]
            arr = dict(a_flat=np.array([t for x in a for t in x], dtype=np.uint16),
                       a_off=np.concatenate([[0], np.cumsum([len(x) for x in a])]).astype(np.int64))
            if kind == "edit":
                b = [x[:2] + [rng.choice(words)] + x[3:] for x in a]
                arr.update(b_flat=np.array([t for x in b for t in x], dtype=np.uint16),
                           b_off=np.concatenate([[0], np.cumsum([len(x) for x in b])]).astype(np.int64))
                if has_instr:
                    arr["instr"] = np.zeros(n, dtype=np.int32)
            f = d / "pools" / f"{name}_{split}.npz"
            np.savez(f, **arr)
            meta["splits"][split] = dict(file=f"pools/{name}_{split}.npz", records=n)
        pools[name] = meta
    man = dict(pools=pools, cat_instr={t: [[40, 41, 42]] for t in TASKS}, sha256="testsha")
    (d / "manifest.json").write_text(json.dumps(man))
    return [{"pool": p, "weight": w} for p, w in zip(specs, (0.4, 0.2, 0.2, 0.2))]


def _cfg(mix_dir, mix, layout="edit", init=None):
    cfg = {
        "run": {"name": "em", "seed": 1, "data_seed": 5},
        "task": {"name": "editmix", "mix_dir": str(mix_dir), "layout": layout, "schedule_episodes": 64, "dev_episodes": 16,
                 "p_instr": 0.5, "manifest_sha256": "testsha",
                 "restoration": {"prefix_frac": 0.35, "policy": {"p_del": 0.7, "p_move": 0.18, "p_ins": 0.12}}},
        "data": {"mix": mix, "mask_doomed": layout == "edit", "loss_norm": "eligible", "max_len": 96, "sparse_logits": True},
        "optim": {"lr": 1e-3, "warmup": 1, "steps": 3, "batch_size": 4, "accum": 2, "trim_micro": True},
        "log": {"every": 1}, "eval": {"every": 3}, "ckpt": {"every": 3},
    }
    if init:
        cfg["init"] = {"ckpt": str(init)}
    else:
        cfg["model"] = {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "max_seq_len": 128, "attn_bias": False}
    return cfg


def test_editmix_episodes_both_layouts(tmp_path):
    from reviser_v2.tasks.editmix import EditMixSource
    mix = _write_mix(tmp_path / "mix")
    for layout in ("edit", "ar"):
        task = _cfg(tmp_path / "mix", mix, layout)["task"]
        src = EditMixSource(SPACE, task, mix, 5, "train")
        src2 = EditMixSource(SPACE, task, mix, 5, "train")
        seen = set()
        for i in range(64):
            ep, label = src.episode(i)
            ep2, _ = src2.episode(i)
            assert ep == ep2  # deterministic
            ids, labels, _ = episode_to_sequence(SPACE, ep, layout == "edit")
            assert max(ids) < SPACE.vocab_size + N_TASKS
            assert all(l == -100 or l < SPACE.vocab_size for l in labels)  # never a task-token target
            kind = label.split(":")[0]
            seen.add(kind)
            n_task = sum(1 for t in ids if t >= SPACE.vocab_size)
            assert n_task == (0 if kind == "restoration" else 1), label
        assert seen == {"edit", "nochange", "restoration"}
    # schedule: every record of a pool used floor/ceil(views) times
    st = src.schedule_stats()
    for name, s in st.items():
        assert s["max_uses"] - s["min_uses"] <= 1, name
        assert s["episodes"] > 0
    assert sum(s["episodes"] for s in st.values()) == 64


def test_ar_and_edit_layouts_see_same_examples(tmp_path):
    from reviser_v2.tasks.editmix import EditMixSource
    mix = _write_mix(tmp_path / "mix")
    e = EditMixSource(SPACE, _cfg(tmp_path / "mix", mix, "edit")["task"], mix, 5, "train")
    a = EditMixSource(SPACE, _cfg(tmp_path / "mix", mix, "ar")["task"], mix, 5, "train")
    for i in range(64):
        ee, le = e.episode(i)
        ea, la = a.episode(i)
        assert le == la
        assert ee.target == ea.target
        if le.startswith("edit") or le.startswith("nochange"):
            assert ea.prompt[:len(ee.prompt)] == ee.prompt  # AR prompt = same [TASK] + instruction, then ":" + draft


def test_train_editmix_with_task_token_init_and_eval(tmp_path):
    mix = _write_mix(tmp_path / "mix")
    torch.manual_seed(0)
    base = ActionLM(ModelConfig(vocab_size=SPACE.vocab_size, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=128,
                                attn_bias=False))
    init = tmp_path / "init.pt"
    torch.save({"model": base.state_dict(), "model_config": base.cfg.to_dict(), "action_space": SPACE.to_dict()}, init)
    for layout in ("edit", "ar"):
        cfg = _cfg(tmp_path / "mix", mix, layout, init)
        p = tmp_path / f"{layout}.yaml"
        p.write_text(yaml.safe_dump(cfg))
        run = tmp_path / f"run_{layout}"
        train.main(["--config", str(p), "--run-dir", str(run), "--device", "cpu"])
        ck = torch.load(run / "ckpt" / "latest.pt", weights_only=False)
        assert ck["model_config"]["vocab_size"] == SPACE.vocab_size + N_TASKS
        assert ck["model_config"]["n_out"] == SPACE.vocab_size
        man = json.load(open(run / "manifest.json"))
        assert man["init"]["task_tokens"]["first_id"] == SPACE.vocab_size
        assert man["data_mix"]["manifest_sha256"] == "testsha"
        assert resolve_task_tokens("auto", run) == SPACE.vocab_size
    assert resolve_task_tokens("auto", init) is None
    with pytest.raises(SystemExit):
        resolve_task_tokens("on", init)
    # rollout with a task-token prompt (edit layout)
    from reviser_v2.evaluate_edits import predict
    recs = apply_task_tokens([dict(intent="gec", instruction_ids=[11, 12], before_ids=[300, 301, 302], after_ids=[300, 302])],
                             SPACE.vocab_size, "coedit")
    traces, meta = predict(recs, str(tmp_path / "run_edit"), "cpu", 4, 0.0, True, 0, 8)
    assert len(traces) == 1 and all(t < SPACE.n_content for t in traces[0].final)
    traces, _ = predict(recs, str(tmp_path / "run_ar"), "cpu", 4, 0.0, True, 0, 8, layout="ar")
    assert len(traces) == 1


def test_manifest_sha_mismatch_refused(tmp_path):
    from reviser_v2.tasks.editmix import EditMixSource
    mix = _write_mix(tmp_path / "mix")
    task = dict(_cfg(tmp_path / "mix", mix)["task"], manifest_sha256="other")
    with pytest.raises(ValueError):
        EditMixSource(SPACE, task, mix, 5, "train")


# ------------------------------------------------------------------ builder pure functions
def test_builder_split_and_sentences():
    b = _builder()
    assert b.split_of("doc-1") == b.split_of("doc-1")
    counts = {"train": 0, "dev": 0, "test": 0}
    for i in range(20000):
        counts[b.split_of(f"k{i}")] += 1
    assert 0.97 < counts["train"] / 20000 < 0.99 and counts["dev"] > 100 and counts["test"] > 100
    assert b.sentences("It rained. Then it stopped! \"Why?\" she asked. 3 cats.") == \
        ["It rained.", "Then it stopped!", "\"Why?\" she asked.", "3 cats."]
    assert b.norm("  Hello,  World!! ") == "hello world"
    assert b.in_band(b.kept(([1, 2, 3, 4, 5], [1, 2, 3, 4, 9]))) and not b.in_band(b.kept(([1], [2, 3])))


def test_builder_allocation_caps_and_spill():
    b = _builder()
    pm = {}
    uniq = {"restoration": 100, "nochange_para": 50, "nochange_sent": 50, "grammar_sent_coedit": 2, "grammar_sent_c4200m": 500,
            "grammar_para_llm": 40, "fluency_sent_real": 10, "clarity_sent_real": 40, "coherence_sent_real": 30,
            "coherence_para_llm": 30, "polish_sent_real": 5, "polish_para_iterater": 5, "polish_para_llm": 5}
    for p, u in uniq.items():
        cat = next(c for (c, lv), lst in b.SLOT_POOLS.items() for _, q in lst if q == p)
        lvl = next(lv for (c, lv), lst in b.SLOT_POOLS.items() for _, q in lst if q == p)
        kind = cat if cat in ("restoration", "nochange") else "edit"
        pm[p] = dict(kind=kind, cat=None if kind != "edit" else cat, level=lvl,
                     splits={"train": dict(plan_tokens=u, mean_plan_tokens=1.0)})
    t, notes = b.allocate(pm, 1000, 4.0)
    assert abs(sum(t.values()) - 1000) < 1e-6
    for p, v in t.items():
        if p != b.SPILL_POOL:
            assert v <= 4.0 * uniq[p] + 1e-9, p
    assert abs(t["grammar_sent_coedit"] - 8) < 1e-9  # real pool first, at the cap
    assert abs(t["restoration"] - 150) < 1e-9
    assert notes  # fluency / polish short
    so, so_notes = b.sent_only_variant(pm, t, 6.0)
    assert abs(sum(so.values()) - 1000) < 1e-6 and not any(pm[p]["level"] == "para" for p in so)
    assert so["restoration"] == t["restoration"] and abs(so["nochange_sent"] - 100) < 1e-9  # no change: all 10% in sentences
    for p, v in so.items():
        if p != b.SPILL_POOL:
            assert v <= 6.0 * uniq[p] + 1e-9, p
    nr = b.variant(pm, t, {"restoration"})
    assert abs(sum(nr.values()) - 1000) < 1e-6 and "restoration" not in nr


# ------------------------------------------------------------------ the built mix (skipped when absent)
MIX = ROOT / "data" / "editdata" / "mix_v2"


@pytest.mark.skipif(not (MIX / "manifest.json").exists(), reason="mix_v2 not built")
def test_built_mix_splits_disjoint_and_no_eval_overlap():
    man = json.load(open(MIX / "manifest.json", encoding="utf-8"))
    tok = _tok()
    b = _builder()
    ek = b.EvalKeys()
    rng = random.Random(0)
    for name, pm in man["pools"].items():
        uids = {}
        for split, s in pm["splits"].items():
            z = np.load(MIX / s["file"])
            u = bytes(z["uids"]).decode("utf-8").split("\n") if z["uids"].size else []
            assert len(u) == s["records"] == len(z["a_off"]) - 1, (name, split)
            uids[split] = set(u)
            if split == "train" and s["records"]:
                for j in rng.sample(range(s["records"]), min(300, s["records"])):
                    a = z["a_flat"][z["a_off"][j]:z["a_off"][j + 1]].tolist()
                    texts = [tok.decode(a)]
                    if "b_flat" in z.files:
                        texts.append(tok.decode(z["b_flat"][z["b_off"][j]:z["b_off"][j + 1]].tolist()))
                    assert not ek.hit(*texts), (name, j)
        assert not (uids["train"] & uids["dev"]) and not (uids["train"] & uids["test"]) and not (uids["dev"] & uids["test"]), name
