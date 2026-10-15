import torch
import torch.nn.functional as F

from reviser_v2.model import ActionLM, ModelConfig


def test_sparse_logits_loss_matches_dense():
    torch.manual_seed(0)
    m = ActionLM(ModelConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=64, dropout=0.0))
    ids = torch.randint(0, 97, (3, 20))
    labels = torch.randint(0, 97, (3, 20))
    labels[:, :7] = -100
    labels[1, 15:] = -100
    logits, _ = m(ids)
    dense = F.cross_entropy(logits.view(-1, 97), labels.view(-1), ignore_index=-100, reduction="sum")
    _, _, h = m(ids, return_hidden=True, skip_logits=True)
    lm = labels != -100
    sparse = F.cross_entropy(m.logits_from_hidden(h[lm]), labels[lm], reduction="sum")
    assert torch.allclose(dense, sparse, rtol=1e-5, atol=1e-5)
    g1 = torch.autograd.grad(dense, m.embed.weight)[0]
    g2 = torch.autograd.grad(sparse, m.embed.weight)[0]
    assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-6)
