import numpy as np

from probe_sample_coherence import compute_sample_coherence, parse_val_txt_grouped


def test_perfect_separation_gives_auc_1():
    """Embeddings perfectly clustered by sample -> AUC = 1."""
    rng = np.random.RandomState(42)
    centers = rng.randn(5, 16) * 5
    embeddings = []
    labels = []
    for s in range(5):
        embeddings.append(centers[s] + 0.1 * rng.randn(20, 16))
        labels.extend([s] * 20)
    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)
    result = compute_sample_coherence(embeddings, labels)
    assert result["auc"] > 0.95


def test_random_embeddings_give_auc_near_half():
    """Random embeddings -> AUC ~ 0.5."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 16)
    labels = rng.randint(0, 5, size=100)
    result = compute_sample_coherence(embeddings, labels)
    assert 0.4 < result["auc"] < 0.6


def test_handles_singleton_samples():
    """Sample with only 1 read can't generate within-sample pairs but shouldn't crash."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(20, 16)
    labels = np.array([0] * 10 + [1] * 9 + [2])  # sample 2 has 1 read
    result = compute_sample_coherence(embeddings, labels)
    assert "auc" in result


def test_parse_val_txt_grouped_smoke(tmp_path):
    """Round-trip: write a tiny val.txt, parse it back."""
    vp = tmp_path / "val.txt"
    vp.write_text(
        "<SAMPLE_START> <READ_START> A C G T <PAIRED_END> T G C A <READ_END> "
        "<READ_START> G G G G <READ_END> <SAMPLE_END> "
        "<SAMPLE_START> <READ_START> T T T T <READ_END> <SAMPLE_END>"
    )
    samples = parse_val_txt_grouped(vp, n_per_sample=10)
    assert len(samples) == 2
    assert len(samples[0]) == 2  # first sample has 2 reads
    assert len(samples[1]) == 1  # second sample has 1 read
    assert "<PAIRED_END>" in samples[0][0]  # paired-end marker preserved
