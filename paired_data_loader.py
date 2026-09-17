"""
Sample-aware paired data loader for the MLM + contrastive arm.

Yields batches of (token_ids, attention_lengths, sample_ids) where every batch
contains reads from >=2 distinct samples (so InfoNCE has positives and negatives).

Reuses probe_sample_coherence.parse_val_txt_grouped so train/val sample membership
is derived by the EXACT same logic the probe uses on val.txt.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

from probe_sample_coherence import parse_val_txt_grouped
from prepare_genomic import Tokenizer


# Fixed sequence length for MLM (DNA reads tokenize to ~50 tokens; 64 leaves headroom).
SEQ_LEN_DEFAULT = 64


class PairedDataLoader:
    """Iterable batch generator: K samples × M reads = B reads per batch.

    Returns torch.LongTensors (token_ids[B, T], attention_lengths[B], sample_ids[B]).
    Pad positions use the BOS id (4090) — a special token never selected for MLM masking.
    """
    def __init__(self, txt_path: Path | str, tokenizer: Tokenizer,
                 samples_per_batch: int = 4, reads_per_sample: int = 8,
                 n_reads_cap_per_sample: int = 200, seq_len: int = SEQ_LEN_DEFAULT,
                 seed: int = 42, max_read_len: int | None = None):
        self.txt_path = Path(txt_path)
        self.tokenizer = tokenizer
        self.K = samples_per_batch
        self.M = reads_per_sample
        self.T = seq_len                       # CURRENT context length (mutable via set_seq_len)
        # Per-read drop threshold. For the seq-len curriculum this is the CEILING (keep every read
        # that fits the largest window we will ever use), NOT the current T — so growing T later
        # doesn't require re-tokenizing. Defaults to seq_len, so non-curriculum runs are unchanged.
        self.max_read_len = int(max_read_len) if max_read_len else seq_len
        self.rng = random.Random(seed)
        self.pad_id = tokenizer.enc.encode_single_token("<|bos|>")

        raw = parse_val_txt_grouped(self.txt_path, n_per_sample=n_reads_cap_per_sample, seed=seed)
        # Tokenize once. Each read becomes a list of token ids wrapped in READ_START/READ_END
        # (matches probe_sample_coherence.embed_reads convention).
        self.samples: dict[int, list[list[int]]] = {}
        n_total_reads = 0
        n_dropped_too_long = 0
        for s_idx, bodies in raw.items():
            toks: list[list[int]] = []
            for body in bodies:
                text = f"<READ_START> {body} <READ_END>"
                ids = tokenizer.encode(text)
                if len(ids) > self.max_read_len:
                    n_dropped_too_long += 1
                    continue
                toks.append(ids)
            if toks:
                self.samples[s_idx] = toks
                n_total_reads += len(toks)
        self.sample_ids_list = sorted(self.samples.keys())

        if self.K < 2:
            raise ValueError(f"samples_per_batch must be >=2 (got {self.K}) — InfoNCE needs negatives")
        if len(self.sample_ids_list) < self.K:
            raise ValueError(f"Only {len(self.sample_ids_list)} samples available, need K={self.K}")

        eff_B = self.K * self.M
        print(f"[paired_loader] path: {self.txt_path}")
        print(f"[paired_loader] parsed {len(self.samples)} samples, {n_total_reads} reads "
              f"(dropped {n_dropped_too_long} reads > max_read_len={self.max_read_len} tokens)")
        print(f"[paired_loader] batch recipe: K={self.K} samples × M={self.M} rows = batch size {eff_B}"
              f"; context length T={self.T} (rows pack multiple same-sample reads to fill T)")
        print(f"[paired_loader] pad token id={self.pad_id} (<|bos|>) — never a mask candidate")

    def __iter__(self):
        return self

    def __next__(self):
        return self._build_batch()

    def set_seq_len(self, T: int):
        """Grow (or shrink) the current context length. The seq-len curriculum calls this on a
        val plateau; each row then packs more same-sample reads to fill the wider window."""
        self.T = int(T)

    def _pack_row(self, reads, T):
        """Fill a T-token row by concatenating randomly-drawn reads from ONE sample, in order,
        until the next read would overflow. Each read keeps its <READ_START>/<READ_END> boundary,
        so once T exceeds a single read the model sees within-sample, cross-read structure. At
        T≈64 this yields ~one read per row — matching the pre-curriculum behavior."""
        order = self.rng.sample(reads, len(reads)) if len(reads) > 1 else list(reads)
        row, i = [], 0
        while len(row) < T:
            if i >= len(order):                        # pool exhausted; reshuffle to keep filling
                order = self.rng.sample(reads, len(reads)) if len(reads) > 1 else list(reads)
                i = 0
            r = order[i]; i += 1
            if not row:
                row.extend(r[:T])                      # first read: truncate only if it alone > T
            elif len(row) + len(r) <= T:
                row.extend(r)
            else:
                break                                  # next read won't fit; pad the remainder
        return row[:T]

    def _build_batch(self):
        # Pick K samples preferring those with >=M reads; fall back to any with >=2.
        with_enough = [s for s in self.sample_ids_list if len(self.samples[s]) >= self.M]
        pool = with_enough if len(with_enough) >= self.K else [s for s in self.sample_ids_list if len(self.samples[s]) >= 2]
        chosen = self.rng.sample(pool, self.K)

        T = self.T
        batch_rows, batch_lens, batch_sample_ids = [], [], []
        for s in chosen:                               # K samples × M rows; each row packs reads to T
            reads = self.samples[s]
            for _ in range(self.M):
                row = self._pack_row(reads, T)
                batch_rows.append(row)
                batch_lens.append(len(row))
                batch_sample_ids.append(s)

        B = len(batch_rows)
        token_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        for i, row in enumerate(batch_rows):
            token_ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        attn_len = torch.tensor(batch_lens, dtype=torch.long)
        sample_ids = torch.tensor(batch_sample_ids, dtype=torch.long)

        assert sample_ids.unique().numel() >= 2, "batch lacks >=2 distinct samples"
        return token_ids, attn_len, sample_ids


if __name__ == "__main__":
    tk = Tokenizer.from_directory()
    loader = PairedDataLoader("output/val.txt", tk, samples_per_batch=3, reads_per_sample=4,
                              n_reads_cap_per_sample=20)
    x, lens, sids = next(loader)
    print(f"\nbatch: x.shape={tuple(x.shape)}, attn_len={lens.tolist()}, sample_ids={sids.tolist()}")
    assert x.shape == (12, 64)
    assert sids.unique().numel() >= 2
    print("self-test: PASS")
