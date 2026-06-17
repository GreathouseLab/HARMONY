# HARMONY Architecture V3 — Unified Canonical Reference

**Status:** DRAFT for Nick review
**Date:** 2026-05-18
**Supersedes:** `ARCHITECTURE_V2_REDESIGN.md` (frozen 2026-02-02, drifted across six decision points; see drift catch in `context.json:architecture_v2_drift_catch_2026_05_18`)
**Source of truth:** synthesized from `context.json` canonical sessions through 2026-04-27; this document is now the single architecture reference until the next versioned successor is committed
**Audience:** Nick Chia, internal team, future grant reviewers

---

## How to read this document

Every architectural choice carries a decision-relevance tag:

- **[PHASE-1 ACTIONABLE]** — affects current nanoGPT / autoresearch work directly
- **[GENESIS-MISSION RELEVANT]** — affects scaled-up Genesis architecture, not Phase -1
- **[GRANT-FRAMING]** — affects grant narrative or competitive positioning
- **[WATCH-LIST]** — track but don't act on yet
- **[OPEN — NEEDS NICK]** — unresolved, blocks downstream work

Anything not tagged is descriptive context, not a decision.

---

## 1. Purpose, scope, and what this document supersedes

### 1.1 What HARMONY is — and what it is not

HARMONY is a **perception layer for microbiome sequencing data**. Given raw reads (FASTQ + protocol metadata), it produces per-read embeddings that are invariant to sequencing protocol bias but informative about microbial community structure. Downstream tools (taxonomy callers, abundance estimators, classifiers) consume those embeddings. HARMONY is infrastructure — the layer that makes everything downstream reliable across protocols.

HARMONY is **not** a taxonomy classifier. It is **not** a disease predictor. It does not emit per-sample collapsed embeddings. Those framings collapse to the BiomeGPT failure mode — see §4.3 red lines.

The product claim is cross-protocol invariance: the same simulated microbial community sequenced under two different protocols should produce read embeddings that, after sample aggregation, cluster by community rather than by protocol. The headline evaluation (Tier 3, §6.2) is PERMANOVA on Bray-Curtis distance over read embeddings, with low protocol R² and high community R² as the pass condition.

### 1.2 Phase scope of this document

| Phase | Status | What this document specifies for it |
|---|---|---|
| Phase -1 | Active — nanoGPT feasibility | Sections 2, 3, 6 (Tiers 1–2). The pipeline that runs today. |
| Phase 0 | Next — Genesis Simulator | Section 6 (Tiers 3–4 harness), inverse-simulator ablation (§6.4) |
| Phase 1 | Future — Production model on Argonne A100s | Sections 4, 5. Full multi-protocol training, compositional outputs, dual-head decomposition. |

Phase-1 content from V2 (the dual-head decomposition, Aitchison loss, zero-inflated dropout model, MC Dropout, CRC classification thresholds) is preserved in §5 but explicitly scoped as Genesis-Mission target architecture, not active Phase -1 work. Treating that material as current-state was the central confusion in V2.

### 1.3 What V2 had right and what it had wrong

**Preserved from V2 unchanged (ported to §5 and §6):**
- Compositionality reasoning — Aitchison geometry, why KL + Bray-Curtis was the wrong loss
- Zero-inflated dropout biology — structural vs sampling zeros, GC-driven PCR dropout
- Inverse-simulator-overfitting ablation plan — cross-simulator generalization test
- MC Dropout uncertainty quantification with calibration test

**Replaced in V3 (drift points from V2):**
1. K-mer tokenizer → BPE 4,096 vocab (Nick 2026-02-03)
2. No paired-end handling → sample-stem grouping + R1+RC(R2) joining (2026-04-28 leakage fix)
3. Generic positional encoding → hierarchical RoPE within-read only (2026-04-27)
4. "Transformer or 1D-CNN" → encoder-only transformer for Phase -1 (2026-04-27)
5. Masked Mean Pooling → Set-Transformer attention pool (2026-04-27)
6. `[PROT]` one-hot prepend → FiLM + DANN ablation arms (2026-04-24, 2026-04-27)
7. Per-sample community embedding emitted → per-read embedding only; per-sample collapse RULED OUT (2026-04-27)

---

## 2. Stage 1 — The input pipeline

This is the part of the architecture that is **active** in Phase -1. Everything in this section runs in the autoresearch repo today (modulo the paired-end leakage fix, which is the single non-negotiable change pending). Read it as a description of current code, not future design.

### 2.1 Visual layout

```
INPUT: Raw demultiplexed FASTQ + protocol metadata
       (R1 + R2 per sample, multi-protocol corpus)
                       │
                       ▼
       ┌──────────────────────────────────┐
       │ PREPROCESSING                    │
       │ • Sample-stem grouping           │   [PHASE-1 ACTIONABLE]
       │   (strip _1/_2, split by stem)   │
       │ • Paired-end joining             │
       │   R1 + RC(R2) via [PAIRED_END]   │
       └──────────────────────────────────┘
                       │
                       ▼
       ┌──────────────────────────────────┐
       │ BPE TOKENIZER                    │   [PHASE-1 ACTIONABLE]
       │ • Vocab 4,096                    │
       │ • Pre-seeded: A,C,G,T,N + spec.  │
       │ • Special tokens: [READ_START],  │
       │   [PAIRED_END], [SAMPLE], [PROT] │
       │ • ~75 tokens per 150 bp read     │
       └──────────────────────────────────┘
                       │
                       ▼
       ┌──────────────────────────────────┐         ┌──────────────────────────┐
       │ READ ENCODER                     │ ◄────── │ PROTOCOL CONDITIONING    │
       │ • Encoder-only transformer       │  FiLM   │ • FiLM (γ, β per layer)  │
       │ • RoPE within-read only          │         │ • DANN ablation arm      │
       │ • No positions across reads      │         │   (gradient reversal)    │
       │                                  │         │ • Inputs: platform,      │
       │ Output: per-read embedding       │         │   region, error rate,    │
       └──────────────────────────────────┘         │   read length, depth     │
                       │                            │                          │
                       │                            │ [GENESIS-MISSION RELEVANT│
                       │                            │  — Phase-1 ablation arm] │
                       ▼                            └──────────────────────────┘
       ┌──────────────────────────────────┐
       │ SAMPLE AGGREGATOR                │   [PHASE-1 ACTIONABLE]
       │ • Set-Transformer attention pool │
       │   (Lee et al. 2019)              │
       │ • No cross-read positional info  │
       └──────────────────────────────────┘
                       │
                       ▼
       PRIMARY OUTPUT: Per-read embeddings        [GRANT-FRAMING — red line]
       AUXILIARY OUTPUT: Per-k-mer logits          (used as reconstruction loss)
                                                   ✗ NEVER emit per-sample
                                                     collapsed embedding alone
```

### 2.2 Component-by-component first principles

#### 2.2.1 Preprocessing: sample-stem grouping + paired-end joining

**Root problem.** Illumina convention emits two FASTQ files per physical DNA sample (`_1.fastq` and `_2.fastq`). If a downstream pipeline treats each file as an independent unit, R1 of sample X can land in train and R2 of the same molecule lands in val. The model learns the molecule in train, "validates" against its mate in val, and val_bpb measures memorization rather than generalization. There is also a sample-level leakage layer: even after pairing is fixed, putting any read from sample X in train and any read from sample X in val means the model has already seen sample X's community before val. **Two layers of leak must be closed before any val_bpb number is meaningful.**

**Solution.** Strip the `_1` / `_2` suffix to recover the sample ID, group all files by stem, split by stem (not by file). After grouping, join R1 with the reverse-complement of R2, separated by the `[PAIRED_END]` special token. This gives the model the full molecule as context without new dependencies. (Overlap-detection merging via BBMerge or PEAR remains the standard amplicon practice but adds dependencies — deferred to Phase 0.)

**Consequence.** Any input-stream format change requires BPE retokenization. All 38 historical val_bpb numbers from R1–R3 become formally non-comparable to post-fix runs. This is annoying but unavoidable.

**Tag.** [PHASE-1 ACTIONABLE — non-negotiable before next training run].

#### 2.2.2 Tokenizer: BPE 4,096

**Root problem.** DNA has a four-letter alphabet (plus N for ambiguous bases). The tokenization choice trades off vocabulary size, compression ratio, and what biological structure ends up encoded as a token. Three candidates were on the table: single-nucleotide (Evo2 approach), fixed-length k-mers (k=6 was V2's default), and learned BPE (DNABERT-2 lineage).

**Why not single-nucleotide.** Context lengths blow up: a 150 bp read becomes 150 tokens, a 30 kb HiFi read becomes 30,000 tokens. Wastes attention budget on positional bookkeeping that doesn't carry biological signal.

**Why not fixed-length k-mers.** A k=6 vocabulary is 4,096 fixed tokens, but every k-mer is treated as equally likely a priori. Real DNA has heavy frequency bias (CpG islands, motif repeats, primer-conserved regions). Fixed k-mers don't compress where compression would actually help.

**Why BPE 4,096.** Learned merges concentrate vocabulary where the frequency mass lives. Compression lands at ~75 tokens per 150 bp read — roughly 2x better than single-nucleotide, comparable to fixed k=6 but distributed by data instead of by grid. Tokens become biological objects (motifs, conserved regions) rather than arbitrary windows. Vocabulary size 4,096 matches Karpathy's reduced-platform guidance for DNA's small alphabet.

**Pre-seeded vocabulary.** The DNA alphabet (A, C, G, T, N) and the special tokens (`[READ_START]`, `[PAIRED_END]`, `[SAMPLE]`, `[PROT]`) are reserved before BPE training. This ensures the model can always represent any read, including those with sequencing ambiguities.

**Decided.** Nick 2026-02-03 (BPE over k-mer); vocab size 2026-03-18 (4,096 not 8,192).

**Tag.** [PHASE-1 ACTIONABLE].

**Open subquestion.** Is the current 4,096 vocab trained on iHMP WGS only, or on a stratified multi-protocol mix? DNABERT-2 found corpus composition matters significantly. A WGS-trained vocabulary will tokenize 16S V4 reads inefficiently (and vice versa). **Recommendation:** stratified mix proportional to expected inference-time data distribution. **Status:** [OPEN — NEEDS NICK 2026-05-19].

#### 2.2.3 Read encoder: transformer + RoPE within-read only

**Root problem.** A read is a sequence (within-read order matters: position of an SNP changes meaning); a sample is a multiset (across-read order is arbitrary: shuffling the FASTQ does not change the biology). A single positional encoding across all tokens forces the model to learn from data that across-read order is meaningless. That's a waste of capacity on something architecture can encode for free.

**Solution.** Apply RoPE (rotary position embeddings) within each read — between `[READ_START]` and the next `[READ_START]` or `[SAMPLE]`. Apply no positional encoding at the sample-aggregator level. The aggregator (§2.2.4) is a set operation, not a sequence operation, by construction.

**Why RoPE specifically.** Relative position encoding rather than absolute. Generalizes better to read lengths not seen at training time (essential when long-read corpora enter the mix). Standard in DNABERT-2 and modern protein language models.

**Encoder type.** Standard encoder-only transformer for Phase -1. Hybrid attention designs (DeepSeek V4 CSA/HCA) and state-space models (Hyena, Mamba, Evo2) are deferred to the Genesis Mission scale-up. The Phase -1 question is "does this architectural family learn anything biological from raw reads?" — answering it requires a clean baseline, not a stack of architectural innovations whose individual contributions can't be separated.

**Decided.** 2026-04-27 architecture refresh, decision points 2 (position embedding) and 3 (read encoder).

**Tag.** [PHASE-1 ACTIONABLE].

#### 2.2.4 Sample aggregator: Set-Transformer attention pool

**Root problem.** Once each read has an embedding, how does the model produce a representation of the whole sample? Mean-pooling treats every read as equally informative — but a chimeric read or a contaminant read should contribute less than a clean, abundant taxon read. Attention pooling lets the model learn which reads matter.

**Solution.** Set-Transformer (Lee et al. 2019). Permutation-invariant attention over the read set, no positional encoding across reads, pooled output is a fixed-size representation per sample. Importantly, this is for training-time aggregation and downstream sample-level probing — the primary output remains per-read (§2.2.5), and the aggregator is a head, not the trunk.

**Why not mean-pool.** Phase -1 baseline only. Mean-pool encodes "every read equally important" as an inductive bias, which is wrong for chimera-heavy or low-biomass data. Use mean-pool to establish a floor; expect attention-pool to beat it cleanly.

**Why not hierarchical second-stage transformer.** Genesis Mission scope. The hierarchical option (a second transformer that consumes read embeddings as tokens) is more expressive but requires more compute and obscures what the first-stage encoder is doing. Phase -1 needs interpretable baselines.

**Decided.** 2026-04-27 architecture refresh, decision point 4.

**Tag.** [PHASE-1 ACTIONABLE].

#### 2.2.5 Output representation: per-read embedding only

**Root problem.** What does HARMONY emit? Three options were on the table: per-k-mer logits (every BPE token gets an embedding), per-read embedding (every read gets one vector), or per-sample embedding (every sample gets one vector).

**Per-sample embedding is ruled out.** Collapsing a sample to a single vector is exactly what BiomeGPT does, and it is the failure mode HARMONY is built to avoid. A single per-sample vector pre-bakes whatever question the downstream user has into a single representation; the user cannot rerun the analysis at finer granularity or under a different aggregator. HARMONY's value is providing the perception layer to downstream tools — those tools need read-resolution input, not pre-digested sample summaries.

**Per-k-mer logits.** Useful as an auxiliary reconstruction loss (the MLM training objective produces these by construction). Not the primary product because the downstream user thinks in reads, not tokens.

**Per-read embedding.** The primary output. Each read in the FASTQ becomes a vector. Downstream tools aggregate however they want — sample-level, taxon-level, time-series across longitudinal samples, etc.

**Decided.** 2026-04-27 architecture refresh, decision point 6. Added to red lines same day.

**Tag.** [GRANT-FRAMING — this is the structural commitment that differentiates HARMONY from BiomeGPT-class models]. See §4.3 red lines.

---

## 3. The six architectural decision points — recap

For reference and grant prose. Each links to §2 component above.

| # | Decision point | Choice | Tag |
|---|---|---|---|
| 1 | Tokenization | BPE 4,096 vocab, pre-seeded with DNA alphabet + special tokens | [PHASE-1 ACTIONABLE] |
| 2 | Position embedding | Hierarchical: RoPE within-read, no positions across reads | [PHASE-1 ACTIONABLE] |
| 3 | Read encoder | Standard encoder-only transformer | [PHASE-1 ACTIONABLE] |
| 4 | Sample aggregator | Set-Transformer attention pool (Phase -1); hierarchical transformer (Genesis Mission) | [PHASE-1 ACTIONABLE] |
| 5 | Protocol conditioning | FiLM + DANN ablation arms; see §4 | [GENESIS-MISSION RELEVANT, with Phase-1 ablation] |
| 6 | Output representation | Per-read embedding primary, per-k-mer reconstruction auxiliary, per-sample collapse RULED OUT | [GRANT-FRAMING] |

Status of each comes from `context.json:architecture_refresh_session_2026_04_27.six_architectural_decision_points`. Treat that JSON section as authoritative if this table ever drifts.

---

## 4. Protocol conditioning and the actual HARMONY product

### 4.1 What "protocol conditioning" means and why HARMONY needs it

Microbiome sequencing produces dramatically different reads depending on the protocol used to generate them. The same biological community sampled under 16S V4 Illumina vs WGS PacBio HiFi vs full-length 16S PacBio gives reads that don't look alike — different lengths, different error profiles, different region coverage, different abundance distortions from PCR. **This is the central reason cross-study microbiome meta-analysis fails today.** Downstream tools see the protocol bias as biology.

HARMONY's promise: take the protocol as a known input alongside the reads, and produce embeddings that strip the protocol signature from the representation while preserving the biology. The protocol metadata is the conditioning signal; the architecture has to use it correctly.

### 4.2 FiLM + DANN — two arms, two complementary jobs

**FiLM (Feature-wise Linear Modulation).** Tells the encoder *what protocol this read came from*. The protocol metadata is encoded as a vector (platform one-hot, region one-hot, plus continuous features: expected error rate, read length, GC bias, depth). That vector projects to per-layer γ and β parameters that scale and shift the encoder's intermediate activations. The encoder sees the protocol; its computation is protocol-aware.

**DANN (Domain-Adversarial Neural Network) — the ablation arm.** Adds a gradient-reversal head that *tries to predict* the protocol from the read embedding. The encoder is trained to make that prediction fail — to produce embeddings from which the protocol cannot be recovered. This is the explicit protocol-invariance objective.

**Why both.** FiLM alone gives the model the information it needs to compensate for protocol bias but does not force it to do so. DANN alone forces invariance without giving the model the information needed to compensate intelligently. Together, FiLM says "here is the protocol, compensate for it" and DANN says "and the compensation must be thorough enough that I can't tell what protocol it was."

**Decided.** 2026-04-24 DeepSeek V4 review (DANN added as ablation arm); 2026-04-27 architecture refresh, decision point 5.

**Tag.** [GENESIS-MISSION RELEVANT — full deployment]. [PHASE-1 ACTIONABLE — ablation experiment to validate the design].

**Open subquestion.** `[PROT]` token vs FiLM-only injection. The `[PROT]` special token from §2.2.2 is currently the sensor-signal channel — it prepends to every read in the encoder input. FiLM injects the same information at every layer via γ/β. Are both needed, or is one redundant? **Recommendation:** keep both; the redundancy cost is small (one extra token per read), the failure cost of removing either is unknown. Flag as [WATCH-LIST] for Genesis Mission ablation.

### 4.3 Red lines

These are commitments that bound the design space. Each red line was established for a specific reason; crossing one without explicit reason and team agreement is regression.

1. **HARMONY does not emit per-sample collapsed embeddings as primary output.** Collapses to BiomeGPT failure mode. Added 2026-04-27. *Tag: [GRANT-FRAMING].*

2. **HARMONY is not a taxonomy classifier.** It is a perception layer; downstream tools (Kraken2, MetaPhlAn, microbiomeMASST) consume its embeddings. Framing HARMONY as taxonomy collapses its differentiation and competes with mature tools we'd lose against. Established in program.md 2026-04-27. *Tag: [GRANT-FRAMING].*

3. **HARMONY does not predict disease state, body site, or any clinical label.** Those are downstream tasks on the embeddings, not the embeddings themselves. Conflating the perception layer with the classifier is exactly the conflation we are differentiating against. *Tag: [GRANT-FRAMING].*

4. **RF baseline on iHMP IBD on the same substrate is the floor.** If HARMONY embeddings + linear probe don't beat a random forest on the same substrate, HARMONY hasn't earned its complexity. This is the published-in-the-paper benchmark. *Tag: [PHASE-1 ACTIONABLE].*

---

## 5. Compositionally-correct outputs (Phase 0 / Phase 1 scope)

The material in this section is preserved from V2 with minor updating. It describes the **target architecture** at Genesis Mission scale, not current Phase -1 work. Treat it as design intent for the production HARMONY model, after Phase -1 feasibility resolves.

### 5.1 Compositional geometry — why the V1 loss was wrong

Microbiome composition vectors live on the simplex (non-negative, sum to one). Euclidean operations on the simplex are not meaningful: adding two compositions doesn't give a valid composition, the Euclidean distance between two compositions can be small when the underlying biology is dramatically different, and Bray-Curtis (a semi-metric, violates triangle inequality) mixes geometries incoherently.

The Aitchison geometry treats the simplex correctly. The centered log-ratio (CLR) transform maps simplex points to unconstrained real space, where standard linear algebra applies. The Aitchison distance is the Euclidean distance in CLR space. The model emits unconstrained CLR logits; softmax(logits) recovers the simplex composition; the training loss is L2 in CLR space.

This was V2's central correct insight and carries forward unchanged.

**Tag.** [GENESIS-MISSION RELEVANT].

### 5.2 Zero-inflated dropout — separating "absent" from "undetected"

Observed zeros in microbiome data have two sources: structural zeros (taxon genuinely not in the community) and sampling zeros (taxon present but not detected due to insufficient depth, PCR dropout from GC bias or primer mismatch, below detection limit). A single composition head cannot disambiguate them — both look like zero.

The architecture separates the prediction: a presence/absence head predicts P(structural zero) per taxon; a composition head predicts CLR logits for the present taxa; the final composition multiplies them. The model uses protocol conditioning, library size, and community context to make the call. A taxon absent from a deep WGS sample is more likely truly absent than the same taxon absent from a shallow 16S V4 sample.

**Tag.** [GENESIS-MISSION RELEVANT].

### 5.3 Dual-head decomposition

At Genesis Mission scale, the model emits five outputs:

1. **Corrected reads** (Head 1A) — error-corrected sequences. Solved problem; not the selling point.
2. **Chimera probability** (Head 1B) — per-read flag.
3. **Composition** (Head 2A × 2B) — CLR-space logits gated by presence probabilities.
4. **Per-taxon presence** (Head 2B) — zero-inflation handling.
5. **Per-read embedding** (Head 2C) — the §2.2.5 output, used for cross-protocol alignment via contrastive loss.

Phase -1 does not implement this decomposition. Phase -1 trains the encoder alone on MLM (or next-token; see §6.5 open question) and measures whether the resulting embeddings carry biological signal at all. The heads come on-line in Phase 0 once the Genesis Simulator produces labeled training pairs.

**Tag.** [GENESIS-MISSION RELEVANT].

### 5.4 Uncertainty quantification — MC Dropout with calibration

The composition head retains dropout at inference time. Running T forward passes (T ≈ 30) gives a distribution of predictions per taxon; the variance is the epistemic uncertainty. The 95% confidence interval is validated by a calibration test: across many samples, the true abundance should fall in the nominal CI at the nominal rate. Systematic over- or under-confidence triggers recalibration.

This is preserved from V2 unchanged.

**Tag.** [GENESIS-MISSION RELEVANT].

---

## 6. Training strategy and four-tier evaluation framework

### 6.1 Phase -1 training (current, autoresearch_llm.py loop)

Today's loop trains the encoder + sample aggregator on a single-protocol corpus (140 iHMP WGS samples) with an MLM-style next-token objective. The autoresearch framework varies hyperparameters and tracks val_bpb. Current champion (r63/exp95) holds val_bpb-joint at 1.8058; baseline random is ~2.0 for the 4,096-token vocabulary.

**Two caveats on val_bpb:**

- **Proxy, not goal.** val_bpb tracks how well the model predicts held-out tokens. It does *not* measure whether the embeddings encode biological structure. A model can have a great val_bpb and still have learned only the k-mer frequency distribution. The four-tier framework (§6.2) exists to close that gap.
- **Currently leaky.** All 38 historical val_bpb numbers are non-comparable to post-paired-end-fix runs (see §2.2.1). The pre-fix numbers measured memorization across paired reads of the same molecule. The actual signal is unknown until the fix is deployed and the corpus is retokenized.

**Tag.** [PHASE-1 ACTIONABLE].

### 6.2 Four-tier evaluation framework

The evaluation framework was settled 2026-04-27 to address the val_bpb-as-proxy problem. Each tier answers a different question; Phase -1 needs at least Tier 1 + Tier 2 passing before moving on.

**Tier 1 — Sanity checks. [PHASE-1 ACTIONABLE — currently passing]**
- Loss decreases monotonically
- Val below random baseline
- k-mer distribution recovery
- *Status:* R1–R3 pass. Does not prove the model learned biology. The current Phase -1 stopping point.

**Tier 2 — Probing tasks. [PHASE-1 ACTIONABLE — missing layer]**

Train a frozen-feature linear probe on top of the pretrained read embeddings. This is Nick's "input/Y → predict X" framing.
- **Read-to-taxon classification** (synthetic ground truth from Genesis Simulator, or labeled MBQC subset)
- **Read-to-sample-property classification** (body site, disease status, cohort — easier at sample level after attention-pooling)
- **Read-to-sample retrieval** (Nick's suggestion; sanity check, not goal — high accuracy may indicate per-sample memorization rather than generalizable structure)

*Highest-value next experiment:* Tier 2 probing on the existing R2 winner. No new training. Converts R1–R3 from a tuning artifact into feasibility evidence.

**Tier 3 — Bias-correction evaluations. [PHASE-1 ACTIONABLE / GRANT-FRAMING — HARMONY's actual deliverable]**

*The single most important evaluation HARMONY will ever run:* protocol-invariance under PERMANOVA. Same simulated community, two simulated protocols, embed all reads, compute PERMANOVA on Bray-Curtis distance over the read embeddings. Low protocol R², high community R² = HARMONY is working.

Complementary:
- Cross-protocol abundance correlation (HARMONY embeddings → scikit-bio CLR → species-level vector → Spearman across protocols)
- Mock community recovery on ZymoBIOMICS or EcoFAB ground truth — does inferred composition match better than DADA2 + standard tools?

Harness can be coded today against synthetic toy data so it's ready when full Genesis Simulator paired outputs are available.

**Tier 4 — Benchmark comparisons. [GRANT-FRAMING — publication-required]**
- RF baseline on iHMP IBD on the same substrate (red line floor)
- GAN-GMHI 34-study benchmark
- Differentiator paragraphs vs BiomeGPT (LOW threat — pre-processed data), Read2Pheno (architectural neighbor — CNN+RNN+attention on raw reads), DeepMicrobes

### 6.3 Phased training (Genesis Mission scope)

When the dual-head decomposition (§5.3) comes online in Phase 0/1, training proceeds in four sub-phases. Preserved from V2:

1. Read heads only — train encoder + error corrector + chimera detector. Freeze composition head.
2. Composition head — freeze encoder, train composition + zero-inflation on simple communities.
3. Joint fine-tuning — unfreeze all, reduced LR.
4. Alignment loss — add contrastive InfoNCE with multi-protocol batches.

**Tag.** [GENESIS-MISSION RELEVANT].

### 6.4 Inverse-simulator-overfitting ablation (Phase 0 prerequisite)

If the model only learns to invert the Genesis Simulator's specific error model, it has memorized an artifact lookup table, not a general denoising function. The ablation: train on Genesis Simulator output, test on reads generated from InSilicoSeq, ART, and CAMISIM — all using the same ground-truth communities. Performance on the alternate simulators must come within 20% of in-distribution performance; >50% degradation is overfitting.

Run DADA2 on all four test sets as a control: if DADA2 also degrades across simulators, the problem is simulator difference rather than model overfitting.

Preserved from V2.

**Tag.** [GENESIS-MISSION RELEVANT — gates Phase 0 → Phase 1 transition].

### 6.5 Open questions — flagged for Nick

| # | Question | Why it matters | Status |
|---|---|---|---|
| 1 | MLM vs causal next-token vs Nick's "next-read prediction"? | Three documents (V2, drafted program.md, Nick 2026-02-03 note) name different training objectives. val_bpb is computed differently under each. | [OPEN — NEEDS NICK] |
| 2 | BPE 4,096 vocab — trained on iHMP WGS only or stratified multi-protocol corpus? | Determines whether multi-modal handling has been exercised or merely deferred. R1–R3 results say nothing about cross-data-type generalization if the answer is single-protocol. | [OPEN — NEEDS NICK] |
| 3 | Primer-strip preprocessing in current pipeline? | Every 16S V4 read shares ~40 bp primer-conserved sequence. Without cutadapt-style stripping, MLM trivially memorizes the conserved region and val_bpb looks great while learning nothing. | [OPEN — NEEDS NICK] |
| 4 | Long-read chunking for HiFi reads at 30 kb? | At ~75 tokens per 150 bp, a 30 kb read is ~15,000 tokens — beyond reasonable nanoGPT context. Recommendation: chunk into 512–1024 token windows with overlap, add `[CHUNK_BOUNDARY]` special token. Phase -1 workaround; long-context architecture deferred to Genesis Mission. | [WATCH-LIST] |

---

## Appendix A — Pointers into context.json

This document synthesizes content from the following `context.json` sections. If V3 ever drifts from canonical state, those sections are authoritative; cross-check before relying on this document for active decisions.

| §V3 | Canonical context.json source |
|---|---|
| §1.3 drift table | `architecture_v2_drift_catch_2026_05_18.input_pipeline_drift_points` |
| §2.2.1 paired-end | `follow_up_clarifications_2026_04_28.data_leakage_clarification` |
| §2.2.2 BPE 4,096 | `architecture_refresh_session_2026_04_27.six_architectural_decision_points.1_tokenization` |
| §2.2.3 RoPE | `architecture_refresh_session_2026_04_27.six_architectural_decision_points.2_position_embedding` |
| §2.2.4 Set-Transformer | `architecture_refresh_session_2026_04_27.six_architectural_decision_points.4_sample_aggregator` |
| §2.2.5 per-read output | `architecture_refresh_session_2026_04_27.six_architectural_decision_points.6_output_representation` |
| §4.2 FiLM + DANN | `architecture_refresh_session_2026_04_27.six_architectural_decision_points.5_protocol_conditioning`, `deepseek_v4_architecture_review_2026_04_24_summary` |
| §4.3 red lines | `prompts.md:Red Lines` (point 7, added 2026-04-27) |
| §6.2 evaluation tiers | `architecture_refresh_session_2026_04_27.four_tier_evaluation_framework` |

## Appendix B — Glossary

| Term | Definition |
|---|---|
| BPE | Byte-Pair Encoding. Learned subword tokenization. |
| DANN | Domain-Adversarial Neural Network. Gradient-reversal head used to force domain invariance. |
| FiLM | Feature-wise Linear Modulation. Per-layer γ/β conditioning from external metadata. |
| MLM | Masked Language Modeling. BERT-style training objective (predict masked tokens from context). |
| PERMANOVA | Permutational Multivariate Analysis of Variance. Tests whether group labels explain variance in a distance matrix. HARMONY's headline evaluation. |
| RoPE | Rotary Position Embeddings. Relative position encoding that generalizes to unseen sequence lengths. |
| Set-Transformer | Attention-based architecture for set-valued inputs (Lee et al. 2019). Permutation-invariant by construction. |
| val_bpb | Validation bits-per-byte. Phase -1's current proxy metric. Caveats in §6.1. |

---

*Architecture V3 — HARMONY / Microbiome Genesis Project*
*Drafted 2026-05-18*
*Panel: Technical Architect + ML Researcher + Microbiome Bioinformatician + Scientific Critic*
*Next revision after Nick reads and the four [OPEN — NEEDS NICK] items resolve.*
