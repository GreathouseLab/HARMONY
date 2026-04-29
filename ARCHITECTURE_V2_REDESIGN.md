# Microbiome Genesis Framework: Architecture V2 — Dual-Head Redesign

## Status: DRAFT — Technical Architecture Review
**Date**: 2026-02-02  
**Primary Claim**: Synthetic-trained models recover true composition more accurately than DADA2/Deblur on mock communities  
**Output**: Dual-headed — read-level error correction AND composition-level bias correction  
**Compositionality**: Robust Aitchison distance (rCLR)  
**Dropout**: Zero-inflated model  

---

## 1. Problem Decomposition

The artifact chain corrupts data at two fundamentally different levels. The architecture must address both, but with separate mechanisms matched to the nature of each corruption.

### Read-Level Artifacts (affect individual sequences)

| Artifact | What happens | Reversible at read level? |
|----------|-------------|--------------------------|
| Sequencing errors | Nucleotide substitutions, insertions, deletions | **Yes** — correct the sequence |
| Chimeras | Two parent reads fuse into a fake hybrid | **Yes** — detect and flag/remove |
| Adapter contamination | Non-biological sequence appended | **Yes** — trim |

### Community-Level Artifacts (affect relative abundances)

| Artifact | What happens | Reversible at read level? |
|----------|-------------|--------------------------|
| PCR amplification bias | Over/under-representation of taxa based on GC content, primer binding | **No** — reads are correct, proportions are wrong |
| Dropout | Low-abundance taxa absent entirely | **No** — missing reads cannot be reconstructed |
| Library size variation | Different total read counts across samples | **No** — affects sampling depth |
| Region bias (16S) | Different variable regions capture different taxa | **No** — unobserved taxa cannot be inferred from reads alone |

**Architectural implication**: The model requires two heads operating at different granularities, connected by a shared representation.

---

## 2. Revised Architecture Diagram

```
INPUT: Raw demultiplexed FASTQ reads + Protocol metadata
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: READ PROCESSING                         │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   K-MER TOKENIZER                            │   │
│  │  Raw read → overlapping k-mers → vocabulary indices          │   │
│  │  k=6 (prototype) or k=8 (Genesis-ready)                     │   │
│  │  Input:  "ACGTACGTNN..." (variable length)                  │   │
│  │  Output: [idx_1, idx_2, ..., idx_L] (fixed vocabulary)      │   │
│  └─────────────────────────┬────────────────────────────────────┘   │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   READ ENCODER (shared backbone)             │   │
│  │  K-mer Embedding (learnable, dim=E_read) + Positional Enc.  │   │
│  │                          │                                   │   │
│  │  Transformer Encoder (N_read layers, e.g. 4)                │   │
│  │  OR 1D-CNN stack (for rapid prototyping)                    │   │
│  │                          │                                   │   │
│  │  Masked Mean Pooling → read_embedding [E_read]              │   │
│  └──────────┬───────────────┴─────────────────┬─────────────────┘   │
│             │                                 │                      │
│             ▼                                 ▼                      │
│  ┌─────────────────────┐         ┌──────────────────────────┐      │
│  │  HEAD 1A: READ      │         │  HEAD 1B: CHIMERA        │      │
│  │  ERROR CORRECTOR    │         │  DETECTOR                │      │
│  │                     │         │                          │      │
│  │  Transformer        │         │  MLP → sigmoid           │      │
│  │  Decoder (N layers) │         │                          │      │
│  │  Autoregressive     │         │  Output: P(chimera)      │      │
│  │  k-mer prediction   │         │  per read [0, 1]         │      │
│  │                     │         │                          │      │
│  │  Output: corrected  │         │  Training target:        │      │
│  │  k-mer sequence     │         │  binary label from       │      │
│  │                     │         │  simulator                │      │
│  └─────────────────────┘         └──────────────────────────┘      │
│                                                                      │
│  Per-read outputs:                                                  │
│  - Corrected read sequence (Head 1A)                               │
│  - Chimera probability (Head 1B)                                   │
│  - Read embedding vector (from shared encoder, passed to Stage 2)  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               │  Set of read embeddings
                               │  [batch, num_reads, E_read]
                               │  (chimera-flagged reads down-weighted)
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: COMMUNITY ENCODING                       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                CHIMERA-AWARE WEIGHTING                       │   │
│  │  w_i = (1 - P_chimera_i) * attention_weight_i               │   │
│  │  Soft filtering: chimeric reads contribute less              │   │
│  └─────────────────────────┬────────────────────────────────────┘   │
│                            │                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                PROTOCOL CONDITIONING                         │   │
│  │  Protocol one-hot → learned embedding [E_prot]              │   │
│  │  Prepended as special [PROT] token to read set              │   │
│  │                                                              │   │
│  │  Protocols encoded:                                          │   │
│  │  - 16S V4 Illumina                                          │   │
│  │  - 16S V3-V4 Illumina                                       │   │
│  │  - 16S V1-V4 (454/Illumina)                                 │   │
│  │  - WGS short-read Illumina                                  │   │
│  │  - WGS PacBio HiFi                                          │   │
│  │  - WGS ONT                                                   │   │
│  │  - Illumina SLR (Tell-Seq)                                  │   │
│  │  - Full-length 16S PacBio                                   │   │
│  └─────────────────────────┬────────────────────────────────────┘   │
│                            │                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                COMMUNITY TRANSFORMER                         │   │
│  │  Cross-read attention (N_comm layers, e.g. 4)               │   │
│  │  Reads "see" each other to infer community patterns         │   │
│  │                          │                                   │   │
│  │  Chimera-Aware Attention-Weighted Pooling                   │   │
│  │  weights = softmax(MLP(read_embeddings)) * (1-P_chimera)    │   │
│  │  community_embedding = Σ(weights * read_embeddings)         │   │
│  └─────────────────────────┬────────────────────────────────────┘   │
│                            │                                         │
│  Output: community_embedding [batch, E_comm]                        │
│          + per-read attention weights (interpretability)            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────────┐
           │                   │                       │
           ▼                   ▼                       ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│ HEAD 2A:         │ │ HEAD 2B:         │ │ HEAD 2C:             │
│ COMPOSITION      │ │ ZERO-INFLATION   │ │ LATENT EMBEDDING     │
│ PREDICTOR        │ │ DETECTOR         │ │ (for alignment)      │
│                  │ │                  │ │                      │
│ MLP layers       │ │ MLP → per-taxon  │ │ Linear → L2 norm    │
│                  │ │ sigmoid          │ │                      │
│ Output:          │ │                  │ │ Output:              │
│ z ∈ R^D          │ │ Output:          │ │ e ∈ S^(L-1)         │
│ (CLR-space       │ │ π ∈ [0,1]^D     │ │ (unit hypersphere)   │
│  logits)         │ │ P(structural     │ │                      │
│                  │ │  zero) per taxon │ │ Protocol-invariant   │
│ Back-transform:  │ │                  │ │ embedding for        │
│ p = softmax(z)   │ │ Separate "who's  │ │ cross-study          │
│                  │ │ there" from      │ │ alignment            │
│                  │ │ "how much"       │ │                      │
└────────┬─────────┘ └────────┬─────────┘ └──────────┬───────────┘
         │                    │                       │
         ▼                    ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     COMBINED OUTPUT                               │
│                                                                   │
│  Final composition:                                              │
│  p_final_i = (1 - π_i) × softmax(z)_i                          │
│                                                                   │
│  Where:                                                          │
│  - π_i ≈ 1  → taxon i predicted absent (structural zero)        │
│  - π_i ≈ 0  → taxon i predicted present, abundance = softmax(z)_i│
│  - Uncertainty: MC Dropout or ensemble for confidence intervals  │
│                                                                   │
│  Outputs:                                                        │
│  1. Corrected reads (Head 1A) — error-corrected sequences       │
│  2. Chimera flags (Head 1B) — per-read chimera probability      │
│  3. Composition vector (Head 2A × 2B) — CLR-space + simplex     │
│  4. Presence probabilities (Head 2B) — per-taxon detection       │
│  5. Community embedding (Head 2C) — for alignment/retrieval      │
│  6. Uncertainty estimates — via MC Dropout                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Loss Functions — Compositionally Correct

### 3.1 Why the Original Loss Was Wrong

The V1 architecture used: `L = λ₁·KL(pred‖true) + λ₂·BrayCurtis(pred, true) + λ₃·Sparsity`

Problems:
1. **KL divergence** operates on probability distributions in Euclidean space — it does not respect the simplex geometry of compositional data
2. **Bray-Curtis** is a semi-metric (violates triangle inequality) and is sensitive to rare taxa subsetting
3. **Mixing geometries** is incoherent — KL assumes information-theoretic space, Bray-Curtis assumes ecological space
4. Neither accounts for the closure constraint properly

### 3.2 Revised Loss: Robust Aitchison + Zero-Inflated Components

The total loss decomposes into four components, each operating at the correct level:

```
L_total = λ₁·L_composition + λ₂·L_zero_inflation + λ₃·L_read_correction + λ₄·L_chimera + λ₅·L_alignment

Where:
  λ₁ = 1.0   (primary objective)
  λ₂ = 0.5   (important for handling sparsity)
  λ₃ = 0.3   (secondary objective — solved problem)
  λ₄ = 0.2   (auxiliary task)
  λ₅ = 0.3   (alignment objective — for cross-protocol)
```

#### L_composition: Robust Aitchison Distance

```python
def robust_aitchison_loss(z_pred, p_true, pi_pred):
    """
    Composition loss in CLR space using robust Aitchison distance.
    
    Args:
        z_pred: predicted CLR logits [batch, D]
        p_true: true composition [batch, D] (from simulator, may contain zeros)
        pi_pred: predicted zero-inflation probabilities [batch, D]
    
    Returns:
        Aitchison distance computed over non-zero taxa
    """
    # Identify taxa present in ground truth
    present_mask = (p_true > 0).float()  # [batch, D]
    
    # For present taxa: compute CLR of true composition
    # Use multiplicative replacement for numerical stability
    p_true_replaced = multiplicative_replacement(p_true, delta=0.65)
    clr_true = clr_transform(p_true_replaced)  # [batch, D]
    
    # Aitchison distance: Euclidean distance in CLR space
    # Only computed over taxa present in ground truth
    diff = (z_pred - clr_true) * present_mask
    aitchison_dist = torch.sqrt((diff ** 2).sum(dim=-1))  # [batch]
    
    return aitchison_dist.mean()


def clr_transform(x):
    """Centered log-ratio transform."""
    log_x = torch.log(x)
    geometric_mean = log_x.mean(dim=-1, keepdim=True)
    return log_x - geometric_mean


def multiplicative_replacement(x, delta=0.65):
    """
    Replace zeros with small value, adjust non-zeros to maintain closure.
    Following Martín-Fernández et al. (2003).
    delta: replacement value (fraction of detection limit)
    """
    n_zeros = (x == 0).sum(dim=-1, keepdim=True).float()
    D = x.shape[-1]
    
    # Replacement value scaled by number of zeros
    replacement = delta / (D ** 2)
    
    x_replaced = x.clone()
    zero_mask = (x == 0)
    x_replaced[zero_mask] = replacement
    
    # Adjust non-zeros to maintain unit sum
    adjustment = 1 - (n_zeros * replacement)
    x_replaced[~zero_mask] = x_replaced[~zero_mask] * adjustment / x_replaced[~zero_mask].sum()
    
    return x_replaced
```

**Why robust Aitchison**: Standard Aitchison distance requires all values > 0 (log is undefined for zero). The robust variant (Martino et al. 2019) handles zeros either by multiplicative replacement or by computing distances only over observed (non-zero) features using matrix completion (RPCA). For our supervised setting where we know the ground truth, multiplicative replacement on the ground truth plus masking to the present taxa is the cleanest approach.

#### L_zero_inflation: Binary Cross-Entropy on Presence/Absence

```python
def zero_inflation_loss(pi_pred, p_true):
    """
    Binary cross-entropy for taxon presence/absence prediction.
    
    Args:
        pi_pred: predicted P(structural zero) per taxon [batch, D]
        p_true: true composition [batch, D]
    
    Returns:
        BCE loss
    """
    # Ground truth: 1 if taxon truly absent, 0 if present
    true_absent = (p_true == 0).float()
    
    return F.binary_cross_entropy(pi_pred, true_absent)
```

#### L_read_correction: Sequence-Level Cross-Entropy

```python
def read_correction_loss(kmer_logits_pred, kmer_indices_true):
    """
    Cross-entropy loss for corrected k-mer sequence prediction.
    
    Args:
        kmer_logits_pred: predicted k-mer logits [batch, num_reads, seq_len, vocab_size]
        kmer_indices_true: true (uncorrupted) k-mer indices [batch, num_reads, seq_len]
    
    Returns:
        Per-position cross-entropy averaged over reads
    """
    B, R, L, V = kmer_logits_pred.shape
    logits = kmer_logits_pred.reshape(-1, V)
    targets = kmer_indices_true.reshape(-1)
    
    return F.cross_entropy(logits, targets)
```

#### L_chimera: Binary Cross-Entropy on Chimera Detection

```python
def chimera_loss(chimera_pred, chimera_true):
    """
    Args:
        chimera_pred: predicted P(chimera) per read [batch, num_reads]
        chimera_true: binary chimera labels from simulator [batch, num_reads]
    """
    return F.binary_cross_entropy(chimera_pred, chimera_true)
```

#### L_alignment: Contrastive Loss (InfoNCE)

```python
def alignment_loss(embeddings, community_labels, temperature=0.07):
    """
    InfoNCE contrastive loss.
    Same community under different protocols should cluster;
    different communities should separate.
    
    Only active when training with multi-protocol batches.
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)
    
    # Compute similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Positive pairs: same community, different protocol
    positive_mask = (community_labels.unsqueeze(0) == community_labels.unsqueeze(1))
    positive_mask.fill_diagonal_(False)
    
    # InfoNCE loss
    loss = -torch.log(
        (sim.exp() * positive_mask).sum(dim=-1) / 
        sim.exp().sum(dim=-1)
    )
    
    return loss.mean()
```

---

## 4. Zero-Inflated Dropout Model

### 4.1 The Biological Model

Observed zeros in microbiome data arise from two distinct processes:

1. **Structural zeros**: The taxon is genuinely absent from the community (true biological absence)
2. **Sampling zeros**: The taxon is present but not detected due to:
   - Insufficient sequencing depth (Poisson sampling)
   - PCR dropout (primer mismatch, GC bias at extremes)
   - Below detection limit

### 4.2 Simulator Implementation

```python
class ZeroInflatedDropout:
    """
    Zero-inflated dropout model for the simulator.
    
    Combines structural absence with sampling-based dropout.
    """
    
    def __init__(self, 
                 structural_zero_rate=0.7,     # fraction of reference taxa absent
                 detection_limit=1e-5,          # minimum detectable abundance
                 pcr_dropout_gc_threshold=0.3,  # GC < this → high dropout risk
                 pcr_dropout_gc_upper=0.7):     # GC > this → high dropout risk
        self.structural_zero_rate = structural_zero_rate
        self.detection_limit = detection_limit
        self.gc_low = pcr_dropout_gc_threshold
        self.gc_high = pcr_dropout_gc_upper
    
    def apply(self, true_abundances, gc_contents, library_size, primer_binding_scores):
        """
        Apply zero-inflation to true abundances.
        
        Args:
            true_abundances: [D] true relative abundances (sum to 1)
            gc_contents: [D] GC content per taxon
            library_size: int, total reads to generate
            primer_binding_scores: [D] primer binding efficiency per taxon
        
        Returns:
            observed_abundances: [D] abundances after dropout
            dropout_mask: [D] boolean, True = dropped out
            dropout_type: [D] 'structural', 'sampling', 'pcr', or 'none'
        """
        D = len(true_abundances)
        dropout_type = np.full(D, 'none', dtype=object)
        observed = true_abundances.copy()
        
        # 1. Structural zeros (already in true_abundances as 0)
        structural = (true_abundances == 0)
        dropout_type[structural] = 'structural'
        
        # 2. PCR dropout — taxa with extreme GC or poor primer binding
        # P(pcr_dropout) increases at GC extremes and low primer binding
        gc_risk = np.where(
            (gc_contents < self.gc_low) | (gc_contents > self.gc_high),
            0.3,  # 30% additional dropout risk at extremes
            0.01  # 1% baseline
        )
        primer_risk = (1 - primer_binding_scores) * 0.5  # up to 50% dropout
        pcr_dropout_prob = np.minimum(gc_risk + primer_risk, 0.8)
        pcr_dropout = np.random.binomial(1, pcr_dropout_prob).astype(bool) & ~structural
        observed[pcr_dropout] = 0
        dropout_type[pcr_dropout] = 'pcr'
        
        # 3. Sampling dropout — Poisson sampling from library
        # Expected reads = abundance * library_size
        # P(zero reads) = exp(-abundance * library_size) for Poisson
        expected_reads = observed * library_size
        sampling_dropout = np.random.poisson(expected_reads) == 0
        sampling_dropout = sampling_dropout & ~structural & ~pcr_dropout & (true_abundances > 0)
        observed[sampling_dropout] = 0
        dropout_type[sampling_dropout] = 'sampling'
        
        # 4. Detection limit — below threshold → zero
        below_limit = (observed > 0) & (observed < self.detection_limit)
        observed[below_limit] = 0
        dropout_type[below_limit] = 'sampling'
        
        # Renormalize non-zero taxa
        if observed.sum() > 0:
            observed = observed / observed.sum()
        
        dropout_mask = (true_abundances > 0) & (observed == 0)
        
        return observed, dropout_mask, dropout_type
```

### 4.3 How the Model Learns to Undo Dropout

During training, the model sees simulated reads (with dropout applied) and the true composition (before dropout). The zero-inflation head learns:

- **π_i close to 1**: "I don't see evidence for taxon i in these reads, AND the protocol/library size suggests it could easily have dropped out" → taxon might be present but undetected
- **π_i close to 0**: "I see clear evidence for taxon i" OR "The library size was deep enough that absence is likely real" → taxon is present/genuinely absent

The key insight: **the model uses protocol conditioning + library size + community context to distinguish structural zeros from sampling zeros.** A taxon absent from a deep WGS sample is more likely truly absent than the same taxon absent from a shallow 16S V4 sample.

---

## 5. Protocol Conditioning — Detailed Design

### 5.1 Protocol Representation

Protocols are not just categorical labels. They encode specific physical/chemical processes that determine bias structure:

```python
class ProtocolEncoder(nn.Module):
    """
    Encodes protocol information as a rich embedding.
    
    Protocol is represented as:
    1. Categorical: platform type (one-hot)
    2. Continuous: expected error rate, read length, primer GC
    3. Regional: which 16S variable regions are captured (if applicable)
    """
    
    def __init__(self, E_prot=64):
        super().__init__()
        
        # Categorical embeddings
        self.platform_embed = nn.Embedding(8, 32)  # 8 platform types
        self.region_embed = nn.Embedding(10, 16)    # V1-V9 + WGS
        
        # Continuous feature projection
        self.continuous_proj = nn.Linear(4, 16)  # error_rate, read_length, gc_bias, depth
        
        # Combine
        self.combine = nn.Linear(32 + 16 + 16, E_prot)
    
    def forward(self, platform_id, region_id, continuous_features):
        plat = self.platform_embed(platform_id)
        reg = self.region_embed(region_id)
        cont = self.continuous_proj(continuous_features)
        
        return self.combine(torch.cat([plat, reg, cont], dim=-1))
```

### 5.2 Conditioning Mechanism

The protocol embedding is injected at two points:

1. **Community encoder input**: Prepended as a special `[PROT]` token, so cross-read attention can condition on protocol
2. **Composition head**: Concatenated with community embedding before the MLP, so bias correction is protocol-specific

```python
# In Community Encoder:
prot_token = self.protocol_encoder(protocol_metadata)  # [batch, E_prot]
prot_token = prot_token.unsqueeze(1)                    # [batch, 1, E_prot]
# Project to same dim as read embeddings
prot_token = self.prot_proj(prot_token)                 # [batch, 1, E_read]

# Prepend to read embeddings
read_set = torch.cat([prot_token, read_embeddings], dim=1)  # [batch, 1+R, E_read]

# Cross-read attention includes protocol token
community_out = self.community_transformer(read_set)
```

---

## 6. Compositionality: Working in the Aitchison Simplex

### 6.1 The Mathematical Framework

Compositional data lives on the simplex S^D = {x ∈ R^D : x_i > 0, Σx_i = 1}. The simplex is NOT a Euclidean space — Euclidean operations (addition, subtraction, norms) are not meaningful.

The Aitchison geometry provides the correct operations:

| Euclidean Operation | Aitchison Equivalent |
|--------------------|---------------------|
| Addition | Perturbation: x ⊕ y = C(x₁y₁, ..., x_Dy_D) |
| Scalar multiplication | Power: α ⊙ x = C(x₁^α, ..., x_D^α) |
| Distance | d_A(x,y) = ‖clr(x) - clr(y)‖₂ |
| Inner product | ⟨x,y⟩_A = ⟨clr(x), clr(y)⟩ |

Where C(·) is the closure operator (normalize to sum to 1).

### 6.2 Model Output Space

The composition head outputs **unconstrained logits z ∈ R^D**. These are mathematically equivalent to CLR coordinates:

```
z_i = log(p_i / G(p))     where G(p) = (∏p_i)^(1/D)

To recover simplex probabilities:
p = softmax(z)             (inverse CLR via softmax)
```

This is elegant because:
- The model predicts in unconstrained real space (easy for neural nets)
- softmax(z) exactly recovers the simplex composition
- Loss is computed as Euclidean distance in z-space = Aitchison distance in simplex
- No need for explicit simplex constraints during training

### 6.3 Handling Zeros in Ground Truth

The CLR transform is undefined for zeros (log(0) = -∞). Our training data (synthetic) has known ground truth compositions that include true zeros.

**Strategy**: Separate the composition prediction from presence/absence prediction.

```python
class CompositionOutput(nn.Module):
    """
    Compositionally-correct output with zero-inflation.
    
    Predicts: (1) which taxa are present, (2) their relative abundances
    in CLR space.
    """
    
    def __init__(self, E_comm, D_taxa):
        super().__init__()
        
        # Presence/absence head (zero-inflation)
        self.presence_head = nn.Sequential(
            nn.Linear(E_comm, 256),
            nn.ReLU(),
            nn.Linear(256, D_taxa),
            nn.Sigmoid()  # P(present) per taxon
        )
        
        # Abundance head (CLR logits for present taxa)
        self.abundance_head = nn.Sequential(
            nn.Linear(E_comm, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  # for MC Dropout uncertainty
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, D_taxa)  # CLR logits
        )
    
    def forward(self, community_embedding):
        presence_prob = self.presence_head(community_embedding)  # [batch, D]
        clr_logits = self.abundance_head(community_embedding)    # [batch, D]
        
        # Final composition: abundance gated by presence
        # In CLR space: set absent taxa to very negative logit
        gated_logits = clr_logits * presence_prob + (-30.0) * (1 - presence_prob)
        
        # Simplex composition via softmax
        composition = F.softmax(gated_logits, dim=-1)
        
        return {
            'clr_logits': clr_logits,
            'presence_prob': presence_prob,
            'gated_logits': gated_logits,
            'composition': composition
        }
```

### 6.4 Training Loss Computation

```python
def composition_loss(outputs, true_composition):
    """
    Combined compositional loss.
    """
    clr_logits = outputs['clr_logits']       # [batch, D]
    presence_prob = outputs['presence_prob']   # [batch, D]
    
    # Ground truth
    true_present = (true_composition > 0).float()  # [batch, D]
    
    # --- Loss 1: Presence/absence (BCE) ---
    L_presence = F.binary_cross_entropy(presence_prob, true_present)
    
    # --- Loss 2: Aitchison distance on present taxa ---
    # Multiplicative replacement for CLR of ground truth
    true_replaced = multiplicative_replacement(true_composition)
    clr_true = clr_transform(true_replaced)
    
    # Mask to present taxa only
    mask = true_present
    diff = (clr_logits - clr_true) * mask
    L_aitchison = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8).mean()
    
    # --- Combined ---
    return 0.7 * L_aitchison + 0.3 * L_presence
```

---

## 7. Training Strategy — Revised

### 7.1 Curriculum Learning (Unchanged but Specified)

```
Phase 1 (weeks 1-2):   Simple communities (10-30 taxa), mild artifacts
                        Focus: read encoder convergence
                        
Phase 2 (weeks 3-4):   Medium communities (30-100 taxa), moderate artifacts
                        Focus: composition head calibration
                        
Phase 3 (weeks 5-8):   Complex communities (100-300 taxa), realistic artifacts
                        Include zero-inflated dropout
                        Focus: zero-inflation head accuracy
                        
Phase 4 (weeks 9-12):  Full complexity (300-500 taxa), extreme edge cases
                        Multi-protocol batches for alignment loss
                        Focus: cross-protocol invariance
```

### 7.2 Phased Training with Head Warmup

```
Step 1: Train read encoder + error corrector + chimera detector ONLY
        (freeze composition head)
        → Validates that read-level processing works before composition
        
Step 2: Freeze read encoder, train composition head + zero-inflation
        → Composition accuracy on simple communities
        
Step 3: Unfreeze all, joint fine-tuning with reduced LR
        → End-to-end optimization
        
Step 4: Add alignment loss (contrastive) with multi-protocol batches
        → Cross-protocol invariance
```

### 7.3 Multi-Task Auxiliary Losses

```
Primary:   Composition prediction (Aitchison distance)
Secondary: Read error correction (cross-entropy)
Auxiliary:
  - Chimera detection (BCE)
  - Library size regression (MSE) — predict total reads from community embedding
  - Shannon diversity prediction (MSE) — predict alpha diversity
  - Protocol prediction from reads ONLY (CE) — sanity check that reads carry protocol info
  
NOTE: Protocol prediction is used ONLY as a diagnostic. 
      The latent embedding (Head 2C) should NOT be predictive of protocol.
      Read-level protocol prediction should be high; embedding-level should be low.
```

---

## 8. Inverse Simulator Ablation Plan

### 8.1 The Risk

If the model only learns to invert our specific simulator, it has learned an artifact-specific lookup table, not a general denoising function. The model would perfectly correct our simulated errors but fail on real data or differently-simulated data.

### 8.2 The Test

```
EXPERIMENT: Cross-Simulator Generalization

SETUP:
  1. Train model on reads from OUR simulator (Genesis Simulator)
  2. Generate test reads from:
     a. Genesis Simulator (in-distribution control)
     b. InSilicoSeq (different error model, no PCR bias)
     c. ART (Illumina-specific error profiles)
     d. CAMISIM (comprehensive, includes abundance simulation)
  3. All test sets use the SAME ground truth communities

METRICS:
  - Aitchison distance to ground truth
  - Bray-Curtis to ground truth (for literature comparability)
  - Presence/absence AUC
  - Per-taxon L1 error

PASS CRITERIA:
  - Performance on (b,c,d) within 20% of (a) → model is general
  - Performance on (b,c,d) >50% worse than (a) → model overfit to simulator
  
ADDITIONAL CONTROLS:
  - Run DADA2 on all four test sets → establishes baseline
  - If DADA2 also degrades across simulators, the problem is simulator 
    difference, not model overfitting
```

### 8.3 Simulator Calibration Test (Prerequisite to Ablation)

Before the ablation, we need to verify our simulator produces realistic reads:

```
EXPERIMENT: Simulator Realism

SETUP:
  1. Generate synthetic community matching MBARC-26 composition
  2. Simulate reads using Genesis Simulator
  3. Compare simulated reads to real MBARC-26 FASTQ:
  
TESTS:
  a. Read length distribution: KS test, p > 0.05
  b. Quality score distribution: KS test, p > 0.05
  c. Error rate by position: Pearson r > 0.9
  d. k-mer frequency spectrum: Jensen-Shannon divergence < 0.01
  e. DADA2 output comparison: Bray-Curtis(DADA2(sim), DADA2(real)) < 0.15
  
GO/NO-GO (from rubric):
  - Test (e) < 0.15 → GO to model training
  - Test (e) 0.15-0.25 → Investigate, recalibrate simulator
  - Test (e) > 0.25 → STOP
```

---

## 9. Validation Plan — Aligned to Go/No-Go Rubric

### Phase 0: Simulator Validation (Days 1-90)

| Checkpoint | Test | Data | Threshold | Decision |
|-----------|------|------|-----------|----------|
| C0.1 | Reads pass FASTQC | Simulated | No anomalies | Debug if fail |
| C0.2 | Error profiles match literature | Simulated vs empirical | r > 0.9 | Recalibrate |
| C0.3 | Chimera rate correct | Simulated | 1-3% | Adjust params |
| C0.4 | Abundance distributions realistic | Sim vs iHMP | KS p > 0.05 | Refit distribution |
| C0.5 | DADA2 consistency | DADA2(sim) vs DADA2(real) | BC < 0.15 | **STOP if > 0.25** |

### Phase 1: Proof of Concept (Months 3-4)

| Checkpoint | Test | Data | Threshold | Decision |
|-----------|------|------|-----------|----------|
| C1.1 | Composition accuracy (synthetic) | Synthetic test set | BC < 0.15 | Architecture problem |
| C1.2 | Composition accuracy (mock) | MBARC-26 raw reads | BC < 0.20 | **STOP — syn-to-real gap** |
| C1.3 | Beat DADA2 on mock | MBARC-26 | > 5% BC improvement | **STOP if < 2%** |
| C1.4 | Ablation: PCR bias module | Remove PCR bias sim | > 3% degradation | Module adds value |
| C1.5 | Ablation: protocol conditioning | Remove protocol token | > 3% degradation on multi-protocol | Conditioning adds value |
| C1.6 | Cross-simulator generalization | InSilicoSeq/ART/CAMISIM reads | Within 20% of in-dist | Overfit if > 50% |

### Phase 2: Multi-Lab Harmonization (Months 4-5)

| Checkpoint | Test | Data | Threshold | Decision |
|-----------|------|------|-----------|----------|
| C2.1 | Same-specimen cross-lab Spearman | MBQC-Baseline (22 specimens × 15 labs) | > 0.70 | Harmonization not working |
| C2.2 | Inter-lab variance reduction | MBQC-Baseline | > 30% variance reduction vs raw | No value over ComBat |
| C2.3 | Mock community accuracy | MBQC artificial communities (n=2) | BC < 0.10 to known composition | Ground truth recovery failed |

### Phase 3: CRC Classification (Months 5-6)

| Checkpoint | Test | Data | Threshold | Decision |
|-----------|------|------|-----------|----------|
| C3.1 | Within-cluster leave-one-out AUC | V4 cluster (Baxter, Hannigan, Zeller) | > 0.70 | No classification utility |
| C3.2 | WGS cluster leave-one-out AUC | WGS studies (6 datasets) | > 0.70 | No classification utility |
| C3.3 | AUC improvement over baseline | Corrected vs raw | > 0.03 AUC | **PUBLISH** if achieved |
| C3.4 | Cross-cluster attempt | Train V4, test WGS (exploratory) | Report result | Informs future work |

---

## 10. Uncertainty Quantification

### Implementation: MC Dropout

```python
class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    At inference, run T forward passes with dropout enabled.
    Variance across passes = epistemic uncertainty.
    """
    
    def __init__(self, model, T=30):
        self.model = model
        self.T = T
    
    def predict_with_uncertainty(self, batch):
        self.model.train()  # Keep dropout active
        
        predictions = []
        for _ in range(self.T):
            with torch.no_grad():
                out = self.model(batch)
                predictions.append(out['composition'])
        
        predictions = torch.stack(predictions)  # [T, batch, D]
        
        mean_composition = predictions.mean(dim=0)  # [batch, D]
        std_composition = predictions.std(dim=0)    # [batch, D]
        
        # 95% confidence interval (approximate)
        ci_lower = mean_composition - 1.96 * std_composition
        ci_upper = mean_composition + 1.96 * std_composition
        
        # Calibration: what fraction of true values fall in 95% CI?
        # (checked during validation)
        
        return {
            'mean': mean_composition,
            'std': std_composition,
            'ci_95_lower': ci_lower.clamp(min=0),
            'ci_95_upper': ci_upper.clamp(max=1),
            'all_samples': predictions
        }
```

### Calibration Test

The 95% CI must be calibrated: if the model says "95% CI is [0.02, 0.08]" for a taxon, then the true abundance should fall in that interval 95% of the time across many samples.

```
EXPERIMENT: Calibration Curve

For each nominal CI level (50%, 75%, 90%, 95%, 99%):
  1. Compute CI on synthetic test set
  2. Check fraction of true values inside CI
  3. Plot observed coverage vs nominal coverage
  
PASS: Points should lie on y=x diagonal (within ±5%)
FAIL: Systematic over/under-confidence requires recalibration
```

---

## 11. Hyperparameters and Compute Estimate

### Model Size (Prototype → Genesis-Ready)

| Component | Prototype | Genesis-Ready |
|-----------|-----------|---------------|
| k-mer vocabulary | 4,096 (k=6) | 65,536 (k=8) |
| Read encoder layers | 4 | 8 |
| Community encoder layers | 4 | 8 |
| Read embedding dim (E_read) | 128 | 256 |
| Community embedding dim (E_comm) | 256 | 512 |
| Latent embedding dim (L) | 64 | 128 |
| Protocol embedding dim (E_prot) | 32 | 64 |
| Max reads per sample | 1,000 | 10,000 |
| Max taxa (D) | 500 | 2,000 |
| Total parameters | ~15M | ~200M |
| Training time (est.) | ~100 GPU-hours | ~5,000 GPU-hours |

### Training Configuration

```yaml
optimizer: AdamW
learning_rate: 1e-4 (with cosine annealing)
weight_decay: 0.01
batch_size: 32 (prototype), 16 (Genesis-ready)
gradient_accumulation: 4 steps
mixed_precision: fp16
max_epochs: 100 (with early stopping, patience=10)
warmup_steps: 1000
```

---

## 12. What This Architecture Does NOT Do (Scope Boundaries)

1. **Does not replace DADA2 for routine use** — the read denoiser (Head 1A) is not the selling point. The composition corrector (Heads 2A+2B) is.

2. **Does not generate new reads** — Head 1A corrects existing reads, it does not hallucinate reads for dropped-out taxa. Dropout is handled at the composition level (Head 2B).

3. **Does not perform 16S ↔ WGS cross-protocol harmonization yet** — no paired dataset available for validation. This is deferred until Usyk et al. HCHS/SOL or equivalent is secured.

4. **Does not predict function** — composition only. Functional extension is a future direction (EVO2 integration, aspirational).

5. **Does not handle metatranscriptomics or metabolomics** — DNA sequencing only.

---

## 13. Key Differences from V1 Architecture

| Aspect | V1 (Original) | V2 (This Document) |
|--------|---------------|---------------------|
| Output | Composition vector only | Dual: corrected reads + composition + presence + embedding |
| Loss | KL + Bray-Curtis (wrong geometry) | Robust Aitchison + BCE + CE (compositionally correct) |
| Zero handling | Not addressed | Zero-inflated model separating structural vs sampling zeros |
| Protocol encoding | Simple one-hot → embedding | Rich encoding with continuous features (error rate, depth, GC) |
| Chimera handling | Not specified | Soft chimera gating in community pooling |
| Uncertainty | Mentioned, not designed | MC Dropout with calibration test |
| Training | End-to-end | Phased: read heads → composition head → joint → alignment |
| Validation | Vague ("95% agreement") | Quantitative rubric with named datasets, metrics, thresholds |
| Compositionality | Softmax output | CLR-space prediction with Aitchison distance loss |

---

## 14. Immediate Next Steps (Post-Architecture Approval)

1. **Implement Genesis Simulator v0.1** — community sampling + error injection + zero-inflated dropout
2. **Implement k-mer tokenizer** — k=6 prototype
3. **Implement read encoder** — 4-layer transformer, validate on synthetic error correction
4. **Checkpoint C0.1-C0.5** — simulator realism tests
5. **Download and stage validation data** — MBARC-26 FASTQ, MBQC-Baseline from SRA (BioProject SRP047083), NIST RM8376

---

*Architecture V2 — Microbiome Genesis Framework*  
*Developed during Critical Design Review session, 2026-02-02*  
*Roles: Technical Architect + ML Researcher panel*
