# Ko-CENTaUR Implementation Workflow

**Version**: 1.0
**Date**: 2025-01-07
**Based On**: LLM_SELECTION_STRATEGY.md v1.0
**Timeline**: 12 months (3 phases)
**Total Budget**: ~$3,336 ($936 training + $2,400 inference)

---

## Executive Summary

This workflow document provides week-by-week implementation guidance for the Ko-CENTaUR project, breaking down the 3-phase staged approach (MVP → Scale-up → Full System) into actionable tasks with clear dependencies, resource requirements, and success criteria.

**Critical Path**: IRB approval and data collection are the primary bottlenecks. All technical work should proceed in parallel where possible.

---

## Table of Contents

1. [Timeline Overview](#timeline-overview)
2. [Phase 1: MVP Validation (Weeks 1-12)](#phase-1-mvp-validation-weeks-1-12)
3. [Phase 2: Scale-up (Weeks 13-26)](#phase-2-scale-up-weeks-13-26)
4. [Phase 3: Full System (Weeks 27-52)](#phase-3-full-system-weeks-27-52)
5. [Resource Allocation](#resource-allocation)
6. [Risk Management](#risk-management)
7. [Success Metrics](#success-metrics)
8. [Parallel Execution Map](#parallel-execution-map)

---

## Timeline Overview

```
Weeks 1-12:   Phase 1 - MVP Validation (EEVE-10.8B)
Weeks 13-26:  Phase 2 - Scale-up (EXAONE-32B + Distillation)
Weeks 27-52:  Phase 3 - Full System (Hybrid Architecture)

Key Milestones:
├─ Week 4:   Go/No-Go #1 - IRB submission complete
├─ Week 12:  Go/No-Go #2 - MVP evaluation → Phase 2 decision
├─ Week 26:  Go/No-Go #3 - Korean norms validated → Phase 3 decision
├─ Week 40:  Go/No-Go #4 - Multimodal integration verified
└─ Week 52:  Public release and publication submission
```

---

## Phase 1: MVP Validation (Weeks 1-12)

**Goal**: Validate pipeline with minimal investment ($200)
**Success Criteria**: NLL improvement ≥20%, Korean tokenization efficient, error-free pipeline

### Week 1-2: Infrastructure & Planning

#### Task 1.1: Environment Setup [2 days]
**Priority**: Critical
**Dependencies**: None
**Owner**: Technical lead

```bash
# Deliverables:
# - conda environment with all dependencies
# - tokenization benchmark results
# - repository structure

# Commands:
conda create -n ko-centaur python=3.10
conda activate ko-centaur
pip install torch==2.1.0 transformers==4.36.0 datasets==2.15.0 \
    peft==0.7.0 unsloth==2023.12 bitsandbytes==0.41.3 \
    scipy scikit-learn pandas numpy

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Success Criteria**:
- ✅ All packages install without errors
- ✅ CUDA available and working
- ✅ Can load a small model (e.g., EXAONE-3.0-7.8B test)

**Blockers**: GPU access issues → Use Colab/Kaggle as fallback

---

#### Task 1.2: Repository Restructuring [1 day]
**Priority**: High
**Dependencies**: Task 1.1
**Owner**: Technical lead

```bash
# Create new structure
cd /Users/jiookcha/Documents/git/CENTaUR
mkdir -p ko_centaur/{data/{transcription,metadata,norms,raw},training,evaluation,models,tasks}
mkdir -p legacy scripts docs/experiments

# Archive legacy code
mv llama choices13k HorizonTask ExperientialSymbolicTask legacy/

# Create initial files
touch ko_centaur/__init__.py
touch ko_centaur/training/__init__.py
touch ko_centaur/evaluation/__init__.py
```

**Deliverables**:
- New directory structure
- Legacy code archived but accessible
- README.md updated

---

#### Task 1.3: Tokenization Benchmark [3 days]
**Priority**: Critical
**Dependencies**: Task 1.1
**Owner**: Technical lead
**Code**: See Appendix A in LLM_SELECTION_STRATEGY.md

```bash
# Run benchmark
python scripts/test_tokenization.py

# Expected output:
# Llama-3.1:  236 tokens (baseline 1.0×)
# EXAONE:     100 tokens (2.36× better)
# EEVE:       115 tokens (2.05× better)
# Qwen:       150 tokens (1.57× better)
```

**Success Criteria**:
- ✅ EXAONE/EEVE show 2.0-2.4× efficiency vs Llama
- ✅ Benchmark results match strategy document predictions
- ✅ Results documented in `docs/experiments/tokenization_benchmark.md`

**Parallel Opportunity**: Run this while waiting for IRB feedback

---

#### Task 1.4: IRB Application Preparation [PARALLEL - Week 1-4]
**Priority**: Critical (blocks data collection)
**Dependencies**: None (can start immediately)
**Owner**: PI/Research coordinator

**Deliverables**:
- IRB protocol document
- Consent forms (Korean + English)
- Data management plan
- Security protocols
- Recruitment materials

**Components**:
1. **Protocol Narrative** (5-10 pages):
   - Background and significance
   - Participant recruitment (N=20 pilot, N=5,500 full)
   - Age range: 0-18+ (developmental span)
   - Tasks: K-MMSE, PHQ-9, SDQ, ABCD tasks
   - Data storage and security
   - Risk/benefit analysis

2. **Consent Forms**:
   - Adult consent (18+)
   - Parental consent (minors)
   - Child assent (ages 7-17)
   - Korean translations required

3. **Data Management**:
   - De-identification procedures
   - Encryption standards (AES-256)
   - Access controls
   - Storage location and backup
   - Retention period (10 years)

**Timeline**:
- Week 1: Draft protocol
- Week 2: Internal review
- Week 3: Submit to IRB
- Week 4-8: IRB review (expect revisions)

**Decision Gate**: If IRB not submitted by Week 4 → Red flag, escalate

---

### Week 3-4: Pilot Data Collection

#### Task 1.5: Pilot Participant Recruitment [2 weeks]
**Priority**: Critical
**Dependencies**: Task 1.4 (IRB submission, not approval)
**Owner**: Research coordinator

**Strategy**: While waiting for IRB approval, prepare recruitment materials

**Target**: N=20 pilot participants
- Age distribution: 5 each from child (6-11), adolescent (12-17), young adult (18-25), adult (26+)
- Gender balance: 10M/10F
- Clinical mix: 15 healthy control, 5 mild clinical symptoms

**Recruitment Channels**:
1. Hospital outpatient clinics
2. University student populations
3. Community centers
4. Online recruitment platforms (Korean)

**Inclusion Criteria**:
- Korean native speaker
- Age 6+
- No severe cognitive impairment
- Informed consent/assent obtained

**Compensation**: ₩30,000 per session (~$22)

**Timeline**:
- Week 3-4: Recruitment materials preparation
- Week 5-8: Active recruitment (parallel with IRB approval)

---

#### Task 1.6: Data Collection Protocol [1 week]
**Priority**: High
**Dependencies**: Task 1.5
**Owner**: Clinical staff + research coordinator

**Instruments**:
1. **K-MMSE** (Korean Mini-Mental State Examination)
   - 30 questions
   - ~10 minutes
   - Gold standard cognitive screening

2. **PHQ-9** (Patient Health Questionnaire)
   - 9 questions
   - ~5 minutes
   - Depression screening

3. **SDQ** (Strengths and Difficulties Questionnaire)
   - 25 questions
   - ~5 minutes
   - Behavioral screening (children/adolescents)

**Data Format**: JSONL with `<< >>` response masking

**Example JSONL Entry**:
```json
{
  "text": "[meta: age_m=180; sex=M; cbcl_int_t=72]\n\n지남력 평가:\n오늘은 몇 년도인가요? <<2025>>\n오늘은 몇 월인가요? <<1>>",
  "experiment": "k_mmse",
  "participant": {
    "id": "KO_MMSE_001",
    "age_months": 180,
    "age_group": "adolescent",
    "gender": "M"
  },
  "questionnaire_metadata": {
    "instrument": "K-MMSE",
    "total_score": 27,
    "normative_percentile": 45
  }
}
```

**Quality Control**:
- Double-entry verification
- Immediate QC check after each session
- Flag incomplete/ambiguous responses

**Deliverable**: 20 complete JSONL files in `ko_centaur/data/raw/pilot/`

---

### Week 5-8: Training Pipeline Development

#### Task 1.7: Training Script Implementation [1 week]
**Priority**: Critical
**Dependencies**: Task 1.2, Task 1.3
**Owner**: ML engineer
**Code**: Adapt from LLM_SELECTION_STRATEGY.md lines 550-735

**File**: `ko_centaur/training/train_qlora.py`

**Key Features**:
1. Model loading with 4-bit quantization
2. LoRA configuration (r=16, α=32)
3. Response masking (only `<< >>` contributes to loss)
4. Training with gradient accumulation
5. Checkpoint saving

**Implementation Steps**:
```bash
# Step 1: Create training module structure
touch ko_centaur/training/train_qlora.py
touch ko_centaur/training/utils.py
touch ko_centaur/training/config.py

# Step 2: Implement core functions
# - load_model_and_tokenizer()
# - create_response_masking_collator()
# - train_ko_centaur()

# Step 3: Unit tests
pytest tests/test_training.py
```

**Success Criteria**:
- ✅ Can load EEVE-10.8B with 4-bit quantization
- ✅ Response masking correctly identifies `<< >>` regions
- ✅ Training runs for 10 steps without errors
- ✅ Loss decreases (sanity check)

**Testing**: Use synthetic data (5 samples) before real data

---

#### Task 1.8: Evaluation Framework [1 week, PARALLEL]
**Priority**: High
**Dependencies**: Task 1.2
**Owner**: ML engineer
**Code**: Adapt from LLM_SELECTION_STRATEGY.md lines 967-1178

**File**: `ko_centaur/evaluation/metrics.py`

**Metrics to Implement**:
1. **NLL (Negative Log-Likelihood)**:
   - Primary metric
   - Only computed on `<< >>` tokens
   - Lower is better

2. **Accuracy**:
   - Token-level exact match
   - Secondary metric

3. **Korean Norm Correlation**:
   - Pearson r between predicted and actual scores
   - Target: r ≥ 0.70

4. **Age Effect Validation**:
   - Correlation between age and performance
   - Expected: positive correlation

5. **Clinical Discrimination**:
   - AUC for clinical vs control
   - Target: AUC ≥ 0.75

**Implementation**:
```python
# Core class
class CENTaURMetrics:
    def compute_nll_accuracy(test_data) -> dict
    def evaluate_korean_norms(test_data, korean_norms) -> dict
    def evaluate_age_effects(test_data) -> dict
    def evaluate_clinical_discrimination(test_data) -> dict
    def full_evaluation_report(test_data, norms) -> dict
```

**Testing**: Use synthetic test data with known properties

**Parallel Opportunity**: Can develop this while training runs

---

#### Task 1.9: First Training Run [4 days]
**Priority**: Critical
**Dependencies**: Task 1.6 (pilot data), Task 1.7 (training script)
**Owner**: ML engineer

**Configuration**:
```python
model_config = {
    "model_name": "yanolja/EEVE-Korean-10.8B-v1.0",
    "data_path": "ko_centaur/data/raw/pilot/pilot_20.jsonl",
    "output_dir": "ko_centaur/models/eeve-mvp-v1",
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_length": 4096
}
```

**Resource Requirements**:
- GPU: 1× A100 40GB (or 2× 3090)
- Time: 48 hours
- Cost: $197

**Monitoring**:
- Loss curves (should decrease)
- GPU utilization (target >90%)
- Memory usage (should not OOM)
- Save checkpoints every epoch

**Expected Output**:
- Trained LoRA weights
- Training logs
- Final evaluation metrics

**Blockers**: OOM errors → Reduce batch size, increase gradient accumulation

---

### Week 9-12: Evaluation & Decision

#### Task 1.10: Comprehensive Evaluation [1 week]
**Priority**: Critical
**Dependencies**: Task 1.9 (trained model), Task 1.8 (evaluation framework)
**Owner**: ML engineer + Research lead

**Evaluation Protocol**:
1. **Test Data Split**: Hold out 4 participants (20%) from training
2. **Baseline Comparison**: Random baseline (NLL = -log(1/vocab_size))
3. **Compute All Metrics**:
   ```python
   evaluator = CENTaURMetrics(model, tokenizer)
   report = evaluator.full_evaluation_report(
       test_data=held_out_4,
       korean_norms=k_mmse_norms
   )
   ```

**Expected Results (Success)**:
- NLL improvement: ≥20% vs random baseline
- Accuracy: ≥60% (token-level)
- Korean tokenization: No truncation issues
- Pipeline: Error-free execution

**Failure Modes**:
- NLL improvement <10% → Model not learning, check data format
- Truncation errors → Sessions too long, need splitting
- Training instability → Hyperparameter tuning required

---

#### Task 1.11: Go/No-Go Decision Gate #2 [1 day]
**Priority**: Critical
**Dependencies**: Task 1.10
**Owner**: PI + Technical lead

**Decision Criteria**:

| Metric | Threshold | Result |
|--------|-----------|--------|
| NLL improvement | ≥20% | PASS/FAIL |
| Korean tokenization | No truncation | PASS/FAIL |
| Pipeline execution | Error-free | PASS/FAIL |
| Data quality | <5% issues | PASS/FAIL |

**Outcomes**:

**GO (Proceed to Phase 2)**:
- ✅ All criteria passed
- Action: Begin EXAONE-32B setup
- Budget: Approve $444 for Phase 2

**NO-GO (Iterate on MVP)**:
- ❌ 1+ criterion failed
- Actions:
  1. Identify failure root cause
  2. Implement fixes (2-4 weeks)
  3. Re-evaluate
  4. If still failing → Reconsider approach or pivot

**Contingency**: If Korean tokenization fails, consider switching to character-level tokenization or different model

---

#### Task 1.12: Phase 1 Documentation [1 week, PARALLEL]
**Priority**: Medium
**Dependencies**: Task 1.10
**Owner**: Technical lead

**Deliverables**:
1. **Technical Report**: `docs/experiments/phase1_mvp_report.md`
   - Methodology
   - Results
   - Lessons learned
   - Recommendations for Phase 2

2. **Code Documentation**:
   - Docstrings for all functions
   - README for training pipeline
   - Example notebooks

3. **Data Documentation**:
   - Participant demographics
   - Data quality report
   - Known issues and limitations

**Parallel Opportunity**: Write documentation while waiting for Go/No-Go decision

---

## Phase 2: Scale-up (Weeks 13-26)

**Goal**: Production-ready Korean cognitive model ($444)
**Success Criteria**: Korean norm correlation r ≥0.70, age effects replicate, clinical AUC ≥0.75

### Week 13-14: Data Collection Scale-Up

#### Task 2.1: Full Dataset Recruitment [12 weeks, PARALLEL]
**Priority**: Critical (long lead time)
**Dependencies**: Task 1.4 (IRB approval)
**Owner**: Research coordinator + clinical partners

**Target**: N=5,500 total sessions
- N=1,500 children (6-11 years)
- N=1,500 adolescents (12-17 years)
- N=1,500 young adults (18-25 years)
- N=1,000 adults (26+ years)

**Distribution**:
- 70% healthy controls (N=3,850)
- 30% clinical samples (N=1,650)
  - ADHD: N=400
  - Anxiety/Depression: N=400
  - Learning disabilities: N=300
  - Autism spectrum: N=200
  - Other: N=350

**Instruments per Session**:
1. K-MMSE or age-appropriate variant
2. PHQ-9 (depression)
3. SDQ (behavior - children/adolescents)
4. GAD-7 (anxiety)
5. Optional: ABCD tasks (publicly available subset)

**Timeline**:
- Weeks 13-16: Ramp up recruitment (100 sessions/week)
- Weeks 17-20: Peak collection (200 sessions/week)
- Weeks 21-24: Maintain (150 sessions/week)
- Week 25: Final push and QC

**Budget**: ₩30,000 × 5,500 = ₩165M (~$120,000 USD)

**Parallel Strategy**: Data collection continues throughout Phase 2 and 3

---

#### Task 2.2: EXAONE-32B Environment Setup [1 week]
**Priority**: High
**Dependencies**: Task 1.11 (Go decision)
**Owner**: ML engineer

```bash
# Test EXAONE loading
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'LGAI-EXAONE/EXAONE-3.0-32B-Instruct',
    load_in_4bit=True,
    device_map='auto'
)
print(f'Model loaded: {model.num_parameters()/1e9:.1f}B params')
"

# Verify memory requirements
# Expected: ~18-22 GB with 4-bit quantization
```

**Success Criteria**:
- ✅ EXAONE-32B loads successfully
- ✅ Memory footprint acceptable (<24 GB)
- ✅ Can run inference on test prompt

**Blockers**: OOM → Need A100 80GB or multi-GPU setup

---

### Week 15-16: Distillation Pipeline

#### Task 2.3: Teacher Model (Qwen 72B) Setup [1 week]
**Priority**: High
**Dependencies**: Task 2.2
**Owner**: ML engineer
**Code**: Adapt from LLM_SELECTION_STRATEGY.md lines 737-963

**File**: `ko_centaur/distillation/teacher.py`

**Implementation**:
```python
class TeacherModel:
    def __init__(self, model_name="Qwen/Qwen2.5-72B-Instruct"):
        # Load teacher with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(...)

    def generate_reasoning_traces(self, dataset, save_path):
        # For each session:
        # 1. Forward pass through teacher
        # 2. Extract logits and hidden states
        # 3. Save as training targets
        pass
```

**Resource Requirements**:
- GPU: 1× A100 80GB or 2× A100 40GB
- Time: ~24 hours for 5,500 sessions
- Cost: ~$50 (API inference) or $100 (self-hosted)

**Strategy Decision**: Use API inference for cost efficiency
- Qwen API: $0.01 per 1K tokens
- 5,500 sessions × 2K tokens = 11M tokens
- Cost: $110 (slightly higher than estimate, but faster)

**Alternative**: Self-host if API unavailable

---

#### Task 2.4: Generate Teacher Outputs [1 week]
**Priority**: High
**Dependencies**: Task 2.3, Task 2.1 (partial data available)
**Owner**: ML engineer

**Process**:
```python
# Load available data (expect ~500-1000 sessions by Week 16)
from datasets import load_dataset
dataset = load_dataset(
    "json",
    data_files="ko_centaur/data/transcription/*.jsonl"
)

# Generate teacher outputs
teacher = TeacherModel()
teacher_data = teacher.generate_reasoning_traces(
    dataset['train'],
    save_to="ko_centaur/data/distillation/qwen_teacher_batch1.pt"
)
```

**Batch Strategy**: Process data in batches as it becomes available
- Batch 1 (Week 16): ~500 sessions
- Batch 2 (Week 20): ~2,000 sessions
- Batch 3 (Week 24): ~3,000 sessions
- Batch 4 (Week 25): Final ~5,500 sessions

**Success Criteria**:
- ✅ All sessions processed without errors
- ✅ Teacher outputs saved with logits, hidden states, metadata
- ✅ File size reasonable (~10-20 GB for 5,500 sessions)

---

### Week 17-20: EXAONE Training with Distillation

#### Task 2.5: Distillation Training Script [1 week]
**Priority**: Critical
**Dependencies**: Task 2.4
**Owner**: ML engineer
**Code**: Adapt from LLM_SELECTION_STRATEGY.md lines 737-963

**File**: `ko_centaur/distillation/train_student.py`

**Key Components**:
1. **Combined Loss Function**:
   ```python
   loss = (
       0.5 * human_nll_loss +      # Match human choices
       0.3 * teacher_kl_loss +      # Learn from teacher logits
       0.2 * hidden_mse_loss        # Align representations
   )
   ```

2. **Custom Trainer**:
   ```python
   class DistillationTrainer(Trainer):
       def compute_loss(self, model, inputs):
           # Get student outputs
           student_out = model(**inputs, output_hidden_states=True)

           # Compute distillation loss
           loss = self.distiller.compute_distillation_loss(
               student_out,
               inputs['labels'],  # Human choices
               inputs['teacher_logits'],
               inputs['teacher_hidden']
           )
           return loss
   ```

**Success Criteria**:
- ✅ Training script runs without errors
- ✅ Combined loss decreases across all components
- ✅ Checkpoints saved properly

---

#### Task 2.6: EXAONE-32B Training Run [4 days]
**Priority**: Critical
**Dependencies**: Task 2.5, Task 2.4 (Batch 1 teacher data)
**Owner**: ML engineer

**Configuration**:
```python
student_config = {
    "student_model": "LGAI-EXAONE/EXAONE-3.0-32B-Instruct",
    "teacher_data": "ko_centaur/data/distillation/qwen_teacher_batch1.pt",
    "human_data": "ko_centaur/data/transcription/batch1_500.jsonl",
    "output_dir": "ko_centaur/models/exaone-distilled-v1",
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_length": 32768
}
```

**Resource Requirements**:
- GPU: 1× A100 40GB
- Time: 96 hours (4 days)
- Cost: $394

**Monitoring**:
- Track all 3 loss components separately
- Validate that human_nll_loss converges fastest
- Ensure no catastrophic forgetting (Korean performance maintained)

**Expected Output**:
- Distilled EXAONE model with combined knowledge
- Training curves showing convergence
- Checkpoint at each epoch

---

### Week 21-22: Validation & Iteration

#### Task 2.7: Intermediate Evaluation [1 week]
**Priority**: High
**Dependencies**: Task 2.6
**Owner**: ML engineer + Research lead

**Evaluation on Batch 1 Test Set** (N=100 held-out):
```python
evaluator = CENTaURMetrics(exaone_model, tokenizer)
report = evaluator.full_evaluation_report(
    test_data=batch1_test,
    korean_norms=k_mmse_norms
)
```

**Key Metrics**:
1. **Korean Norm Correlation**: Target r ≥ 0.70
2. **Age Effects**: Positive correlation expected
3. **NLL Improvement**: Should exceed Phase 1 MVP
4. **Clinical Discrimination**: Target AUC ≥ 0.75 (if data available)

**Decision Point**:
- If metrics meet targets → Proceed with full training
- If metrics below targets → Investigate and iterate:
  - Adjust loss weights
  - Increase training epochs
  - Tune learning rate
  - Add more data

---

#### Task 2.8: Hyperparameter Optimization [1 week, OPTIONAL]
**Priority**: Medium
**Dependencies**: Task 2.7 (if metrics below target)
**Owner**: ML engineer

**Parameters to Tune**:
1. Loss weights: {human_nll, teacher_kl, hidden_mse}
2. Learning rate: [1e-5, 2e-5, 5e-5]
3. LoRA rank: [8, 16, 32]
4. Temperature: [1.0, 2.0, 3.0]

**Strategy**: Grid search with 2-3 values per parameter
- Run 5-10 ablation experiments
- Each experiment: 1 epoch on Batch 1 data
- Select best configuration for full training

**Cost**: ~$50-100 (short experiments)

**Skip if**: Intermediate evaluation already meets targets

---

### Week 23-26: Full Training & Adapter System

#### Task 2.9: Full EXAONE Training [1 week]
**Priority**: Critical
**Dependencies**: Task 2.4 (all teacher data), Task 2.8 (hyperparameters)
**Owner**: ML engineer

**Final Configuration** (with optimized hyperparameters):
```python
final_config = {
    "student_model": "LGAI-EXAONE/EXAONE-3.0-32B-Instruct",
    "teacher_data": "ko_centaur/data/distillation/qwen_teacher_full.pt",
    "human_data": "ko_centaur/data/transcription/full_5500.jsonl",
    "output_dir": "ko_centaur/models/exaone-distilled-production",
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,  # Or optimized value
    "loss_weights": {
        "human_nll": 0.5,
        "teacher_kl": 0.3,
        "hidden_mse": 0.2
    }
}
```

**Resource Requirements**:
- GPU: 1× A100 40GB
- Time: 96 hours
- Cost: $394 (already budgeted)

**Checkpointing Strategy**:
- Save every epoch
- Keep best checkpoint based on validation loss
- Upload to HuggingFace Hub (private repo)

---

#### Task 2.10: Dual Adapter System [1 week]
**Priority**: High
**Dependencies**: Task 2.9
**Owner**: ML engineer

**Architecture**:
```
Base Model: EXAONE-32B (distilled, frozen)
    ↓
├─ Public Adapter (shareable)
│  ├─ Training data: K-MMSE, PHQ-9, SDQ, GAD-7
│  ├─ Output: Public cognitive scores
│  └─ Release: HuggingFace Hub
│
└─ Private Adapter (restricted)
   ├─ Training data: ABCD metadata, WISC, sensitive clinical
   ├─ Output: Clinical diagnoses, ABCD predictions
   └─ Release: NEVER (research use only)
```

**Implementation**:
```python
# ko_centaur/adapters/dual_adapter.py

class PublicAdapter:
    def __init__(self, base_model, data):
        # Train LoRA adapter on public tasks
        self.adapter = self.train_adapter(
            base_model,
            data=open_tasks,
            lora_r=8,  # Smaller rank for adapter
            epochs=1
        )

class PrivateAdapter:
    def __init__(self, base_model, data):
        # Train LoRA adapter on private tasks
        self.adapter = self.train_adapter(
            base_model,
            data=restricted_tasks,
            lora_r=8,
            epochs=1
        )
        # Add access controls
        self.restricted = True
```

**Training Cost**: ~$100 (2 adapters × 1 day each)

**Success Criteria**:
- ✅ Public adapter performs well on K-MMSE, PHQ-9
- ✅ Private adapter improves ABCD predictions
- ✅ Adapters can be swapped without reloading base model

---

#### Task 2.11: Phase 2 Comprehensive Evaluation [1 week]
**Priority**: Critical
**Dependencies**: Task 2.9, Task 2.10
**Owner**: Research lead + ML engineer

**Full Test Suite**:
1. **Korean Normative Validation** (N=1,000 held-out):
   ```python
   norm_eval = evaluator.evaluate_korean_norms(
       test_data=korean_test_set,
       korean_norms=age_stratified_norms
   )
   # Target: r ≥ 0.70
   ```

2. **Age Effects Replication**:
   ```python
   age_eval = evaluator.evaluate_age_effects(test_data)
   # Target: Positive correlation, matches literature
   ```

3. **Clinical Discrimination**:
   ```python
   clinical_eval = evaluator.evaluate_clinical_discrimination(
       clinical_sample=adhd_anxiety_samples,
       control_sample=healthy_controls
   )
   # Target: AUC ≥ 0.75
   ```

4. **Cross-Task Generalization**:
   - Train on K-MMSE, test on PHQ-9
   - Train on PHQ-9, test on GAD-7
   - Target: r ≥ 0.60 cross-task

**Deliverable**: Comprehensive evaluation report in `docs/experiments/phase2_evaluation.md`

---

#### Task 2.12: Go/No-Go Decision Gate #3 [1 day]
**Priority**: Critical
**Dependencies**: Task 2.11
**Owner**: PI + Technical lead

**Decision Criteria**:

| Metric | Threshold | Weight | Result |
|--------|-----------|--------|--------|
| Korean norm r | ≥0.70 | Critical | PASS/FAIL |
| Age effects | Positive & significant | High | PASS/FAIL |
| Clinical AUC | ≥0.75 | High | PASS/FAIL |
| NLL improvement | ≥40% vs baseline | Medium | PASS/FAIL |

**Outcomes**:

**GO (Proceed to Phase 3)**:
- ✅ ≥3/4 criteria passed (must include Korean norm r)
- Action: Begin multimodal integration
- Budget: Approve $300 for Phase 3

**CONDITIONAL GO**:
- ⚠️ 2/4 criteria passed
- Action: 2-week iteration to address gaps
- Re-evaluate before Phase 3

**NO-GO**:
- ❌ <2 criteria passed
- Action: Deep investigation, consider pivot
- Options:
  1. Increase training data (collect more samples)
  2. Try different distillation strategy
  3. Use Qwen 72B directly (higher cost)

---

## Phase 3: Full System (Weeks 27-52)

**Goal**: Multimodal cognitive architecture with vision capability
**Success Criteria**: Full system validated, publication ready, public release

### Week 27-32: Multimodal Integration

#### Task 3.1: Qwen2-VL-7B Integration [2 weeks]
**Priority**: High
**Dependencies**: Task 2.12 (Go decision)
**Owner**: ML engineer

**Purpose**: Add vision capability for infant/visual tasks (~10% of sessions)

**Architecture**:
```python
# ko_centaur/multimodal/hybrid_model.py

class HybridCognitiveModel:
    def __init__(self):
        # Core text model (always active)
        self.text_model = load_exaone_distilled()

        # Vision module (on-demand)
        self.vision_model = load_qwen2_vl()

        # Router
        self.router = TaskRouter()

    def process(self, task):
        # Determine modality
        if self.router.has_visual_stimuli(task):
            # Vision → Text pipeline
            visual_features = self.vision_model.encode_image(task.image)
            text_context = self.vision_model.describe_visual(visual_features)
            return self.text_model.process(text_context + task.text)
        else:
            # Text-only pipeline
            return self.text_model.process(task.text)
```

**Training**:
- Collect visual tasks: N=550 (10% of 5,500)
- Examples:
  - Infant visual preference (2 images, which attended longer?)
  - Picture naming tasks
  - Visual memory tasks
  - Figure drawing interpretation

**Training Cost**: $295 (72 hours × $4.10/hr)

---

#### Task 3.2: Visual Task Data Collection [PARALLEL, Weeks 27-35]
**Priority**: Medium
**Dependencies**: Task 2.12
**Owner**: Research coordinator

**Target**: N=550 visual tasks
- N=200 infant visual preference (ages 0-2)
- N=150 picture naming (ages 3-6)
- N=100 visual memory (ages 7-12)
- N=100 figure drawing (ages 6-18)

**Data Format**:
```json
{
  "text": "[meta: age_m=18; sex=F]\n\n시각 선호도 검사:\n두 이미지가 제시됩니다. 아기가 더 오래 본 이미지는 어느 것인가요? <<왼쪽>>",
  "images": [
    "data/images/task_001_left.jpg",
    "data/images/task_001_right.jpg"
  ],
  "experiment": "infant_visual_preference",
  "participant": {
    "id": "KO_INF_001",
    "age_months": 18,
    "age_group": "infant"
  }
}
```

**Challenges**:
- Infant recruitment difficult → Partner with pediatric clinics
- Visual stimuli require standardization → Use validated test batteries
- Annotation requires expert coding → Train research assistants

**Parallel Strategy**: Collect visual data while text training continues

---

#### Task 3.3: Vision Model Training [1 week]
**Priority**: Medium
**Dependencies**: Task 3.2 (partial visual data)
**Owner**: ML engineer

**Configuration**:
```python
vision_config = {
    "model_name": "Qwen/Qwen2-VL-7B-Instruct",
    "data_path": "ko_centaur/data/multimodal/visual_tasks_batch1.jsonl",
    "output_dir": "ko_centaur/models/qwen2-vl-ko",
    "epochs": 2,
    "batch_size": 4,
    "learning_rate": 1e-5,
    "lora_r": 16,
    "max_seq_length": 4096
}
```

**Resource Requirements**:
- GPU: 1× A100 40GB
- Time: 72 hours
- Cost: $295

**Success Criteria**:
- ✅ Model processes image+text inputs correctly
- ✅ Visual feature extraction works
- ✅ Performance on visual tasks ≥ baseline

---

#### Task 3.4: Hybrid System Integration [1 week]
**Priority**: High
**Dependencies**: Task 3.3, Task 2.9
**Owner**: ML engineer

**Integration Tasks**:
1. **Router Implementation**:
   ```python
   class TaskRouter:
       def has_visual_stimuli(self, task):
           # Check if task contains image paths
           return 'images' in task

       def route(self, task):
           if self.has_visual_stimuli(task):
               return 'vision_branch'
           else:
               return 'text_branch'
   ```

2. **Model Fusion**:
   - Vision encoder extracts features
   - EXAONE receives visual features as additional context
   - Combined processing in single forward pass

3. **Inference Pipeline**:
   ```python
   def hybrid_inference(task):
       if task.has_images():
           # Vision branch
           visual_ctx = vision_model.encode(task.images)
           text_input = merge_context(visual_ctx, task.text)
       else:
           # Text branch
           text_input = task.text

       # Core cognitive processing
       output = exaone_model.process(text_input)
       return output
   ```

**Testing**:
- Unit tests for router
- Integration tests with sample visual tasks
- End-to-end validation on mixed batch (90% text, 10% visual)

---

#### Task 3.5: DeepSeek-R1 Integration [OPTIONAL, 1 week]
**Priority**: Low
**Dependencies**: Task 3.4
**Owner**: ML engineer

**Purpose**: Add explicit reasoning traces for interpretability

**Use Cases**:
- Complex multi-step reasoning tasks
- Clinical decision explanation
- Research into reasoning development

**Architecture**:
```python
class ReasoningModule:
    def __init__(self):
        self.reasoning_model = load_deepseek_r1()

    def generate_reasoning_trace(self, task):
        # Generate step-by-step reasoning
        trace = self.reasoning_model.reason(task)
        return trace

    def explain_prediction(self, task, prediction):
        # Post-hoc explanation
        explanation = self.reasoning_model.explain(task, prediction)
        return explanation
```

**Decision**: Defer to future work if time/budget constrained

**Cost**: ~$200 (optional, not in core budget)

---

### Week 33-40: Full System Training & Evaluation

#### Task 3.6: Complete Data Collection [Weeks 33-36]
**Priority**: High
**Dependencies**: Task 2.1 (ongoing)
**Owner**: Research coordinator

**Final Push**:
- Ensure all 5,500 text sessions complete
- Ensure 550 visual sessions complete
- Final QC pass on all data
- Resolve any data quality issues

**Data Preparation**:
```bash
# Organize final dataset
ko_centaur/data/final/
├── train/
│   ├── text_tasks_4400.jsonl (80% of 5,500)
│   └── visual_tasks_440.jsonl (80% of 550)
├── validation/
│   ├── text_tasks_550.jsonl (10%)
│   └── visual_tasks_55.jsonl (10%)
└── test/
    ├── text_tasks_550.jsonl (10%)
    └── visual_tasks_55.jsonl (10%)
```

**Success Criteria**:
- ✅ All sessions collected
- ✅ Data quality >95% (manual review)
- ✅ Train/val/test split stratified by age, gender, clinical status

---

#### Task 3.7: Full System Training [2 weeks]
**Priority**: Critical
**Dependencies**: Task 3.6, Task 3.4
**Owner**: ML engineer

**Training Sequence**:
1. **Text Model** (EXAONE): Already trained (Task 2.9)
2. **Vision Model** (Qwen2-VL): Train on full visual dataset
3. **Hybrid Integration**: Fine-tune fusion layer

**Vision Model Final Training**:
```python
vision_final_config = {
    "model_name": "Qwen/Qwen2-VL-7B-Instruct",
    "data_path": "ko_centaur/data/final/train/visual_tasks_440.jsonl",
    "output_dir": "ko_centaur/models/qwen2-vl-ko-final",
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-5,
}
```

**Resource Requirements**:
- GPU: 1× A100 40GB
- Time: 72 hours
- Cost: $295 (already budgeted)

**Fusion Layer Training** (optional):
- Train lightweight fusion layer to combine visual + text
- 24 hours, $100
- Skip if direct concatenation works well

---

#### Task 3.8: Comprehensive System Evaluation [2 weeks]
**Priority**: Critical
**Dependencies**: Task 3.7
**Owner**: Research lead + ML engineer + Clinical partners

**Full Evaluation Suite**:

1. **Text-Only Performance** (N=550 held-out):
   - Korean norm correlation
   - Age effects
   - Clinical discrimination
   - Cross-task generalization

2. **Vision-Only Performance** (N=55 held-out):
   - Visual task accuracy
   - Age-appropriate vision norms
   - Infant preference prediction

3. **Hybrid Performance**:
   - Text+vision integration
   - Multimodal reasoning tasks

4. **Clinical Validation** (N=200 clinical cases):
   - ADHD detection
   - Anxiety/depression screening
   - ASD assessment
   - Comparison with clinician judgments

5. **Developmental Trajectory**:
   - Performance across age groups (0-18+)
   - Replication of known developmental patterns
   - Novel developmental insights

**Deliverable**: Complete evaluation report (30-50 pages) in `docs/experiments/phase3_full_evaluation.md`

---

#### Task 3.9: Go/No-Go Decision Gate #4 [1 day]
**Priority**: Critical
**Dependencies**: Task 3.8
**Owner**: PI + Technical lead + Clinical advisory board

**Decision Criteria**:

| Metric | Threshold | Weight | Result |
|--------|-----------|--------|--------|
| Text performance maintained | r ≥ 0.70 | Critical | PASS/FAIL |
| Vision performance | Acc ≥ 70% | High | PASS/FAIL |
| Clinical validation | AUC ≥ 0.75 | High | PASS/FAIL |
| Developmental trajectories | Match literature | Medium | PASS/FAIL |
| System stability | No critical bugs | Critical | PASS/FAIL |

**Outcomes**:

**GO (Proceed to Publication)**:
- ✅ ≥4/5 criteria passed
- Action: Begin publication preparation
- Timeline: Submit paper by Week 50

**CONDITIONAL GO**:
- ⚠️ 3/5 criteria passed
- Action: Address specific issues, 2-4 week delay
- Re-evaluate before publication

**NO-GO**:
- ❌ <3 criteria passed
- Action: Major revision or pivot
- Consider phased release (text-only first, vision later)

---

### Week 41-48: Publication & Release Preparation

#### Task 3.10: Academic Paper Writing [4 weeks, PARALLEL]
**Priority**: High
**Dependencies**: Task 3.8
**Owner**: PI + Co-authors

**Target Journal**: Nature Human Behaviour, Psychological Science, or PNAS

**Paper Structure**:
1. **Abstract** (150 words)
2. **Introduction** (4-5 pages)
   - Motivation: CENTaUR → Psych-201 → Ko-CENTaUR
   - Korean cognitive modeling gap
   - Clinical importance
3. **Methods** (6-8 pages)
   - Participants (N=5,500+550)
   - Tasks and measures
   - Model architecture (EXAONE + Qwen2-VL)
   - Training procedure (distillation)
   - Evaluation metrics
4. **Results** (8-10 pages)
   - Tokenization efficiency
   - Korean norm validation
   - Developmental trajectories
   - Clinical discrimination
   - Multimodal integration
5. **Discussion** (4-5 pages)
   - Implications for cognitive science
   - Clinical applications
   - Cross-linguistic cognitive modeling
   - Limitations and future work
6. **References** (3-4 pages)

**Supplementary Materials**:
- Full model specifications
- Extended evaluation tables
- Failure analysis
- Ethical considerations

**Timeline**:
- Week 41-42: First draft
- Week 43-44: Internal review and revision
- Week 45-46: External collaborator feedback
- Week 47-48: Final polishing
- Week 49: Submit to journal

---

#### Task 3.11: Psych-201 Integration [2 weeks, PARALLEL]
**Priority**: High
**Dependencies**: Task 3.7
**Owner**: Technical lead

**Goal**: Prepare Ko-CENTaUR as official Psych-201 contribution

**Requirements**:
1. **Code Integration**:
   - Follow Psych-201 repository structure
   - Adapt to existing data format standards
   - Add Korean language support to core Psych-201

2. **Documentation**:
   - README for Korean module
   - Tutorial notebook
   - API documentation

3. **Pull Request**:
   - Create feature branch: `ko-centaur-integration`
   - Submit PR to Psych-201 main repo
   - Address reviewer feedback

**Communication**:
- Contact Marcel Binz (Psych-201 lead)
- Discuss integration strategy
- Coordinate release timing

**Deliverable**: Merged PR in Psych-201 repository

---

#### Task 3.12: Public Model Release [1 week]
**Priority**: High
**Dependencies**: Task 2.10 (public adapter), Task 3.11
**Owner**: Technical lead

**Release Components**:
1. **Public Adapter**:
   - Upload to HuggingFace Hub: `ko-centaur/exaone-32b-public-adapter`
   - License: Apache 2.0
   - Model card with full documentation

2. **Inference Code**:
   - Release on GitHub: `ko-centaur/inference`
   - Example scripts
   - Docker container for easy deployment

3. **Demo**:
   - HuggingFace Spaces demo
   - Interactive cognitive assessment
   - Korean language interface

**What NOT to Release**:
- ❌ Private adapter (ABCD, sensitive clinical data)
- ❌ Raw participant data (only aggregated/de-identified)
- ❌ Proprietary clinical measures

**Model Card Contents**:
```markdown
# Ko-CENTaUR: Korean Cognitive Developmental Model

## Model Description
- Base: EXAONE-3.0-32B distilled from Qwen 2.5-72B
- Purpose: Predict cognitive and clinical measures from Korean behavioral data
- Training data: 5,500 Korean participants, ages 0-18+

## Intended Use
- Research in developmental cognitive science
- Clinical screening (with appropriate validation)
- Cross-linguistic cognitive modeling

## Limitations
- Trained on Korean population only
- Not a replacement for clinical diagnosis
- Performance may vary in different cultural contexts

## Ethical Considerations
- No identifiable participant data
- Clinical use requires local validation
- Potential biases in training data
```

---

#### Task 3.13: Clinical Deployment Guide [1 week, PARALLEL]
**Priority**: Medium
**Dependencies**: Task 3.9 (validation)
**Owner**: Clinical lead + Technical lead

**Purpose**: Enable hospitals/clinics to deploy Ko-CENTaUR safely

**Guide Contents**:
1. **Prerequisites**:
   - GPU requirements (inference: 1× RTX 3090 or better)
   - Software dependencies
   - IT infrastructure requirements

2. **Installation**:
   - Step-by-step setup instructions
   - Docker deployment (recommended)
   - Security configurations

3. **Validation Protocol**:
   - Local validation dataset requirements (N=100 minimum)
   - Performance benchmarking
   - Bias assessment

4. **Clinical Workflow Integration**:
   - EMR integration considerations
   - Data privacy and security
   - Clinical decision support vs diagnostic tool

5. **Monitoring and Maintenance**:
   - Performance monitoring
   - Model updates
   - Incident response

6. **Regulatory Considerations**:
   - IRB approval for research use
   - Medical device regulation (if applicable)
   - Data protection compliance (Korean PIPA)

**Deliverable**: `docs/CLINICAL_DEPLOYMENT_GUIDE.md` (20-30 pages)

---

### Week 49-52: Finalization & Launch

#### Task 3.14: Paper Submission [Week 49]
**Priority**: Critical
**Dependencies**: Task 3.10
**Owner**: PI

**Actions**:
1. Final proofread
2. Prepare supplementary materials
3. Submit to target journal
4. Preprint on arXiv/PsyArXiv

**Backup Journals**:
- Primary: Nature Human Behaviour
- Secondary: Psychological Science
- Tertiary: Developmental Psychology
- Fallback: PLoS Computational Biology

---

#### Task 3.15: Public Release Coordination [Week 50]
**Priority**: High
**Dependencies**: Task 3.12, Task 3.14 (preprint)
**Owner**: PI + Technical lead

**Release Sequence**:
1. **Day 1 (Monday)**:
   - Preprint goes live
   - Model released on HuggingFace
   - Code released on GitHub
   - Psych-201 PR merged

2. **Day 2 (Tuesday)**:
   - Social media announcement
   - Email to mailing lists (CogSci, Psych-201 users)
   - Blog post with technical details

3. **Day 3 (Wednesday)**:
   - Press release (if major journal acceptance)
   - University press office coordination

4. **Day 4-5 (Thu-Fri)**:
   - Monitor community response
   - Address issues/questions
   - Update documentation based on feedback

**Communication Materials**:
- Twitter/X thread
- LinkedIn post
- Research blog post
- Email announcement template

---

#### Task 3.16: Community Engagement [Week 51-52]
**Priority**: Medium
**Dependencies**: Task 3.15
**Owner**: All team members

**Activities**:
1. **Monitor Feedback**:
   - GitHub issues
   - HuggingFace discussions
   - Social media mentions

2. **User Support**:
   - Answer questions
   - Fix bugs
   - Improve documentation

3. **Collaboration Outreach**:
   - Contact potential collaborators
   - Discuss extensions and applications
   - Plan future research directions

4. **Webinar/Tutorial** (optional):
   - Host online tutorial session
   - Demonstrate use cases
   - Q&A with community

---

#### Task 3.17: Project Retrospective [Week 52]
**Priority**: Medium
**Dependencies**: Task 3.16
**Owner**: All team members

**Goals**:
- Reflect on what worked well
- Identify lessons learned
- Document for future projects

**Topics**:
1. Technical decisions (model selection, architecture)
2. Data collection challenges
3. Timeline and budget adherence
4. Team coordination
5. Future improvements

**Deliverable**: Internal retrospective document (10-15 pages)

---

## Resource Allocation

### Personnel

| Role | Commitment | Weeks | Key Responsibilities |
|------|-----------|-------|---------------------|
| **PI** | 10% | 1-52 | Strategy, IRB, publication, oversight |
| **Technical Lead** | 80% | 1-52 | Architecture, training, release |
| **ML Engineer** | 100% | 1-48 | Implementation, training, evaluation |
| **Research Coordinator** | 50% | 1-40 | Data collection, participant management |
| **Clinical Lead** | 20% | 1-52 | Clinical validation, deployment guide |
| **Research Assistants** | 100% | 5-40 | Data collection, transcription, QC |

**Total FTE**: ~3.5 people over 12 months

---

### GPU Resources

| Phase | GPU Type | Duration | Cost | Purpose |
|-------|----------|----------|------|---------|
| Phase 1 | A100 40GB | 48 hrs | $197 | EEVE MVP training |
| Phase 2 | A100 40GB | 96 hrs | $394 | EXAONE distillation |
| Phase 2 | A100 80GB | 24 hrs | $50-110 | Qwen teacher inference |
| Phase 3 | A100 40GB | 72 hrs | $295 | Qwen2-VL training |
| Inference | RTX 3090 | 12 months | $2,400 | Production serving |
| **Total** | - | - | **$3,336** | - |

**Provider Options**:
- Primary: RunPod (most cost-effective)
- Backup: Google Colab Pro+
- University: Internal compute cluster (if available)

---

### Budget Breakdown

| Category | Phase 1 | Phase 2 | Phase 3 | Total |
|----------|---------|---------|---------|-------|
| **Compute** | $197 | $444 | $295 | **$936** |
| Training (EEVE) | $197 | - | - | $197 |
| Training (EXAONE) | - | $394 | - | $394 |
| Inference (Qwen) | - | $50 | - | $50 |
| Training (Qwen2-VL) | - | - | $295 | $295 |
| **Inference (12mo)** | - | - | - | **$2,400** |
| RTX 3090 rental | - | - | $200/mo × 12 | $2,400 |
| **Participant Compensation** | $600 | $36,000 | $129,000 | **$165,600** |
| Pilot (N=20) | $600 | - | - | $600 |
| Phase 2 (N=1,200) | - | $36,000 | - | $36,000 |
| Phase 3 (N=4,300) | - | - | $129,000 | $129,000 |
| **Personnel** | $15,000 | $60,000 | $75,000 | **$150,000** |
| Research staff | $5,000 | $20,000 | $25,000 | $50,000 |
| Technical staff | $10,000 | $40,000 | $50,000 | $100,000 |
| **Other** | $1,000 | $3,000 | $6,000 | **$10,000** |
| Software licenses | $200 | $500 | $500 | $1,200 |
| IRB fees | $500 | $1,000 | $1,500 | $3,000 |
| Conference travel | $300 | $1,500 | $4,000 | $5,800 |
| **TOTAL** | **$16,797** | **$99,444** | **$210,295** | **$326,536** |

**Note**: Major cost driver is participant compensation (~50% of budget)

---

## Risk Management

### High-Priority Risks

#### Risk 1: IRB Approval Delays
**Probability**: High (60%)
**Impact**: Critical (blocks data collection)

**Mitigation**:
- Submit IRB in Week 3 (early)
- Pre-application consultation with IRB office
- Use template from successful similar studies
- Parallel track: Start with de-identified archival data

**Contingency**:
- If delayed >8 weeks → Use publicly available Korean datasets (K-MMSE norms, KLOSA)
- Reduces sample size but maintains timeline

**Decision Point**: Week 8 - If IRB not approved, activate contingency

---

#### Risk 2: Data Quality Issues
**Probability**: Medium (40%)
**Impact**: High (degrades model performance)

**Mitigation**:
- Rigorous RA training (2-day workshop)
- Real-time QC checks during collection
- Double-entry verification for critical fields
- Weekly data quality audits

**Contingency**:
- Budget 10% time for data cleaning
- Develop automated QC scripts
- If >10% failure rate → Halt collection, retrain RAs

**Decision Point**: Week 10 - Review pilot data quality, adjust protocols

---

#### Risk 3: Model Performance Below Target
**Probability**: Medium (30%)
**Impact**: High (may need architecture change)

**Mitigation**:
- Phased validation (MVP → Intermediate → Full)
- Early Go/No-Go gates
- Multiple architecture candidates ready

**Contingency**:
- If Phase 1 MVP fails → Try EXAONE-7.8B (faster iteration)
- If Phase 2 fails → Direct Qwen 72B (higher cost but proven)
- If Korean tokenization fails → Character-level encoding

**Decision Points**: Week 12, Week 26 - Formal evaluation gates

---

#### Risk 4: Budget Overrun
**Probability**: Medium (35%)
**Impact**: High (may force scope reduction)

**Mitigation**:
- Detailed budget tracking (weekly reviews)
- 10% contingency reserve ($32,600)
- Staged funding approval (phase-by-phase)

**Contingency**:
- Reduce sample size (5,500 → 3,000)
- Skip vision module (Phase 3 optional component)
- Use cheaper GPU alternatives (3090 instead of A100 for inference)

**Decision Point**: Monthly budget reviews, escalate if >10% over

---

#### Risk 5: Key Personnel Departure
**Probability**: Low (20%)
**Impact**: Critical (major delay)

**Mitigation**:
- Comprehensive documentation (code comments, process docs)
- Knowledge sharing sessions (weekly team meetings)
- Code review and pair programming
- Version control with detailed commit messages

**Contingency**:
- Cross-train team members (ML engineer ↔ Research coordinator)
- External consultant on standby
- Phased handoff if known in advance

**Decision Point**: Immediate escalation if departure risk identified

---

### Risk Monitoring Dashboard

```
Weekly Risk Review (15 minutes):
├─ IRB Status: [On track / Delayed / Critical]
├─ Data Quality: [Good / Acceptable / Poor]
├─ Model Performance: [On target / Below / Above]
├─ Budget: [Under / On track / Over]
└─ Team: [Stable / Concerns / Critical]

Escalation Triggers:
- 2+ categories in "Critical" or "Poor" → Team meeting within 24 hours
- IRB delayed >4 weeks → Activate contingency plan
- Budget >15% over → Seek additional funding or reduce scope
```

---

## Success Metrics

### Phase 1 Success (Week 12)

| Metric | Target | Measurement | Decision |
|--------|--------|-------------|----------|
| NLL Improvement | ≥20% | Test set NLL vs baseline | Go/No-Go |
| Korean Tokenization | No truncation | Max tokens used <4096 | Go/No-Go |
| Pipeline Execution | Error-free | End-to-end run success | Go/No-Go |
| Data Quality | >95% valid | Manual QC review | Continue/Fix |

**Outcome**: Proceed to Phase 2 if all "Go/No-Go" metrics pass

---

### Phase 2 Success (Week 26)

| Metric | Target | Measurement | Decision |
|--------|--------|-------------|----------|
| Korean Norm r | ≥0.70 | Pearson correlation | Critical |
| Age Effects | Positive, p<0.05 | Correlation test | High priority |
| Clinical AUC | ≥0.75 | ROC-AUC | High priority |
| NLL Improvement | ≥40% | vs baseline | Medium priority |

**Outcome**: Proceed to Phase 3 if ≥3 metrics pass (must include Korean Norm r)

---

### Phase 3 Success (Week 40)

| Metric | Target | Measurement | Decision |
|--------|--------|-------------|----------|
| Text Performance | r ≥0.70 maintained | Correlation | Critical |
| Vision Performance | Acc ≥70% | Classification | High priority |
| Clinical Validation | AUC ≥0.75 | ROC-AUC | High priority |
| Developmental Trajectories | Match literature | Qualitative | Medium priority |
| System Stability | No critical bugs | Testing | Critical |

**Outcome**: Proceed to publication if ≥4 metrics pass

---

### Publication Success (Week 52)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Paper Submission | Week 49 | Submitted |
| Preprint Published | Week 50 | Live on arXiv |
| Model Released | Week 50 | HuggingFace Hub |
| Code Released | Week 50 | GitHub |
| Psych-201 Integration | Week 51 | PR merged |

**Definition of Success**: All 5 deliverables completed by Week 52

---

## Parallel Execution Map

### Critical Path Analysis

**Critical Path** (no slack, must complete on time):
```
IRB Submission (W4) → IRB Approval (W8) → Data Collection (W8-25) →
Full Training (W23-26) → Evaluation (W26) → Go/No-Go #3 (W26) →
System Training (W33-40) → Publication (W41-49)
```

**Total Critical Path Duration**: 49 weeks

---

### Parallel Tracks

#### Track A: Infrastructure (Weeks 1-4)
**Can run in parallel with IRB**
- Environment setup
- Repository restructure
- Tokenization benchmark
- Training script development

**Dependencies**: None
**Blocking**: None

---

#### Track B: IRB & Recruitment (Weeks 1-8)
**Longest lead time, starts immediately**
- IRB preparation (W1-3)
- IRB submission (W4)
- IRB review (W4-8)
- Recruitment materials (W3-5)

**Dependencies**: None
**Blocking**: All data collection (Track C)

---

#### Track C: Data Collection (Weeks 5-36)
**Depends on IRB, blocks training**
- Pilot collection (W5-8): 20 participants
- Phase 1 training data (W9-12): Use pilot data
- Phase 2 collection (W13-25): 2,000 participants
- Phase 3 collection (W26-36): Remaining 3,300 + visual 550

**Dependencies**: Track B (IRB approval)
**Blocking**: All training tasks

---

#### Track D: Model Development (Weeks 1-26)
**Can proceed with synthetic/pilot data**
- Training pipeline (W5-8)
- Evaluation framework (W5-8): Parallel with training
- MVP training (W9-12): Uses pilot data (20 samples)
- EXAONE setup (W13-14): Parallel with data collection
- Distillation implementation (W15-16)
- Full training (W17-26): Incremental as data arrives

**Dependencies**: Track A (infrastructure)
**Blocking**: Phase 3 (multimodal)

---

#### Track E: Multimodal Extension (Weeks 27-40)
**Depends on Phase 2 completion**
- Vision model setup (W27-28)
- Visual data collection (W27-35): Parallel with training
- Vision training (W33-35)
- Hybrid integration (W36-40)

**Dependencies**: Track D (Phase 2 complete)
**Blocking**: Publication

---

#### Track F: Publication & Release (Weeks 41-52)
**Final phase, some parallelism possible**
- Paper writing (W41-48): Can start while system finalizes
- Psych-201 integration (W45-48): Parallel with paper
- Model release (W49-50)
- Public launch (W50-51)
- Retrospective (W52)

**Dependencies**: Track E (full system)
**Blocking**: None (end of project)

---

### Parallel Execution Gantt Chart

```
Week:  1  4  8  12 16 20 24 28 32 36 40 44 48 52
       |  |  |  |  |  |  |  |  |  |  |  |  |  |
A: Infra [====]
B: IRB  [=========]
C: Data         [=============================]
D: Models       [====================]
E: Multimodal                  [==============]
F: Publication                            [====]

Critical Path: B → C → D → E → F
Parallelizable: A (anytime), C+D (partial), F (partial)
```

---

### Optimization Opportunities

1. **Early Start on Infrastructure** (Track A):
   - No dependencies, start Week 1
   - Saves 2-4 weeks if done in parallel with IRB

2. **Incremental Training** (Track D):
   - Train on partial data as it arrives
   - Don't wait for full 5,500 samples
   - Batch 1 (500 samples, W16) → Intermediate model
   - Batch 2 (2,000 samples, W20) → Production model v1
   - Batch 3 (5,500 samples, W26) → Final model

3. **Documentation During Execution** (All tracks):
   - Write docs while waiting for training to complete
   - Parallel paper writing with final evaluations
   - Saves 2-3 weeks at project end

4. **Conditional Multimodal** (Track E):
   - Vision module is optional
   - If budget/time constrained, release text-only first
   - Add vision in post-publication update

---

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1.1 Env Setup | None | 1.2, 1.3, 1.7 |
| 1.4 IRB Prep | None | 1.5, 1.6, 2.1 |
| 1.7 Training Script | 1.1, 1.2 | 1.9, 2.5 |
| 1.9 MVP Training | 1.6 (data), 1.7 | 1.10, 1.11 |
| 2.1 Data Collection | 1.4 (IRB) | 2.4, 2.6, 2.9 |
| 2.6 EXAONE Training | 2.4 (teacher), 2.1 (data) | 2.11, 3.1 |
| 3.3 Vision Training | 3.2 (visual data) | 3.4, 3.7 |
| 3.10 Paper Writing | 3.8 (evaluation) | 3.14 (submission) |

---

## Appendix: Quick Reference

### Key Milestones

| Week | Milestone | Decision |
|------|-----------|----------|
| 4 | IRB Submitted | Go/No-Go #1 |
| 8 | IRB Approved | Data collection starts |
| 12 | MVP Evaluated | Go/No-Go #2 → Phase 2 |
| 26 | Phase 2 Validated | Go/No-Go #3 → Phase 3 |
| 40 | System Complete | Go/No-Go #4 → Publication |
| 49 | Paper Submitted | - |
| 50 | Model Released | Public launch |

---

### Contact Information

**Model Providers**:
- EXAONE: LG AI Research (https://www.lgresearch.ai/)
- Qwen: Alibaba Cloud (https://qwenlm.github.io/)
- EEVE: Yanolja (https://github.com/yanolja/EEVE)

**Collaborations**:
- Psych-201: Marcel Binz (marcel.binz@helmholtz-munich.de)
- Korean Clinical Partners: TBD (coordinate through university IRB)

---

### File Structure Reference

```
CENTaUR/
├── ko_centaur/
│   ├── __init__.py
│   ├── training/
│   │   ├── train_qlora.py          # Phase 1, 2 training
│   │   ├── config.py               # Hyperparameters
│   │   └── utils.py                # Helper functions
│   ├── distillation/
│   │   ├── teacher.py              # Qwen 72B wrapper
│   │   ├── train_student.py        # Distillation training
│   │   └── loss.py                 # Combined loss functions
│   ├── evaluation/
│   │   ├── metrics.py              # CENTaURMetrics class
│   │   └── reports.py              # Report generation
│   ├── multimodal/
│   │   ├── hybrid_model.py         # EXAONE + Qwen2-VL
│   │   ├── vision.py               # Vision processing
│   │   └── router.py               # Task routing
│   ├── adapters/
│   │   ├── public.py               # Public adapter
│   │   └── private.py              # Private adapter
│   ├── data/
│   │   ├── raw/                    # Original JSONL
│   │   ├── transcription/          # Processed text
│   │   ├── metadata/               # Participant info
│   │   ├── norms/                  # Korean norms data
│   │   ├── distillation/           # Teacher outputs
│   │   └── final/                  # Train/val/test splits
│   ├── models/                     # Saved model weights
│   └── tasks/                      # Task-specific code
├── scripts/
│   ├── test_tokenization.py        # Benchmark script
│   ├── data_collection.py          # Collection utilities
│   └── qc_checks.py                # Quality control
├── docs/
│   ├── LLM_SELECTION_STRATEGY.md   # Original strategy
│   ├── IMPLEMENTATION_WORKFLOW.md  # This document
│   ├── CLINICAL_DEPLOYMENT_GUIDE.md
│   └── experiments/                # Evaluation reports
├── legacy/                          # Original CENTaUR code
├── tests/                           # Unit tests
└── README.md
```

---

### Quick Commands

```bash
# Setup environment
conda activate ko-centaur

# Phase 1: MVP Training
python ko_centaur/training/train_qlora.py \
  --model yanolja/EEVE-Korean-10.8B-v1.0 \
  --data ko_centaur/data/raw/pilot/pilot_20.jsonl \
  --output ko_centaur/models/eeve-mvp-v1

# Phase 2: Distillation
python ko_centaur/distillation/train_student.py \
  --student LGAI-EXAONE/EXAONE-3.0-32B-Instruct \
  --teacher-data ko_centaur/data/distillation/qwen_teacher_full.pt \
  --human-data ko_centaur/data/transcription/full_5500.jsonl \
  --output ko_centaur/models/exaone-distilled-production

# Phase 3: Vision Training
python ko_centaur/multimodal/train_vision.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --data ko_centaur/data/multimodal/visual_tasks.jsonl \
  --output ko_centaur/models/qwen2-vl-ko-final

# Evaluation
python ko_centaur/evaluation/metrics.py \
  --model ko_centaur/models/exaone-distilled-production \
  --test-data ko_centaur/data/final/test/ \
  --norms ko_centaur/data/norms/korean_norms.json \
  --output docs/experiments/phase2_evaluation.md
```

---

**End of Document**

**Next Actions**:
1. Review and approve workflow
2. Set up project tracking (Jira, Asana, or GitHub Projects)
3. Schedule Week 1 kickoff meeting
4. Begin Task 1.1 (Environment Setup)
5. Initiate Task 1.4 (IRB Preparation) in parallel
