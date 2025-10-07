# Ko-CENTaUR MVP Go/No-Go Decision Checklist

**Purpose**: Evaluate Phase 1 (Weeks 1-12) outcomes to determine if the project should proceed to Phase 2 (Korean data integration) or requires pivoting.

**Decision Date**: [To be filled after Phase 1 completion]

**Decision Maker**: [PI Name]

---

## Critical Success Criteria (Must Pass All)

### 1. Model Training Success ⬜
**Metric**: Successfully trained EXAONE-3.0-7.8B-Instruct on Psych-101 (60K samples)

**Go Criteria**:
- ✅ Training completes without OOM errors
- ✅ Final loss < 1.0 (reasonable convergence)
- ✅ Model generates coherent responses
- ✅ Training time < 8 hours on single RTX 3090

**Verification**:
```bash
# Check training log
cat /scratch/connectome/connectome1/ko-centaur/logs/train_psych101_full_*.log

# Verify model saved
ls /scratch/connectome/connectome1/ko-centaur/models/exaone-psych101-full/
```

**Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Pass | ⬜ Fail

**Notes**:
_[Record final loss, training time, any issues encountered]_

---

### 2. Evaluation Metrics Implementation ⬜
**Metric**: Core evaluation framework functional

**Go Criteria**:
- ✅ NLL and accuracy computation works
- ✅ Korean norm correlation function implemented
- ✅ Age effect validation functional
- ✅ Clinical discrimination (AUC) computes

**Verification**:
```bash
# Run unit tests
cd /Users/jiookcha/Documents/git/CENTaUR
python -m pytest ko_centaur/evaluation/test_metrics.py
```

**Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Pass | ⬜ Fail

**Notes**:
_[Record test results, any failing tests]_

---

### 3. Tokenization Efficiency Validated ⬜
**Metric**: EXAONE demonstrates superior Korean tokenization vs Llama

**Go Criteria**:
- ✅ EXAONE tokens/char < 1.2 for Korean text
- ✅ EXAONE ≥ 1.5× more efficient than Llama
- ✅ Benchmark runs successfully on all 4 models

**Verification**:
```bash
# Run benchmark
python ko_centaur/benchmarks/tokenization_benchmark.py

# Check results
cat docs/experiments/tokenization_benchmark_results.json
```

**Expected Results**:
- EXAONE: ~1.0-1.1 tokens/char
- Llama: ~2.3-2.5 tokens/char
- Relative efficiency: ~2.0-2.5×

**Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Pass | ⬜ Fail

**Notes**:
_[Record actual efficiency ratios]_

---

### 4. IRB and Data Pipeline Ready ⬜
**Metric**: IRB-ready documentation and data standardization framework

**Go Criteria**:
- ✅ IRB protocol template complete
- ✅ DUA template complete
- ✅ Data standardization script functional
- ✅ De-identification checklist documented

**Verification**:
```bash
# Check documentation
ls docs/irb/
# Should contain: IRB_PROTOCOL_TEMPLATE.md, DUA_TEMPLATE.md

# Test data standardization
cd ko_centaur/data
python standardize_korean_data.py
```

**Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Pass | ⬜ Fail

**Notes**:
_[Record any missing documentation or issues]_

---

## Important Success Criteria (Pass ≥3 of 4)

### 5. Repository Structure and Documentation ⬜
**Metric**: Clean, organized codebase with clear documentation

**Go Criteria**:
- ✅ ko_centaur/ module structure established
- ✅ README.md with quick start guide
- ✅ SETUP_GUIDE.md for server environment
- ✅ Legacy code archived or removed

**Status**: ⬜ Pass | ⬜ Partial | ⬜ Fail

**Notes**:
_[Record documentation quality, usability feedback]_

---

### 6. Model Performance Baseline ⬜
**Metric**: Psych-101 performance meets reasonable baseline

**Go Criteria**:
- ✅ Test set NLL < 1.5
- ✅ Token accuracy > 60%
- ✅ Model predictions better than random (50%)

**Verification**:
```bash
# Run evaluation
python ko_centaur/evaluation/evaluate_psych101.py \
  --model /scratch/.../exaone-psych101-full \
  --dataset marcelbinz/Psych-101
```

**Status**: ⬜ Pass | ⬜ Partial | ⬜ Fail

**Notes**:
_[Record actual NLL, accuracy, comparison to baseline]_

---

### 7. Technical Infrastructure Validated ⬜
**Metric**: Server environment stable and reproducible

**Go Criteria**:
- ✅ Python 3.10 + PyTorch 2.5.1 + Transformers 4.57.0 working
- ✅ 4-bit quantization + LoRA training functional
- ✅ GPU utilization > 70% during training
- ✅ Environment reproducible from SETUP_GUIDE.md

**Verification**:
```bash
# Check environment
conda activate ko-centaur
python --version  # Should be 3.10.x
python -c "import torch; print(torch.__version__)"  # 2.5.1+cu121
python -c "import transformers; print(transformers.__version__)"  # 4.57.0
```

**Status**: ⬜ Pass | ⬜ Partial | ⬜ Fail

**Notes**:
_[Record any environment issues, GPU utilization]_

---

### 8. Collaboration Framework Established ⬜
**Metric**: Multi-lab data sharing framework ready for outreach

**Go Criteria**:
- ✅ DUA template legally reviewed (if required)
- ✅ IRB protocol ready for submission
- ✅ Co-authorship guidelines clear
- ✅ Data format standardization documented

**Status**: ⬜ Pass | ⬜ Partial | ⬜ Fail

**Notes**:
_[Record legal review status, IRB submission timeline]_

---

## Recommended Success Criteria (Nice to Have)

### 9. Preliminary Korean Language Testing ⬜
**Metric**: Model demonstrates basic Korean language understanding

**Test**: Generate responses to simple Korean prompts

**Go Criteria**:
- ✅ Model generates grammatically correct Korean
- ✅ Responses contextually appropriate
- ✅ No obvious language mixing errors

**Verification**:
```bash
# Quick test
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('/scratch/.../exaone-psych101-full')

prompt = '[|user|]오늘 기분이 어떠신가요?[|endofturn|][|assistant|]'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
"
```

**Status**: ⬜ Pass | ⬜ Partial | ⬜ Fail

**Notes**:
_[Record sample outputs, language quality]_

---

### 10. Performance Optimization Potential ⬜
**Metric**: Identified optimization opportunities for Phase 2

**Go Criteria**:
- ✅ Documented training bottlenecks
- ✅ Identified hyperparameter tuning opportunities
- ✅ Multi-GPU scaling strategy outlined

**Status**: ⬜ Pass | ⬜ Partial | ⬜ Fail

**Notes**:
_[Record optimization ideas, expected improvements]_

---

## Decision Matrix

### GO Decision Criteria
**Proceed to Phase 2** if:
1. **All 4 Critical Success Criteria PASS** (100% required)
2. **≥3 of 4 Important Success Criteria PASS** (75% required)
3. **≥1 of 2 Recommended Success Criteria PASS** (50% suggested)

### NO-GO Decision Criteria
**Halt or Pivot** if:
1. **Any Critical Success Criterion FAILS** → Major blocker
2. **<2 Important Success Criteria PASS** → Insufficient foundation
3. **Combination of failures indicates fundamental issues** → Re-evaluate approach

### CONDITIONAL GO
**Proceed with Adjustments** if:
1. **All Critical PASS + 2-3 Important PASS** → Address gaps in parallel with Phase 2
2. **Specific, fixable issues identified** → Short iteration cycle to resolve

---

## Risk Assessment

### High-Risk Red Flags (Automatic NO-GO)
- ⚠️ **Training cannot complete** (OOM errors, divergence)
- ⚠️ **Model performance worse than random** (accuracy < 50%)
- ⚠️ **EXAONE tokenization NOT more efficient** than Llama (defeats core hypothesis)
- ⚠️ **IRB barriers** that block data access indefinitely

### Medium-Risk Yellow Flags (Conditional GO)
- ⚠️ Training time > 8 hours (optimization needed but not blocker)
- ⚠️ Test accuracy 50-60% (weak but potentially improvable)
- ⚠️ Documentation incomplete (can be finished during Phase 2)

### Low-Risk Green Flags (Proceed Confidently)
- ✅ Training completes in 4-6 hours with loss < 1.0
- ✅ Test accuracy > 70%
- ✅ EXAONE 2× more efficient than Llama
- ✅ All documentation complete and tested

---

## Action Items Based on Decision

### If GO (Proceed to Phase 2)
**Immediate Actions**:
1. **Week 13**: Submit IRB protocol to [Institution IRB]
2. **Week 13-14**: Begin outreach to collaborating labs with DUA
3. **Week 14-16**: Optimize training pipeline based on Phase 1 learnings
4. **Week 16+**: Integrate first Korean lab dataset

**Success Metrics for Phase 2**:
- IRB approval within 4-6 weeks
- ≥3 labs sign DUA by Week 20
- First Korean adapter trained by Week 24

---

### If CONDITIONAL GO (Address Gaps)
**Required Improvements**:
1. **Training Issues**: Optimize hyperparameters, try gradient accumulation adjustments
2. **Performance Issues**: Experiment with learning rate, warmup steps, more epochs
3. **Documentation Gaps**: Complete missing sections before lab outreach

**Timeline**: 2-4 week improvement sprint before Phase 2 start

---

### If NO-GO (Halt or Pivot)
**Pivot Options**:
1. **Alternative Model**: Try EEVE-Korean-10.8B or Qwen2.5-7B if EXAONE fails
2. **Reduced Scope**: Focus on single clinical domain (e.g., depression only)
3. **English-Only Pilot**: Continue with Psych-101, defer Korean adaptation
4. **Methodology Shift**: Reconsider QLoRA vs full fine-tuning trade-offs

**Decision Timeline**: 2-week evaluation → pivot decision → restart or terminate

---

## Sign-Off

**Phase 1 Completion Date**: _______________

**Decision**: ⬜ GO | ⬜ CONDITIONAL GO | ⬜ NO-GO

**Primary Decision Maker**: _______________
Signature: _______________ Date: _______________

**Technical Advisor**: _______________
Signature: _______________ Date: _______________

**Notes and Rationale**:
_[Detailed explanation of decision, next steps, timeline adjustments]_

---

## Appendix: Quick Reference Commands

### Check Training Status
```bash
# View latest log
tail -f /scratch/connectome/connectome1/ko-centaur/logs/train_psych101_full_*.log

# Check GPU usage
nvidia-smi

# Monitor training
watch -n 5 nvidia-smi
```

### Run Evaluations
```bash
# Tokenization benchmark
cd /Users/jiookcha/Documents/git/CENTaUR
python ko_centaur/benchmarks/tokenization_benchmark.py

# Evaluation metrics
python ko_centaur/evaluation/evaluate_psych101.py
```

### Environment Verification
```bash
# Activate environment
conda activate ko-centaur

# Check versions
python --version
pip list | grep -E "torch|transformers|peft|bitsandbytes"
```

---

**Last Updated**: [To be filled]

**Document Version**: 1.0
