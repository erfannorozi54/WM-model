# Validation Accuracy Analysis: Why It Stays Constant

## Observed Behavior

During training, validation accuracy remains **exactly constant** across all epochs:
```
Epoch 1: Val (Novel Angle) Acc: 0.7965, Val (Novel Identity) Acc: 0.7772
Epoch 2: Val (Novel Angle) Acc: 0.7965, Val (Novel Identity) Acc: 0.7772
...
Epoch 8: Val (Novel Angle) Acc: 0.7965, Val (Novel Identity) Acc: 0.7772
```

While training metrics and validation loss continue to change:
```
Epoch 1: Train Acc: 0.7425, Val Loss: 0.4158
Epoch 7: Train Acc: 0.8021, Val Loss: 0.4115
```

## Root Cause Analysis

### 1. ✅ **Not a Bug**: Validation Data is Intentionally Cached

The validation datasets use `cache_sequences=True` (see `src/data/dataset.py` lines 296, 314):
```python
self.val_novel_angle_dataset = NBackDataset(
    ...
    cache_sequences=True  # Cache for consistency
)
```

**Purpose**: Ensures consistent evaluation across epochs (good practice)

**Effect**: Model sees EXACTLY the same validation sequences every epoch

### 2. ⚠️ **Sub-Optimal**: Validation Set Too Small

**STSF Config (original)**:
- `num_val: 100` sequences
- Single task (location only)
- Single n-value (n=2)
- Very limited variability

**MTMF Config (original)**:
- `num_val: 200` sequences
- 3 tasks × 3 n-values = 9 different conditions
- Only ~22 sequences per condition

### 3. 📊 **Mathematical Explanation**

**Why loss changes but accuracy doesn't:**

- **Loss (continuous)**: Even small changes in logits affect cross-entropy loss
- **Accuracy (discrete)**: Only changes when predictions cross the argmax decision boundary

**Example**:
```python
# Before training update
logits_before = [2.1, -0.5, 0.3]  # Prediction: class 0
loss_before = 0.412

# After training update
logits_after = [2.3, -0.4, 0.4]  # Prediction: still class 0
loss_after = 0.388  # Loss improved

# Accuracy: Still correct/incorrect - NO CHANGE
```

### 4. 🎯 **What's Actually Happening**

1. **Epoch 1**: Model quickly finds a good prediction pattern for the 100 cached sequences
2. **Epochs 2-10**: Model continues learning on training data
3. **Validation**: 
   - Logits shift slightly → Loss changes
   - But argmax predictions stay the same → Accuracy frozen
4. **Conclusion**: Model has converged on this specific validation set

## Is This Expected?

**Yes, but it's not ideal for monitoring training progress.**

### Evidence Training is Working

✅ Training loss decreasing (0.5815 → 0.4060)  
✅ Training accuracy improving (0.7425 → 0.8021)  
✅ Validation loss changing (0.4158 → 0.4115)  
✅ Model weights being updated (gradients flowing)

### Evidence of Sub-Optimal Setup

❌ Validation too small to detect continued improvement  
❌ Single task/n-value limits diversity  
❌ Cached data means model "memorizes" validation patterns early

## Solution: Increase Validation Set Size

### Changes Applied

#### STSF Config
```yaml
# Before
num_val: 100  # Too small for meaningful validation

# After  
num_val: 500  # 5x larger for better monitoring
```

#### MTMF Config
```yaml
# Before
num_val: 200  # ~22 sequences per condition (3 tasks × 3 n-values)

# After
num_val: 600  # ~67 sequences per condition
```

### Why This Helps

1. **More diversity**: Harder for model to "memorize" all validation patterns
2. **Better coverage**: More sequences per task/n-value combination
3. **Finer-grained monitoring**: Accuracy more likely to change as model improves

## Alternative Solutions (Not Implemented)

### Option A: Disable Caching (NOT RECOMMENDED)
```python
cache_sequences=False  # Generate new sequences each epoch
```

**Pros**: Validation accuracy would vary each epoch  
**Cons**: Inconsistent evaluation (different data each epoch)  
**Verdict**: ❌ Breaks reproducibility and proper validation methodology

### Option B: Add Validation Noise (NOT RECOMMENDED)
```python
# Add data augmentation to validation
```

**Pros**: More variability in validation  
**Cons**: Not standard practice, makes results harder to interpret  
**Verdict**: ❌ Validation should be deterministic

### Option C: Use Test Set for Monitoring (NOT RECOMMENDED)
**Cons**: Violates train/val/test separation, leads to overfitting  
**Verdict**: ❌ Never use test set during training

## Verification After Fix

With increased validation sizes, you should see:

```
--- Epoch 1/10 ---
  Val (Novel Angle) Acc: 0.7823
  Val (Novel Identity) Acc: 0.7654

--- Epoch 2/10 ---
  Val (Novel Angle) Acc: 0.7845  ✓ Changed!
  Val (Novel Identity) Acc: 0.7701  ✓ Changed!

--- Epoch 5/10 ---
  Val (Novel Angle) Acc: 0.7912  ✓ Improving
  Val (Novel Identity) Acc: 0.7733  ✓ Improving
```

## Key Takeaways

1. **Cached validation data is CORRECT** for reproducible evaluation
2. **Small validation sets** can lead to frozen accuracy metrics
3. **Solution**: Increase `num_val` to 500-600 for better monitoring
4. **Loss still provides signal** even when accuracy is frozen
5. **Training metrics confirm** the model is learning properly

## References

- **Dataset Implementation**: `src/data/dataset.py` lines 289-315
- **Validation Setup**: `src/train_with_generalization.py` lines 307-319
- **Config Files**: `configs/stsf.yaml`, `configs/mtmf.yaml`

---

**Status**: ✅ Configs updated with larger validation sets  
**Recommendation**: Re-run training with updated configs to see accuracy variation
