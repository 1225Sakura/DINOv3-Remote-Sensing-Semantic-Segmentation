# iSAID with DOTA Images - Complete Solution

## Problem Summary

Training iSAID with real DOTA satellite images fails catastrophically:
- **RGB mask fallback**: 39.85% mIoU ✅
- **Real DOTA images**: 0.93% mIoU ❌ (only learns background class)

## Root Causes

1. **Extreme Class Imbalance**:
   - Background: >95% of pixels
   - Target objects (planes, ships, etc.): <5% of pixels
   - Model learns to predict only background to minimize loss

2. **Domain Shift**:
   - DinoV3 pretrained on ImageNet (natural images)
   - Remote sensing images have very different feature distributions
   - Frozen backbone cannot adapt

3. **Inadequate Training Strategy**:
   - Backbone fully frozen
   - No class weighting
   - Learning rate too high for fine-tuning

## Solution Strategy

### Phase 1: Use Class Weights (Quick Fix)
- Calculate inverse frequency weights for each class
- Weight loss function to balance training
- Keep backbone frozen initially

### Phase 2: Partial Backbone Unfreezing
- Unfreeze last 2-3 transformer blocks
- Use very small learning rate (1e-5 to 5e-5)
- Longer training (50-100 epochs)

### Phase 3: Advanced Techniques
- Focal Loss for hard examples
- Multi-scale training
- Class-balanced sampling

## Implementation Plan

### Step 1: Add class weights to dataset
✅ Already implemented in `isaid_dataset.py:223`
- Uses inverse frequency weighting
- Normalizes weights

### Step 2: Modify training script
- [ ] Add `--use_class_weights` flag
- [ ] Add `--unfreeze_layers N` option
- [ ] Implement focal loss option

### Step 3: Test with real DOTA images
- [ ] Train with class weights
- [ ] Train with partially unfrozen backbone
- [ ] Compare results

## Expected Results

| Strategy | Expected mIoU | Training Time |
|----------|---------------|---------------|
| RGB mask (baseline) | 39.85% | 593 min |
| Class weights only | 15-25% | 600 min |
| + Partial unfreeze | 30-40% | 1000 min |
| + Focal loss | 40-50% | 1200 min |

## Next Steps

1. Implement class-weighted training
2. Test on small dataset first (100 samples)
3. If successful, run full training
4. Document results
