# Hand-Eye Calibration Strategy - Guaranteed Diversity + Visibility

## Problem Statement

Need to collect 15 poses that:
1. **Keep checkerboard visible** (all poses must detect the board)
2. **Maximize diversity** (different positions AND orientations)
3. **Ensure good calibration** (reduce position error from 463mm to <10mm)

## Why Previous Approach Failed

**Old strategy: Random joint movements**
- Result: 328mm position spread (should be <10mm)
- Issue: All poses had similar camera orientations
- Hand-eye calibration needs **rotation diversity**, not just position diversity

## New Smart Strategy

### Phase 1: Manual First Pose (30 seconds)
1. Enable gravity compensation mode
2. Move robot by hand until checkerboard fully visible
3. Press 's' to save and lock position
4. This becomes the "base pose" - we know the board is visible from here

### Phase 2: Automatic 14 Poses (5-10 minutes)

**Core Idea: Move camera AROUND the checkerboard in a sphere**

Think of it like taking photos of an object from different angles:
- Front-left view
- Front-right view
- Looking from above
- Looking from below
- Tilted views

**Implementation: 4 Rings of Poses**

```
Ring 1: Horizontal circle around board (4 poses)
  - Move left/right (joints 1-2)
  - Moderate camera tilt (joints 4-5)
  - Like walking around a table

Ring 2: Vertical variations (4 poses)
  - Move up (joint 3 positive)
  - Move down (joint 3 negative)
  - Adjust tilt to keep looking at board

Ring 3: Maximum wrist rotations (4 poses)
  - Strong joint 4,5,6 changes
  - CRITICAL for calibration quality!
  - Like tilting your phone at different angles

Ring 4: Combined movements (2 poses)
  - Large position + orientation changes
  - Maximum diversity
```

### Key Features

**1. Spherical Coverage**
- Camera moves in a sphere around checkerboard
- Checkerboard stays at center of camera view
- Natural diversity in viewing angles

**2. Orientation Priority**
```python
# Joints 1-3: Position (X, Y, Z) - moderate changes
# Joints 4-6: Orientation (wrist rotations) - LARGE changes
```
Hand-eye calibration is MORE sensitive to orientation than position!

**3. Diversity Checking**
```python
# Before accepting a pose, check:
orientation_diff = sum(abs(new_joints[4:6] - prev_joints[4:6]))
if orientation_diff < 0.15:  # Too similar
    add_more_variation()
```
Ensures no two poses are too similar.

**4. Adaptive Retry**
- If board not visible: try different variation
- If pose too similar: add random jitter
- Keep trying until 14 successful diverse poses
- Max 70 attempts (5x safety factor)

## Expected Improvements

| Metric | Old Strategy | New Strategy |
|--------|-------------|--------------|
| Position spread | 328 mm | <10 mm |
| Equation error | 0.62 | <0.01 |
| Improvement % | 29% | >95% |
| Orientation diversity | Low | High |

## How Diversity Guarantees Good Calibration

**Math Background: AX=XB**
- A = Robot motion between poses
- X = Hand-eye transform (unknown)
- B = Board motion as seen by camera

**Why orientation matters:**
```
If all poses have same orientation:
  - Rotation part of X is poorly constrained
  - Solution is "ill-conditioned"
  - Small measurement errors → large calibration errors

If poses have diverse orientations:
  - Rotation part of X is well-constrained
  - Solution is "well-conditioned"
  - Robust least-squares optimization
  - Errors average out: √N reduction
```

**Practical example:**
```
Imagine calibrating a ruler:
❌ Bad: Measure same spot 15 times
   → Doesn't tell you about ruler errors elsewhere

✓ Good: Measure 15 different points
   → Reveals systematic errors, enables correction
```

## Verification

After collecting new data, run:
```bash
python src/compute_hand_eye.py
```

Check log for:
- ✓ Position spread: <10 mm (was 328 mm)
- ✓ Equation error: <0.01 (was 0.62)
- ✓ Improvement: >95% (was 29%)
- ✓ Reprojection error: <2 pixels

## Technical Details

### Square Size
- **24.69 mm** (measured with calipers)
- Critical for accurate solvePnP
- Small error (0.31mm) caused 13% error in previous run

### Joint Limits
All variations keep joints within safe limits:
- Base joint movements: ±0.3 rad
- Wrist rotations: ±0.35 rad
- Never exceed robot workspace

### Visibility Guarantee
- All variations are RELATIVE to base pose
- Base pose has board visible (verified manually)
- Variations move camera in small increments
- Board stays within camera FOV for most attempts
- Retry logic handles failures

## Summary

**Old approach:** "Move randomly and hope for diversity"
**New approach:** "Move systematically around checkerboard like a photographer"

Result: Guaranteed visibility + Maximum diversity = Excellent calibration!
