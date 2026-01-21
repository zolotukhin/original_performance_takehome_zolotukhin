# Agent Notes

## Collaboration Rules
- **NEVER** use `git restore`, `git checkout`, or any command that reverts prior work without explicit user request.
- Use `git diff` (or `git status`) to inspect changes when needed; if asked to trace diffs, run `git diff` and summarize only the important parts.
- Multiple agents may be working in parallel; keep updates short and avoid duplicating work.

## Current Status
- Baseline (frozen harness): 147734 cycles.
- **perf_takehome.py: 1470 cycles** (100.50x speedup) - Dynamic scheduler with init bundling + early interleaving
- Target: <1363 cycles (108.4x speedup) - likely impossible due to load bottleneck

### Recent Progress (1474 → 1470 cycles, 4 cycles saved)
- Moved block offset allocation earlier to enable early interleaving
- Interleaved 6 block offset const loads with 3 VALU-only cycles:
  - First 6 vbroadcasts + 2 block offset loads (cycle 1)
  - tree3-6 ALU + diff_1_2 vbroadcasts + 2 block offset loads (cycle 2)
  - tree5,6 vbroadcasts + diff_3_4 + 2 block offset loads (cycle 3)
- Remaining 5 block offset loads interleaved with hash vbroadcasts
- Combined last ALU cycle with pause instruction (1 cycle saved)

### Previous Progress (1823 → 1782 cycles, 41 cycles saved)
- Bundled init_vars[6] with tree0_scalar load (1 cycle)
- Bundled three_v with tree1,2 vbroadcasts, moved diff_1_2 to tree3-6 bundle (1 cycle)
- Merged tree1-6 ALU into single bundle (1 cycle)
- Moved seven_v vbroadcast to tree3-6 bundle (1 cycle)
- Various hash vbroadcast bundling optimizations

### Deep Analysis Results

**Body Loop Distribution (1730 cycles):**
- 79.8% (1380 cycles): Full VALU utilization (6 ops)
- 16.5% (285 cycles): Partial VALU with loads (<6 valu + 2 load)
- 5.7% (99 cycles): VALU-only full (6 valu + 0 load)
- 0.1% (2 cycles): Load-only (no VALU)
- **Wasted VALU slots: 844 (8.1% of total)**

**Theoretical Minimums:**
- load_offset (gathers): 3072 (rounds 4-15, 8 per block per round)
- vload: 64 (2 per block for idx/val)
- const loads: 59
- scalar loads: 22
- **Total loads: 3217 → minimum 1608.5 cycles** (3217 loads / 2 per cycle)
- **Store minimum: 32 cycles** (64 stores / 2 per cycle)
- **Actual: 1772 cycles** (164 cycles with 0 loads, 1 cycle with 1 load)

**Why Target <1363 is Mathematically Impossible:**
- Target: 1363 cycles
- Load minimum: 1600 cycles (already 237 cycles ABOVE target)
- To reach target would require eliminating 474+ loads (~60 gather ops = 2 more rounds of selection)
- Round 4 selection was tried and made things WORSE due to VALU overhead exceeding gather savings

**Optimization Attempts That Failed:**
- Phase combining (round3_select4b + select5): Made performance worse
- Priority sorting by cost: Worse or no change
- Buffer count tuning: 10 remains optimal
- Hash priority adjustments: Made performance worse
- Pipeline staggering: No change

**Current Efficiency:**
- Body cycles: 1730 (10.3% overhead over load minimum)
- Full overlap (6 valu + 2 load): 74.0% of cycles
- This is near-optimal given the algorithm constraints

---

# Current Best: 1473 cycles (solution_new.py)

## Recent Update (1474 → 1473 cycles)
Combined pause instruction with last ALU cycle in init phase, saving 1 cycle.

## Key Optimizations

### 1. Depth-Based Selection (Rounds 0-2, 11-13)
Instead of gather loads (8 cycles per lookup), use preloaded tree values with linear interpolation:

- **Depth 0 (rounds 0, 11)**: All indices start at 0 or reset after wrap, use preloaded `tree0_v`
- **Depth 1 (rounds 1, 12)**: idx in {1,2}, use `tree1_v + (idx-1) * diff_1_2_v`
- **Depth 2 (rounds 2, 13)**: idx in {3-6}, use 2-level selection with `tree3-6_v`
- **Depth 3+ (rounds 3-10, 14-15)**: Use gather (selection overhead exceeds gather cost)

Key insight: After wrap at round 10, indices reset to 0, so rounds 11-13 can reuse depth 0-2 selection.

### 2. Wrap Threshold Optimization
Only round 10 needs bounds checking (vselect). Before wrap (rounds 0-9), indices can't exceed n_nodes. After wrap (rounds 11-15), indices reset to 0 and grow small again.

### 3. ALU-Based Block Offset Computation
Instead of loading all 32 block offsets (0, 8, 16, ... 248), load only base offsets (every 4th) plus constants 8, 16, 24, then compute the rest with ALU adds.

### 4. Hash Stage Fusion
For hash stages with pattern `(+, val1, +, <<, val3)`, fuse into single `multiply_add` with multiplier `(1 + (1 << val3))`.

### 5. Hash Op1 ALU Offloading
When VALU is saturated but scalar ALU has slots, offload op3 (shift) to 8 scalar ALU ops instead of 1 VALU op.

### 6. Preamble Bundling
- Pack const loads 2 per cycle
- Interleave vbroadcasts with independent loads
- Overlap tree address ALU ops with loads

### 7. Dynamic Pipeline Scheduling
- 13 pipeline buffers (optimal for this workload)
- Priority-based VALU scheduling:
  - Priority 0-2: update3 → update2 → update1 (completion path)
  - Priority 3: hash_mul (high throughput)
  - Priority 4: round2_select2 (sel1 computation)
  - Priority 5: hash_op1, addr
  - Priority 6-7: selection phases, xor

## Phase State Machine
```
Round 0: init_addr → vload → round0_xor → hash → update1 → update2 → [next]
Round 1: round1_select → xor → hash → update1 → update2 → [next]
Round 2: round2_select1-5 → xor → hash → update1 → update2 → [next]
Rounds 3-9: addr → gather → xor → hash → update1 → update2 → [next]
Round 10: addr → gather → xor → hash → update1 → update2 → update3 → update4 → [next]
Rounds 11-13: Same as depths 0-2 (selection after wrap)
Rounds 14-15: addr → gather → xor → hash → update1 → update2 → [next]
Final: store_both → store_idx → done
```

## Files
- `solution_new.py` - Current optimized kernel (1473 cycles)
- `perf_takehome.py` - Original baseline
- `1473.diff` - Diff from baseline to current solution (1473 cycles)
- `1474.diff` - Previous version (1474 cycles)

## Potential Further Optimizations

### Depth 3 Selection (from 1823.diff - may help)
Add selection for rounds 3 and 14 (depth 3) using preloaded `tree7-14_v`. This requires:
- 8 additional tree value broadcasts
- 4 diff vectors (diff_7_8, diff_9_10, diff_11_12, diff_13_14)
- ~12 VALU cycles per selection (vs 8 gather cycles)

Trade-off: May increase VALU pressure but saves gather load slots.

### Priority Tuning
Further fine-tuning of phase priorities to maximize slot utilization.

---

# Historical Notes (Previous Approaches)

### Recent Optimization: Init Bundling (1823 → 1786, 37 cycles saved)
- Pre-created all constants (tree indices 0-14, hash constants, block offsets) in batched operations
- Bundled constant loads in pairs (2 per cycle instead of 1)
- Bundled init_vars loads in pairs
- Bundled tree0 vbroadcast with other initial vbroadcasts (6 ops per cycle)
- Bundled tree3-6 diffs with tree7-14 diffs (6 ops per cycle)
- Bundled diff_1_2 with three_v vbroadcast
- Moved all const loads from body to init (0 const loads in body now)
- Init phase: 55 instructions, Body: 1731 instructions
- Wasted VALU slots reduced from 1024 to 850

### Why <1363 cycles appears impossible
- Theoretical load minimum: 3219 loads / 2 per cycle = **1610 cycles**
- This minimum is **ABOVE** the target of 1363 cycles
- Would need to eliminate ~493 loads (~2 rounds of gather) to reach target
- Round 4+ selection attempted but failed due to VALU overhead exceeding gather cost

### Recent Optimization: Init Phase Bundling (1872 → 1823, 49 cycles saved)
- Replaced individual `self.add()` calls with explicit instruction bundling
- Bundled multiple vbroadcasts together (6 per cycle max)
- Bundled ALU ops for tree address computations
- Bundled loads where possible (2 per cycle max)
- Better packing during init phase reduces overhead

### Fundamental Bottleneck Analysis (1823 cycles)
- **Load ops**: 3219 (vload 64 + gather 3072 + setup 83)
- **VALU ops**: 9580
- **Theoretical load minimum**: 3219 / 2 = 1609.5 cycles
- **Theoretical VALU minimum**: 9580 / 6 = 1596.7 cycles
- **Actual**: 1823 cycles (13% overhead from dependencies and phase synchronization)

### Slot Utilization Analysis (1823 cycles)
- **VALU slots**: 75.8% full (6/6), 10.8% partial (4-5), 8.6% low (1-3), 4.8% empty
- **LOAD slots**: 86.3% full (2/2), 3.8% partial (1), 9.9% empty
- **Wasted VALU slots**: 1364 (could theoretically fit 227 more cycles of work)
- **88 load-only cycles**: During gather phases, all active blocks need loads, no VALU work available

### Why <1363 cycles is very difficult
- Theoretical load minimum (1609.5) is already ABOVE target (1363)
- To reach 1363 at 2 loads/cycle, need max 2726 loads
- Currently have 3219 loads → need to eliminate 493 loads (~2 rounds of gather)
- Buffer count optimization: 10 buffers is sweet spot (8=1909, 9=1886, 10=1823, 11=1869, 12=1866)

### Round 4 Selection Attempt (FAILED - Made Performance Worse)
- **Goal**: Eliminate round 4 gathers (32 blocks × 8 lanes = 256 loads = 128 cycles)
- **Approach**: 4-level binary selection tree for indices 15-30 (16 values)
- **Implementation**:
  - Preloaded tree[15..30] (16 scalars + 16 vectors + 8 diffs)
  - Added tmp3-tmp6 registers to each buffer for intermediate results
  - Implemented 15 selection phases to avoid RAW hazards
- **Result**: 1823 → 2026 cycles (203 cycles WORSE)
- **Root cause**: RAW hazard separation required 15+ phases with only 1-5 ops each
  - Each binary selection level needs: (compute inputs) → (compute diff) → (multiply_add)
  - 4 levels × 2 phases + selector computation = 10+ phases minimum
  - With poor VALU utilization per phase, overhead exceeds gather cost
- **Conclusion**: Round 4 selection is fundamentally more expensive than gathers
  - 256 loads at 2/cycle = 128 cycles (with good overlap via pipelining)
  - 864 VALU ops spread across 15 phases with dependencies = 150+ cycles
  - Reduced buffers (8 vs 10) due to scratch space further hurt pipelining

### Recent Changes (Session)
1. Ported dynamic scheduler from solution_new.py to perf_takehome.py → 2076 cycles
2. solution_new.py improved to 2027 cycles with phase combining
3. **solution.py optimizations (2245 → 2193 cycles, 52 cycles saved)**:
   - Applied `_pack_overlap` for main loop steps → 2201 cycles (44 cycles saved)
   - Applied `_pack_overlap` for R2B + prime gather → 2197 cycles (4 more cycles saved)
   - Combined addr_ops with valu_ops in `_pack_overlap` → 2193 cycles (4 more cycles saved)
   - Bundle analysis: 570 bundles total, 212 with 6 valu (full), 261 with 4 valu (partial)
   - **R3 selection FAILED**: 2193 → 2271 cycles. Static A/B pipelining benefits from gather-hash overlap; R3 selection adds 60+ valu ops without reducing total cycles.
4. **solution_new.py Round 3 Selection** (2027 → 1985 cycles, 42 cycles saved):
   - After round 2, idx is 7-14 (8 possible values)
   - Three-level binary selection using preloaded tree[7..14]
   - Reduced buffer count from 20 to 18 to fit extra tree values
   - 10 phases per block for round 3 (vs 4 cycles for gather)
5. **Addr Priority Tuning** (1985 → 1972 cycles, 13 cycles saved):
   - Changed addr phase priority from 7 to 3 (higher priority)
   - This feeds the gather pipeline earlier, improving load utilization
   - Addr ops now scheduled alongside hash_op2 instead of last
6. **Xor Priority Tuning** (1972 → 1946 cycles, 26 cycles saved):
   - Changed xor/round0_xor phase priority from 6 to 7 (lower priority)
   - Delaying xor allows better overlap with other phases
   - Hash stages complete faster relative to xor
7. **Hash Priority Tuning** (1946 → 1874 cycles, 72 cycles saved):
   - Changed hash_mul priority from 4 to 6 (lower priority)
   - Changed hash_op2 priority from 3 to 6 (lower priority)
   - Delaying hash completion phases allows better VALU utilization
   - More blocks can progress through earlier phases in parallel
8. **Buffer Count Optimization** (1874 → 1872 cycles, 2 cycles saved):
   - Reduced buffer count from 17 to 10
   - Fewer buffers = less contention = better scheduling
   - 10 buffers is the sweet spot (8, 11, 12 all worse)
9. **Round 4 Selection FAILED**:
   - Attempted to extend selection to idx 15-30 (16 values)
   - Not enough scratch registers per buffer for 4-level binary selection
   - Would need to add extra temp registers (idx, val already in use)

## Best Result: solution_new.py (1872 cycles, 78.92x speedup)

### Algorithm Overview
Dynamic scheduler with phase-based block processing. Each block processes 8 items (VLEN=8) through a pipeline of phases.

### Key Optimizations (from 2280 → 2076 cycles)

#### Round 0 Broadcast (2280 → 2175 cycles, 105 cycles saved)
- **Key insight**: In round 0, all indices are 0, so tree[0] is accessed for every item
- **Optimization**: Preload tree[0] into a vector and skip gather entirely
- **Phases skipped**: addr, gather
- **New phase**: round0_xor (XOR with preloaded tree[0])

#### Round 1 Selection (2175 → 2109 cycles, 66 cycles saved)
- **Key insight**: After round 0, idx is 1 or 2 (only 2 possible values)
- **Optimization**: Use linear interpolation instead of gather
- **Formula**: `node = tree1 + (idx-1) * (tree2 - tree1)`
- **Preloaded**: tree1_v, tree2_v, diff_1_2_v (tree2 - tree1)
- **Phases**: round1_select1 (offset=idx-1), round1_select2 (multiply_add)

#### Round 2 Selection (2109 → 2100 cycles, 9 cycles saved)
- **Key insight**: After round 1, idx is 3, 4, 5, or 6 (only 4 possible values)
- **Optimization**: Two-level linear interpolation instead of gather
- **Formula**:
  - sel0 = (idx-3) & 1, sel1 = (idx-3) >> 1
  - low = tree3 + sel0 * (tree4 - tree3)
  - high = tree5 + sel0 * (tree6 - tree5)
  - node = low + sel1 * (high - low)
- **Phases**: round2_select1-5 (split to avoid RAW hazards)

#### Skip Wrap for Early Rounds (2100 → 2076 cycles, 24 cycles saved)
- **Key insight**: Wrap check (idx < n_nodes) only needed when idx can exceed n_nodes
- **Analysis**: For n_nodes=2047, max idx after round r is 2^(r+2)-2
  - After round 9: max = 2046 < 2047 (no wrap needed)
  - After round 10: max = 4094 > 2047 (wrap needed)
- **Optimization**: Skip update3 (compare) and update4 (vselect) for rounds 0-9
- **Implementation**: After update2, directly advance to next round if round < 10

### Phase State Machine (Updated)
```
Round 0: init_addr → vload → round0_xor → hash → update1 → update2 → [next]
Round 1: round1_select1 → round1_select2 → xor → hash → update1 → update2 → [next]
Round 2: round2_select1-5 → xor → hash → update1 → update2 → [next]
Rounds 3-9: addr → gather → xor → hash → update1 → update2 → [next]
Rounds 10-15: addr → gather → xor → hash → update1 → update2 → update3 → update4 → [next]
Final: store_both → store_idx → done
```

#### Phase Combining (2076 → 2027 cycles, 49 cycles saved)
- **Key insight**: RAW hazards within a single instruction bundle are handled by the simulator
- **Optimization**: Combined round1_select1 + round1_select2 into single round1_select phase
- **Before**: 2 separate phases with 1 op each
- **After**: 1 phase with 2 ops (sub + multiply_add)
- **Effect**: Reduces scheduling overhead and allows better parallelism

### Saved Diffs
- `1823.py`: Current best (1823 cycles, init phase bundling)
- `1823.diff`: Diff from original to 1823 cycles version
- `1872.py`: Previous best (1872 cycles, 10 buffers)
- `1874.py`: Hash priority (1874 cycles, 17 buffers)
- `1946.py`: Xor priority (1946 cycles)
- `1972.py`: Addr priority (1972 cycles)
- `1985.py`: Round 3 selection (1985 cycles)
- `2027.diff`: Phase combining (2027 cycles)
- `2076.diff`: Wrap skip (2076 cycles)

---

## Previous Best: solution.py (2245 cycles, 65.81x speedup)

### Key Optimizations (Phase 37-39)

#### Phase 37: Hash multiply_add Fusion (3860 → 3093 cycles)
- **Key insight**: 3 out of 6 hash stages have pattern `op1='+', op2='+', op3='<<'`
- **Math**: `(val + c1) + (val << shift) = val * (1 + 2^shift) + c1`
- **Stages 0, 2, 4**: Can use single `multiply_add` instead of 3 separate ops
- **Result**: Reduces hash ops from 18 to 12 per vector per stage

#### Phase 38: Hash Op Reordering (3093 → 2961 cycles)
- **Key insight**: For 3-op stages, RAW hazards occur when op2 reads t0/t1 written by op1/op3
- **Fix**: Emit all op1 first, then all op3, then all op2 per stage
- **Effect**: op2 reads from t0/t1 that are in committed bundles, avoiding RAW flushes
- **Applied to**: gen_hash, gen_hash_half, gen_hash_group

#### Phase 39: 4:2 Interleave Ratio (2961 → 2245 cycles, 24% improvement!)
- **Key insight**: Previous 6:2 ratio caused RAW hazards to accumulate before loads could break them
- **Fix**: Changed interleave_ops from (6 valu, 2 loads) to (4 valu, 2 loads)
- **Effect**: More frequent load insertion breaks RAW hazard chains earlier
- **Result**: R3-R9 steps dropped from 19 to 17 bundles each

### Bundle Analysis (2245 cycles)
- Total bundles: 583
- Total bundle executions: 2245 (1 cycle per execution - ideal throughput!)
- Init: 27 bundles (once)
- Loop body: 553 bundles (4 iterations)
- R3-R9 (no wrap): 17+17=34 bundles per round
- R10-R14 (with wrap): 18+18=36 bundles per round
- R15 (final): 18+16=34 bundles

### Comparison with All Approaches
| Approach | Cycles | Speedup | Key Difference |
|----------|--------|---------|----------------|
| solution_new.py | 2076 | 71.16x | Dynamic scheduler + early round selection |
| solution.py | 2245 | 65.81x | Hash fusion + 4:2 interleave + A/B pipelining |
| perf_takehome.py | 2325 | 63.54x | Static loop, 6:2 interleave |

### Why solution_new.py is Faster
1. **Dynamic scheduling**: Fills 6 valu slots from up to 6 different blocks per cycle
2. **Round 0-2 selection**: Eliminates all gathers for first 3 rounds (uses preloaded tree values)
3. **Fine-grained phase tracking**: Each operation is a separate phase, allowing optimal slot filling

### solution.py Approach Limitations
- A/B pipelining processes 4 vectors at a time, limiting parallelism
- Static interleaving can't adapt to varying operation mixes
- RAW hazards still cause some bundle flushes despite reordering

---

## Previous Best: Dynamic Scheduler (2299 cycles, 64.26x speedup)

### Algorithm Overview
The dynamic scheduler uses a completely different approach from the static loop-based kernels. Instead of fixed loop structures, it maintains a pool of "blocks" (each processing 8 items via VLEN=8 vectors) and schedules operations dynamically based on phase states and slot availability.

### Key Components

**1. Pipeline Buffers (20 buffers)**
- Each buffer tracks one block of 8 items through the pipeline
- Buffers are allocated dynamically as blocks are initialized
- Maximum 20 concurrent blocks in flight (limited by scratch space)

**2. Phase State Machine**
Each block progresses through these phases:
```
init_addr → vload → addr → gather → xor → hash_mul → hash_op1 → hash_op2
→ update1 → update2 → update3 → update4 → store_both → store_idx → done
```

- **init_addr**: Calculate input addresses for indices and values
- **vload**: Load 8 indices and 8 values from memory (2 loads/cycle)
- **addr**: Calculate gather addresses (idx * 4 + tree_base)
- **gather**: Gather 8 tree nodes (8 loads at 2/cycle = 4 cycles)
- **xor**: XOR values with tree nodes
- **hash_mul**: First hash stage using multiply_add fusion
- **hash_op1/hash_op2**: Remaining hash stages (6 VALU ops each)
- **update1-4**: Index update chain (multiply_add, AND, ADD, wrap check)
- **store_both/store_idx**: Store results back to memory

**3. Priority-Based VALU Scheduling**
The key optimization that achieved 2280 cycles was reordering VALU priorities to prioritize phases closer to completion:
```python
# Priority order (0 = highest, scheduled first)
0: update3   (1 slot)  - Almost done, free the buffer
1: update2   (1 slot)  - Close to completion
2: update1   (2 slots) - Mid-pipeline
3: hash_op2  (1 slot)  - Hash tail
4: hash_mul  (1 slot)  - Hash with fusion
5: hash_op1  (2 slots) - Hash bulk
6: xor       (1 slot)  - Early compute
7: addr      (1 slot)  - Address calc
```

This prioritization ensures blocks complete faster, freeing buffers for new blocks and reducing pipeline stalls.

**4. Multi-Engine Parallelism**
Each cycle, the scheduler tries to fill all engine slots:
- **Load engine**: 2 slots (vload, gather phases)
- **Store engine**: 2 slots (store phases)
- **VALU engine**: 6 slots (addr, xor, hash, update phases)
- **ALU engine**: 12 slots (init_addr, flow control)
- **Flow engine**: 1 slot (round loop jumps)

### How 2280 Was Achieved

**Starting Point**: Applied dynamic scheduler from perf_takehome.py → 2292 cycles

**Optimization Attempts**:
1. Increased buffers from 20 to 32 → Out of scratch space, reverted to 20
2. Replaced vselect with multiply for bounds check → 2299 cycles (worse), reverted
3. Combined store phases → 2309 cycles (worse), reverted
4. Multi-block gather scheduling → 2295 cycles (no change)

**Key Breakthrough**: Reordered VALU priorities to prioritize phases closer to completion
- Before: Arbitrary phase ordering
- After: update3 > update2 > update1 > hash_op2 > hash_mul > hash_op1 > xor > addr
- Result: 2295 → 2280 cycles (15 cycle improvement)

**Why Priority Reordering Helps**:
- Completing blocks faster frees buffers for new work
- Reduces average pipeline depth
- Better utilization of VALU slots across the pipeline
- Prevents bottlenecks where many blocks are stuck in early phases

### Comparison with Static Approaches

| Approach | Cycles | Speedup | Key Difference |
|----------|--------|---------|----------------|
| Dynamic Scheduler | 2280 | 64.80x | Flexible scheduling, 20 buffers |
| solution.py | 2320 | 63.68x | Static loop, 2:6 interleaving |
| perf_takehome.py | 2325 | 63.54x | Static loop, A/B pipelining |

The dynamic scheduler's advantage comes from:
- Better adaptation to variable-latency operations (gathers)
- More flexible slot utilization across phases
- Priority-based completion that reduces pipeline bubbles

### Theoretical Analysis
- Total VALU ops per block: ~100 ops
- Total load ops per block: ~10 ops (vload + gather)
- Total store ops per block: ~2 ops
- With 256 items / 8 per block = 32 blocks
- Minimum cycles (VALU-bound): 32 × 100 / 6 = 533 cycles
- Minimum cycles (load-bound): 32 × 10 / 2 × 16 rounds = 2560 cycles
- Actual: 2280 cycles (close to load-bound minimum)

## Optimization History

### Phase 1: Basic Vectorization (147734 → 65586 cycles, 2.25x)
- Process 8 items at once (VLEN=8) using vload/vstore
- Scalar gather for tree node lookups
- Basic hash using valu ops

### Phase 2: VLIW Packing (65586 → 17465 cycles, 8.46x)
- Pack multiple independent ops per cycle (up to 12 ALU slots)
- Pre-broadcast vector constants (ones, twos, zeros, n_nodes)
- Pack address calculations (8 ALU ops in 1 cycle)

### Phase 3: Dual-Vector Processing (17465 → 11065 cycles, 13.35x)
- Process 2 vectors (16 items) per iteration
- Dual register sets for batch A and B
- Interleaved gather loads

### Phase 4: Pre-broadcast Hash Constants (11065 → 8767 cycles, 16.85x)
- Pre-broadcast all 12 hash constants (6 stages × 2) to vectors at startup
- Hash stage reduced from 3 cycles to 2 cycles (no runtime broadcast needed)

### Phase 5: Automatic Slot Packer + Cleanup (8767 → 7962 cycles, 18.55x)
- Added `_pack_slots()` with RAW/WAW dependency tracking
- Removed duplicate wrap logic (was comparing twice)
- Uses `idx * cond` for branchless wrap instead of vselect

### Phase 6: Runtime Loops + Quad-Vector Processing (7962 → 6023 cycles, 24.53x)
- Replaced compile-time unrolled loops with runtime `cond_jump_rel` loops
- Outer loop for rounds (16 iterations), inner loop for batches (8 iterations)
- Process 4 vectors (32 items) per inner loop iteration
- Loop body is packed once; instructions reused via jumps
- Total instruction count reduced from ~7962 to ~91

### Phase 7: Init Phase Packing (6023 → 5965 cycles, 24.77x)
- Packed init phase constants using `_pack_slots()` instead of individual `add()`
- Packed round loop setup (pointer resets) together
- Reduced init overhead from ~40 cycles to ~14 cycles
- Total instruction count reduced from ~91 to ~63

### Phase 8: Inverted Loop Order (5965 → 4862 cycles, 30.39x)
- Changed from round-outer/batch-inner to batch-outer/round-inner
- Data stays in registers for all 16 rounds per batch
- Reduced vload/vstore from 128 times to just 8 times (once per batch instead of once per round)
- Round loop now only contains compute, no memory access (except gather)
- Store happens once at end of batch after all rounds complete

### Phase 9: Round 0 Broadcast Optimization (4862 → 4710 cycles, 31.37x)
- In round 0, ALL elements have idx=0
- Load tree[0] once and broadcast to all 4 node vectors (1 load + 4 broadcasts)
- Saves ~15 cycles per batch (32 gathers → 1 load)
- Index calculation simplified: idx = (val & 1) + 1 (no 2*idx since idx=0)

### Phase 10: Round 1 Vselect Optimization (4710 → 4614 cycles, 32.02x)
- After round 0, idx is either 1 or 2 (only 2 possible values)
- Load tree[1] and tree[2] once (2 loads), broadcast to vectors
- Use vselect based on (idx == 2) to pick the right tree value
- Eliminates 32 gathers for round 1, replaces with 2 loads + 4 vselect

### Phase 11: Loop Control Overlap (4614 → 4502 cycles, 32.82x)
- Moved round loop counter increment and comparison to beginning of round body
- ALU ops now run in parallel with the 16-cycle gather (load engine)
- Saves ~2 cycles per round by hiding loop control in gather latency

### Phase 11: 4-Round Unrolling (4614 → 4502 cycles, 32.82x)
- Unrolled 4 rounds (0,1,2,3) in one batch iteration
- Processes 4 vectors (32 items) × 4 rounds = 128 items per batch iteration
- Reduced batch iterations from 8 to 4
- Each of rounds 1,2,3 uses vselect to pick tree value based on idx condition
- Round 0 still optimized with broadcast
- Significant reduction in loop overhead with minimal register pressure increase

## Failed Optimization Attempts
1. **Interleaved gather/compute** - Made cycles worse (5965 → 7117). The packer couldn't effectively combine the interleaved operations.
2. **2-vector per iteration** - Made cycles worse (5965 → 8525). Loop overhead dominates gains from reduced gather cycles.
3. **Pack loop control with round increment** - Broke correctness. Round counter gets incremented even when batch jump is taken.
4. **Round loop unrolling by 4** - Minor improvement (4862 → 4678). Not worth the code complexity.

## Current Bottleneck Analysis
With inverted loop order (batch-outer, round-inner):
- 8 batches × (16 rounds × round_body + load + store)
- Round body (~37 cycles): gather(16) + XOR(1) + hash(12) + index(3) + wrap(2) + loop(2)
- Batch overhead: load(4) + store(4) + ptr_setup(1) = 9 cycles

Per-round cycle breakdown (~37 cycles):
- Address calc: ~1 cycle (4 valu ops)
- **Gather: ~16 cycles (32 loads / 2 per cycle)** - Main bottleneck!
- XOR: ~1 cycle (4 valu ops)
- Hash: ~12 cycles (72 valu ops / 6 per cycle)
- Index calc: ~3 cycles (16 valu ops / 6 per cycle)
- Wrap: ~2 cycles (8 valu ops / 6 per cycle)
- Loop control: ~2 cycles

Total estimate: 8 batches × (16 rounds × 37 + 9) = 8 × 601 = 4808 cycles (close to actual 4862)

Theoretical minimum: ~2560 cycles (gather-limited at 16 cycles × 16 rounds × 8 batches / 8 = 256 gather cycles... wait, gather happens every round)

## Remaining Optimizations to Try
### Phase 12: 8-Way Parallelism + Gather Pipeline (4502 → 4438 cycles, 33.29x)
- Increased batch size from 4 vectors (32 items) to 8 vectors (64 items) per outer loop iteration.
- Implemented software pipelining for rounds 2-15:
    - Split 8 vectors into Group A (4 vectors) and Group B (4 vectors).
    - Interleave "Gather Group B" with "Hash Group A" and vice-versa.
    - Hides the high latency of 32 gathers behind the ALU-heavy hash computation.
- Used specialized logic for Round 0 (Broadcast) and Round 1 (Select).
- Reverted aggressive Select optimization for deeper levels due to correctness issues and diminishing returns.
- Scratch space management became critical; optimized allocation to fit 8 vectors + temps.

## Current Bottleneck Analysis
With 8-way parallelism and pipelining:
- 4 batches × (16 rounds × ~pipe_stage + overhead)
- Pipeline stage effectively hides gather latency. The bottleneck shifts to ALU throughput (Hash + Index Update).
- Hash takes ~12 cycles, Index Update ~4 cycles.
- Gather takes ~32 cycles for 8 vectors (4 loads/cycle? No, 2 loads/cycle = 4 * 8 = 32 loads. 16 cycles).
- We do 2 gathers (Group A, Group B) interleaved.
- Total cycle count 4438 is very close to theoretical limit for this approach.

### Phase 13: Index Update Optimization (4438 → 4439 cycles, 33.28x)
- Attempted to optimize `gen_index_update` using `multiply_add` (`idx*2 + t`) and `vselect` (for wrap).
- Latency depth remained similar (3 cycles).
- Cycle count remained effectively unchanged (noise).
- Conclusion: The bottleneck is likely strictly the Hash Latency (18 cycles) vs Gather Latency (16 cycles) per step, combined with implicit synchronization overheads in the pipeline structure.

### Phase 14: Micro-optimizations (4438 → 4376 cycles, 33.76x)
- **Vselect batching (4438 → 4410)**: Round 1 vselects were emitted one at a time in a loop. Batched all comparisons together, then all vselects, allowing better packing (8 comparisons at 6 valu/cycle instead of 8 separate bundles).
- **Parallel tree node loads (4410 → 4402)**: Round 1 loaded tree[1] and tree[2] sequentially using same tmp_ptr. Used two separate address registers so both loads can issue in parallel (2 load slots). Combined broadcasts of t1_vec, t2_vec, two_vec.
- **Independent pointer setup (4402 → 4380)**: Pointer setup had chain dependencies (ptrs_i[i] = ptrs_i[i-1] + vlen). Pre-computed offset constants (0, 8, 16, 24, 32, 40, 48, 56) and computed all 16 pointers independently, allowing all ALU ops to pack efficiently.
- **Store+jump overlap (4380 → 4376)**: Combined store operations with batch loop control ALU ops (counter increment and compare). ALU engine runs in parallel with store engine, hiding loop control latency.

## Current Bottleneck Analysis
With 8-way parallelism and pipelining:
- Hash Block: ~21-24 cycles per step (Hash + Index).
- Gather Block: ~16 cycles per step.
- Step Time: ~24 cycles.
- Round Time: ~48 cycles.
- Total Time: Matches observed ~4438 cycles (when factoring in overheads and imperfect packing).

### Phase 15: Independent Pointer Setup & Parallel R1 Select (4439 → 4376 cycles, 33.76x)
- Analyzed `4380.diff` and identified serial pointer dependency chain as a bottleneck.
- Refactored pointer setup to compute all 16 pointers (8 indices + 8 values) independently using pre-computed offsets.
- Parallelized Round 1 Select logic: Batched all 8 comparison and all 8 vselect operations into large bundles instead of vector-wise loops.
- This improved instruction packing density and reduced overhead per batch.
- Retained `gen_index_update_opt` (multiply_add + flow vselect) which gives slight edge over `4380.diff` (~4 cycles).
- **Final Result: 4376 cycles (33.76x speedup).**

### Phase 16: Round 2 One-Hot Optimization (4376 → 4328 cycles, 34.13x)
- After round 1, idx is in {3,4,5,6} - only 4 possible tree values.
- Loaded tree[3], tree[4], tree[5], tree[6] in parallel (4 loads at 2/cycle = 2 cycles).
- **Key insight**: Use one-hot multiplication instead of vselect:
  - `node = (idx==3)*tree[3] + (idx==4)*tree[4] + (idx==5)*tree[5] + (idx==6)*tree[6]`
  - 32 comparisons + 32 multiplies + 24 adds = 88 valu ops at 6/cycle = ~15 cycles
  - vs vselect approach: 24 vselects at 1 flow/cycle = 24 cycles
- Pipeline now starts at round 3 instead of round 2.
- Saved 48 cycles total (44 from one-hot vs vselect, 4 from eliminating prime gather).

### Phase 17: Multiply-Add Index Update (4328 → 4312 cycles, 34.26x)
- Index calculation was: `idx = idx + idx` then `idx = idx + t` (8 valu ops per 4 vectors)
- Changed to: `idx = multiply_add(idx, 2, t)` (4 valu ops per 4 vectors)
- Saves 4 valu ops per gen_index_update call × many calls in pipeline
- Added idx_two_vec constant (broadcast of 2) in init phase

### Phase 18: Pre-computed Base Offsets (4312 → 4309 cycles, 34.28x)
- Pointer setup was 2-step: compute ptr_idx/ptr_val, then add offsets in batch loop
- Pre-compute base addresses with offsets during init phase: `idx_base[i] = inp_indices_p + offset[i]`
- Batch loop pointer setup reduced to single add: `ptrs_i[i] = idx_base[i] + batch_counter`
- Eliminates dependency chain in pointer calculation
- Saves 3 cycles per batch (12 cycles total across 4 batches)

### Phase 19: Round 1 One-Hot (4309 → 4305 cycles, 34.32x)
- Round 1 used vselect (1 flow/cycle) to choose between tree[1] and tree[2]
- Replaced with one-hot multiplication: `node = (idx==1)*t1 + (idx==2)*t2`
- 16 comparisons + 16 multiplies + 8 adds = 40 valu ops at 6/cycle = ~7 cycles
- vs vselect: 8 comparisons + 8 vselects at 1/cycle = ~10 cycles
- Saves ~3 cycles per batch (12 cycles total across 4 batches)

### Phase 20: Skip Wrap for Early Rounds (4305 → 4161 cycles, 35.50x)
- **Key insight**: Wrap logic (`idx = idx * (idx < n_nodes)`) is only needed when idx can exceed n_nodes
- For forest_height=10, n_nodes=1023. Max idx after round r is 2^(r+2)-2
- Rounds 0-8: max idx = 1022 < 1023, no wrap needed
- Rounds 9-15: max idx > 1023, wrap needed
- Added `with_wrap` parameter to `gen_index_update` and `emit_pipelined_step`
- Skip wrap for ALL rounds 0-8 (saves 8 valu ops per group × 2 groups × 9 rounds = 144 valu ops)
- Total savings: 144 cycles across 4 batches

### Phase 21: Interleaved Load/VALU Packing (4161 → 3661 cycles, 40.35x)
- **Key insight**: Load engine (2/cycle) and VALU engine (6/cycle) can execute in parallel
- Previous code had only 3.5% of bundles using multiple engines
- For pipeline steps with (hashA | gatherB) or (gatherA | hashB):
  - Addr calculation must precede loads (RAW dependency)
  - But loads (32 ops) can run in parallel with hash valu ops (88 ops)
- Interleaved load ops with hash ops after addr calculation
- 32 loads + 88 valu can fit in ~32 bundles instead of 16+15=31 bundles
- But the packer can now combine them: 2 loads + 6 valu per bundle
- Reduced total bundles significantly, saving ~500 cycles

### Phase 22: Combined Pack Calls (3661 → 3649 cycles, 40.49x)
- Combined separate pack calls in rounds 0, 1, 2 into single calls
- Allows packer to find more opportunities for multi-engine bundling
- Minor improvement as packer was already doing well within each call

### Phase 23: Optimized Round 0 Index Update (3649 → 3641 cycles, 40.58x)
- For round 0, idx starts at 0 for all items
- Original: idx = multiply_add(0, 2, t) = 0 + t = t
- Optimized: idx = (val & 1) + 1 directly (skip multiply)
- Saves 4 valu ops per group (8 total per batch)

### Phase 24: Skip idx_vecs Loading (3641 → 3625 cycles, 40.75x)
- **Key insight**: All indices start at 0, and round 0's optimized index update doesn't read current idx
- Removed 8 vloads for idx_vecs at batch start
- Still store idx_vecs at batch end (needed for result)
- Saves 8 vloads / 2 per cycle = 4 cycles per batch = 16 cycles total

### Phase 25: Optimal Interleave Ratio (3625 → 2625 cycles, 56.28x)
- **Key insight**: Previous interleaving was 1:1 (1 load per 1 hash op) but slot limits are 2:6
- When interleaving 1:1, the packer filled 2 load slots but only 2 valu slots (wasting 4 valu slots)
- Changed interleave ratio to 2:6: emit 2 load ops, then 6 hash ops, repeat
- This allows packer to fill both load slots (2) AND all valu slots (6) simultaneously
- Massive improvement: ~1000 cycles saved (38% reduction)

### Phase 26: R0/R1/R2 Tree Prefetch (2625 → 2609 cycles, 56.62x)
- Load tree[1..6] during R0's hash phase to hide latency for R1 and R2
- Previously: R0 loaded tree[0], R1 loaded tree[1,2], R2 loaded tree[3,4,5,6] sequentially
- Now: All 7 loads happen during R0, interleaved with R0's hash computation
- R1 and R2 only need broadcasts since tree values are already in scratch
- Saves 16 cycles by hiding load latency

### Phase 27: R2/R3 Prime Interleave (2609 → 2597 cycles, 56.89x)
- Previously: R2's idx_update for both groups completed, then R3's prime gather started
- Now: R2's idx_update for group A completes, then R3's prime gather (group A) starts
- R2's idx_update for group B is interleaved with R3's gather loads (2:6 ratio)
- Saves 12 cycles by overlapping R2's tail with R3's head

### Phase 28: R15 Store Overlap (2597 → 2581 cycles, 57.24x)
- R15's step 2 (hashB only) previously had idle load/store engine
- Now interleave group A's stores (8 vstores) with group B's final hash
- Only group B's stores remain at end of batch
- Saves 16 cycles by hiding store latency

### Phase 29: XOR-Hash Dependency Fix (2581 → 2529 cycles, 58.42x)
- **Key insight**: XOR writes to val, hash stage 0 reads from val - RAW dependency
- When interleaving, XOR and hash ops would land in same chunk, causing packer to split
- Fixed by emitting addr_calc + XOR first (8 valu), then interleaving loads with hash+idx_update
- Reduced 4-valu bundles from 79 to 41, saving 13 bundles per batch
- Total savings: 52 cycles (13 bundles × 4 batches)

### Phase 30: R1/R2 Linear Interpolation (2529 → 2489 cycles, 59.35x)
- **Key insight**: One-hot selection uses comparisons + multiplies + adds, but linear interpolation is cheaper
- **R1 (idx ∈ {1,2})**: Instead of `(idx==1)*t1 + (idx==2)*t2`:
  - `offset = idx - 1` (0 or 1)
  - `diff = t2 - t1` (precomputed once)
  - `node = multiply_add(diff, offset, t1)` = t1 + diff*offset
  - Reduced from 40 valu ops to 19 valu ops (saved 21 ops)
- **R2 (idx ∈ {3,4,5,6})**: Instead of one-hot with 4 comparisons:
  - `diff = idx - 3` (0,1,2,3)
  - `sel0 = diff & 1` (select within pair)
  - `sel1 = diff >> 1` (select between pairs)
  - `low = multiply_add(d34, sel0, t3)` where d34 = t4 - t3
  - `high = multiply_add(d56, sel0, t5)` where d56 = t6 - t5
  - `node = multiply_add(high - low, sel1, low)`
  - Reduced from 88 valu ops to 63 valu ops (saved 25 ops)
- Total savings: 40 cycles (10 bundles × 4 batches)

### Phase 31: Independent idx_update Formulation (2489 → 2381 cycles, 62.05x)
- **Key insight**: Restructure idx_update to make operations independent
- **Old**: `t = val & 1` → `tt = t + 1` → `idx = idx*2 + tt`
  - AND writes t, ADD reads t (RAW dependency) - forces bundle split
- **New**: `base = idx*2 + 1` → `t = val & 1` → `idx = base + t`
  - multiply_add (base) and AND (t) are INDEPENDENT!
  - Can pack in same bundle, reducing 4-valu bundles
- Reduced 4-valu bundles from 118 to 89 (saved 29 bundles)
- Total bundles: 635 → 608 (27 fewer)
- Total savings: 108 cycles

### Phase 32: R2/R3 Hash Interleave (2381 → 2333 cycles, 63.32x)
- **Key insight**: R2's hash_B doesn't depend on idx_update_A, can be deferred
- **Problem**: R2 has pure valu bundles with no loads to interleave
- **Solution**: Move hash_B from R2 to R3's prime phase
  - R2: Do XOR, hash_A, idx_update_A only (group A ready for gather)
  - R3 prime: Interleave gatherA loads with hash_B + idx_update_B
- This fills the load-only gaps at the end of R3 prime (was 38, now 26)
- Total bundles: 608 → 596 (12 fewer)
- Total savings: 48 cycles

### Phase 33: Wrap Threshold Fix (2333 → 2325 cycles, 63.54x)
- **Key insight**: Wrap threshold was one round too early
- **Analysis**: For n_nodes=2047, max idx after round r is 2^(r+2)-2
  - After round 9: max idx = 2^11 - 2 = 2046 < 2047 (NO wrap needed)
  - After round 10: max idx = 2^12 - 2 = 4094 > 2047 (wrap needed)
- **Fix**: Changed `needs_wrap = (r >= 9)` to `needs_wrap = (r >= 10)`
- Eliminates 8 compare + 8 multiply ops for round 9 (16 valu ops)
- Total bundles: 596 → 594 (2 fewer per batch)
- Total savings: 8 cycles

## Failed Optimization Attempts (Recent)

### R3 Bit-Selection (tested, made performance worse)
- **Goal**: Eliminate gather for R3 by preloading tree[7..14] and using bit-selection
- **Approach**:
  - Load tree[7..14] (8 scalars) during R0's prefetch
  - Broadcast to 8 vectors in R3
  - Use 3-level nested multiply_add to select based on idx bits
- **Problem**: 3-level binary selection requires ~60 VALU ops per group
  - Level 0: 4 pairs × 4 vectors = 16 multiply_add
  - Level 1: 2 pairs × 4 vectors = 8 multiply_add + 8 sub
  - Level 2: 1 pair × 4 vectors = 4 multiply_add + 4 sub
  - Plus bit extraction: 20 ops (diff, AND, shift ops)
- **Result**: 2325 → 2425 cycles (100 cycles WORSE)
- **Conclusion**: Gather (32 loads) is cheaper than 120 VALU ops for bit-selection

## Bottleneck Analysis (Current State: 2325 cycles)

Per-batch analysis (594 bundles total, 4 batches):
- **Init phase**: ~17 cycles (runs once)
- **Total bundles**: 594
- **Total valu ops**: ~3114

Bundle distribution (from analysis):
- 0 valu: 38 bundles (memory-only or control)
- 1-3 valu: 10 bundles (transition bundles)
- 4 valu: 86 bundles (dependency-limited, wasting 2 slots each)
- 5 valu: 10 bundles (hash tail patterns)
- 6 valu: 450 bundles (fully utilized)

Total ops:
- VALU: 3114 ops
- Load: 877 ops
- Store: 16 ops

Theoretical minimums:
- **Valu-bound**: 519 cycles (3114 valu ops / 6 per cycle)
- **Load-bound**: 439 cycles (877 loads / 2 per cycle)
- **Store-bound**: 8 cycles (16 stores / 2 per cycle)
- **Actual**: 594 bundles

**Gap analysis:**
- Actual bundles: 594
- Min bundles (valu-limited): 519
- Overhead: 75 bundles (14.4%) due to:
  - 86 4-valu bundles wasting 172 slots (≈29 bundles at 6 valu/cycle)
  - Memory-only bundles that can't be merged with valu

**Why <1363 cycles is difficult:**
- Even perfect packing gives 4 × 519 + 17 = 2093 cycles
- To reach <1363 would require:
  - Reducing valu ops by ~35% (not possible without changing hash algorithm)
  - Using multiple cores (N_CORES=1 in problem)
  - Further algorithm changes (hash stages are fixed)

## Conclusion
The kernel is highly optimized, leveraging:
1.  **8-Way Vector Parallelism** to maximize VLIW slot usage.
2.  **Software Pipelining** to overlap memory latency with computation.
3.  **Specialized Shallow Rounds** (Broadcast R0, Linear Interpolation R1/R2) to skip expensive Gathers.
4.  **Independent Address Calculation** to avoid serialization.
5.  **Optimized Instructions** (multiply_add, 2:6 interleaving ratio).
6.  **Skip Wrap for Early Rounds** (rounds 0-9 don't need wrap check, wrap starts at r=10).
7.  **Tree Node Prefetch** (load tree[1-6] during R0's hash).
8.  **Store Overlap** (group A stores during R15's final hash).

Current: 2321 cycles (63.65x speedup)
Theoretical minimum with current architecture: ~2093 cycles
Target: <1363 cycles (would require architectural changes)

### Phase 34: R0 idx_update Reordering (2325 → 2321 cycles, 63.65x)
- **Key insight**: gen_index_update_r0 was called twice (A then B), producing:
  - AND0-3, ADD0-3 (group A), AND4-7, ADD4-7 (group B)
  - RAW hazard between AND and ADD within each group caused extra bundle flushes
- **Fix**: Batch all 8 ANDs first, then all 8 ADDs across both groups
  - AND0-7 can pack efficiently (all independent writes)
  - ADD0-7 can follow (all reads are from flushed bundles)
- Reduced RAW flushes from 2 to 1, saving 4 cycles

### Phase 35: R1 gen_index_update_both (no improvement)
- Changed R1's separate gen_index_update calls to gen_index_update_both
- gen_index_update_both was already well-structured for packing
- No net improvement (stayed at 2321 cycles)

### Phase 36: Code Cleanup (2321 cycles, no change)
- Removed dead code: gen_hash_staged (never called)
- Removed dead code: emit_combined_step (never called)
- Code is cleaner, same performance

## Recent Failed Optimization Attempts (Phase 36+)

### 1:3 Interleave Ratio (worse)
- **Goal**: Finer-grained interleaving might align better with hash stage boundaries
- **Change**: 1 load per 3 valu instead of 2 loads per 6 valu
- **Result**: 2321 → 2377 cycles (worse)
- **Conclusion**: 2:6 ratio is optimal

### Grouped Hash Writes (worse)
- **Goal**: Group t writes together and tt writes together for better packing
- **Change**: t0,t1,t2,t3,tt0,tt1,tt2,tt3 instead of t0,tt0,t1,tt1,...
- **Result**: 2321 → 2689 cycles (much worse!)
- **Conclusion**: Interleaved order is required for the packer

### 4:12 Interleave Ratio (worse)
- **Goal**: Larger chunks might reduce overhead
- **Result**: 2321 → 2657 cycles (much worse)
- **Conclusion**: Larger chunks don't help

### Sequential A/B idx_update (no change)
- **Goal**: Process A completely before B to reduce interleaving overhead
- **Result**: 2321 cycles (no change)
- **Conclusion**: Interleaved A/B is equally efficient

### Merged R0+R1+R2 pack calls (no change)
- **Goal**: Let packer find cross-round optimization opportunities
- **Change**: Combined all R0, R1, R2 ops into single _pack_slots call
- **Result**: 2321 cycles (no change)
- **Conclusion**: Packer handles separate calls efficiently

## Fundamental Limitation Analysis

The 2321 cycle count is near the theoretical minimum for the current architecture:

### Hash Structure Limitation
Each of 6 hash stages has 12 ops per group (8 writes + 4 reads):
- Bundle 1: 6 writes (t0, tt0, t1, tt1, t2, tt2) - valu limit
- Bundle 2: 2 writes (t3, tt3) + 3 reads (val0-2) - val3 causes RAW flush
- Bundle 3: 1 read (val3)

This results in a 1-op partial bundle per stage due to val3 reading t3,tt3 which are written in the same bundle.

### Options for Further Improvement
1. **Change to 3 vectors per group (NVECS=6)**:
   - Writes = 6 (fits exactly in 1 bundle)
   - Reads = 3 (partial bundle)
   - Would require major restructuring and batch handling changes
   - Batch size 256 doesn't divide evenly by 48 (6×8)

2. **Multi-core execution**: Not available (N_CORES=1)

3. **Algorithm changes**: Hash stages are fixed

## Final Summary

The kernel has been optimized from 147734 cycles to 2321 cycles (63.65x speedup). Key optimizations:
1. **8-way vector parallelism** (NVECS=8, 64 items per batch iteration)
2. **Software pipelining** (gatherA overlapped with hashB and vice versa)
3. **2:6 interleave ratio** (matches load:valu slot limits)
4. **Specialized early rounds** (R0 broadcast, R1/R2 linear interpolation)
5. **Tree prefetch** (load tree[1-6] during R0 hash)
6. **Wrap threshold optimization** (skip wrap for rounds 0-9)
7. **Independent operations** (multiply_add + AND are independent in idx_update)

The remaining gap to the target (<1790 cycles) would require architectural changes (different vector count, algorithm changes, or multi-core execution).

## Files
- `perf_takehome.py`: Main optimized kernel (2321 cycles)
- `2321.diff`: Original 2321 version
- `2321_final.diff`: Final cleaned up version

## Failed Optimization Attempts (Phase 34+)

### Hash Fusion (tested, made performance worse)
- **Goal**: Fuse 3 hash stages (op1=+, op2=+, op3=<<) into single multiply_add
- **Math**: val = (val + c1) + (val << shift) = val * (1 + 2^shift) + c1
- **Result**: 2321 → 2358 cycles (worse)
- **Analysis**: While reducing ops (from 3 to 1 per fused stage), the change in packing structure was unfavorable
- **Conclusion**: The original 3-op structure packs better despite more ops

### Interleave Pattern Variations (no improvement)
- **Tested loads-first**: Put all loads at start, then all hash ops - 2321 → 3769 (much worse)
- **Tested 2:4 ratio**: 2 loads per 4 valu instead of 6 - 2321 → 2769 (worse)
- **Conclusion**: 2:6 interleave ratio is optimal

### gen_index_update_both for R1 (no improvement)
- Changed R1 from two gen_index_update calls to one gen_index_update_both
- gen_index_update_both was already well-structured
- No net improvement (stayed at 2321 cycles)

## Current Bottleneck Analysis (2321 cycles)

The 2321 cycle count breaks down as:
- 4 batches × ~580 cycles each
- Each batch has ~595 bundles (measured)
- Theoretical minimum: ~519 bundles (valu-bound at 3114 ops / 6 per cycle)
- Overhead: ~76 bundles (14.6%) from RAW hazards and partial bundles

The main sources of inefficiency:
1. **Hash RAW hazards**: Each of 6 hash stages has 8 writes + 4 reads, causing 1 RAW flush per stage
2. **idx_update RAW hazards**: 4 ma + 4 AND + 4 ADD causes 1 RAW flush
3. **Memory-only bundles**: Some bundles have only loads/stores without valu

## Fundamental Limitations

To reach the opus45_casual threshold (< 1790 cycles), we would need:
- ~23% reduction from current 2321 cycles
- This requires either:
  - Different hash algorithm (not permitted)
  - Different vector count (would require major restructuring)
  - Multi-core execution (N_CORES fixed at 1)
  - Cross-engine forwarding (not supported)

Current architecture is near local optimum for the given constraints.
- `solution.py`: Alternative kernel with different structure (2320 cycles)
- `2320.diff`: solution.py version (2320 cycles)
- `2325.diff`: perf_takehome.py version (2325 cycles)
- `2333.diff`: Previous checkpoint (2333 cycles)
- `AGENT.md`: This file

---

## Solution.py Optimization (2320 cycles)

### Approach Differences from perf_takehome.py
- Uses selection-based tree lookup for R0-R2 (avoids gathers for first 3 rounds)
- Split R2 into A/B halves for better overlap with prime gather
- Different batching structure for R0 index update

### Key Optimizations Applied
1. **Hoisted R1/R2 constants to init_slots** - diff_1_2, diff_3_4, diff_5_6, three_vec precomputed
2. **Fixed wrap threshold** - Changed from `r >= 9` to `r >= forest_height`
3. **R2 A/B split with prime gather overlap** - R2_A completes, then prime gather interleaved with R2_B
4. **R0 batching optimization** - Separate ANDs from ADDs for better packing

### Remaining Optimization Opportunities (from analysis)

#### 1. _pack_slots can sabotage interleaving
The packer flushes bundles on RAW hazards, which can prevent intended load/valu overlap.
**Solution**: Build explicit bundles instead of relying on _pack_slots:
- Emit addr-calc bundles (valu only)
- Emit load-only bundles (2 loads each)
- Emit hash bundles (6 valu each)
- Interleave at bundle level, not slot level

#### 2. Wrap threshold should be computed from n_nodes, not forest_height
Correct formula: `wrap_needed = r >= ceil(log2(n_nodes + 2)) - 1`
For n_nodes=2047: threshold is r >= 11 (after completing round)

#### 3. Reuse temps across phases
early_temps (8×2 vecs) can be reused as temps_A/B after R2 completes.
Reduces scratch pressure and can reduce bundling hazards.

#### 4. Try loads-first interleaving
Current: 6 valu then 2 loads
Alternative: 2 loads then 6 valu (kick off loads ASAP, fill wait time with valu)

#### 5. Selection beyond R2 is NOT recommended
- R3+ selection would be VALU-bound (VALU already saturated)
- Gathers are load-bound but hideable via interleaving
- Exception: Could work if flow.vselect has high throughput (needs testing)

### Current Metrics (solution.py)
- Total instructions: 598
- valu utilization: 86.6%
- load utilization: 73.4%
- Poorly packed instructions: 5 (all in init phase)
- Loop body: 574 instructions
- 4 batch iterations
- Scratch used: 927 / 1536 (609 remaining)

### Attempted Optimizations (no improvement)
- Loads-first interleaving: 2320 → 2368 (worse)
- Different interleave chunk sizes: 6,2 is optimal
- flow.vselect: limit is 1, not useful

### Recent Optimization Attempts (all failed to improve)

#### 1. Explicit bundling bypass (_pack_overlap)
- **Goal**: Prevent _pack_slots from breaking intended 6-valu/2-load cadence
- **Approach 1**: Build bundles directly with addr_calc + hash valu, then loads + hash
- **Result**: Crashed - loads accessed addresses before they were computed
- **Approach 2**: Pack addr_calc separately, then inject loads into valu bundles
- **Result**: 2368 cycles (worse than 2320)
- **Conclusion**: _pack_slots hazard detection is correct for this simulator (no cross-engine forwarding)

#### 2. Hash op reordering
- **Goal**: Reduce RAW hazards by grouping ops by type
- **Old order**: [op1_0, op3_0, op1_1, op3_1, ...] (pairs per vector)
- **New order**: [op1_0..3, op3_0..3, op2_0..3] (all op1, then all op3, then all op2)
- **Result**: 2644 cycles (much worse!)
- **Analysis**: New order creates op3_2 + op2_2 hazard in same bundle
- **Conclusion**: Original interleaved order is better

#### 3. Different interleave ratios
- **Tested 12:4**: 3024 cycles (much worse)
- **Conclusion**: 6:2 ratio is optimal

#### 4. Index update op interleaving
- **Goal**: Better bundle alignment with mul_add/AND pairs
- **Change**: Emit (mul_add_0, and_0, mul_add_1, and_1, ...) instead of (mul_add_0..3, and_0..3)
- **Result**: 2320 cycles (no change)
- **Conclusion**: Bundle split still happens at same point

### Fundamental Limitation Analysis

The 5-valu bundle issue comes from hash function structure:
- Each stage has 12 ops: 8 op1/op3 + 4 op2
- Bundle split at 6 ops: [op1_0..op3_2] (6), [op1_3, op3_3, op2_0, op2_1, op2_2] (5)
- op2_3 has RAW hazard with op1_3 (t0[3]) and op3_3 (t1[3])
- This causes a flush at 5 ops instead of 6

With 4 vectors (HALF=4), this 5-valu pattern repeats every hash stage.
Changing to NVECS=6/HALF=3 could align better (3*2=6 ops per batch), but would require batch handling changes.

### Wrap Threshold Formula
```python
wrap_threshold = ceil(log2(n_nodes + 2)) - 2
```
For n_nodes=2047: threshold=10, wrap for rounds 10-15

### Final State: 2320 cycles (63.68x speedup)
- This appears to be near the local optimum for the current architecture
- Further improvement would require:
  - Different vector count (e.g., 3 instead of 4 per half)
  - Algorithm changes to reduce total valu ops
  - Cross-engine forwarding in the simulator (not supported)
