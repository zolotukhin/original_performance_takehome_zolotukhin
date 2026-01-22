# Anthropic's Original Performance Take-Home

## Current Result: 1307 cycles (113x speedup)

**Optimization journey: 147734 → 1307 cycles**

| Stage | Cycles | Speedup | Key Technique |
|-------|--------|---------|---------------|
| Baseline | 147734 | 1.0x | Scalar reference implementation |
| Vectorization | 65586 | 2.3x | VLEN=8 vector operations |
| VLIW packing | 17465 | 8.5x | Multi-slot instruction bundling |
| Loop inversion | 4862 | 30x | Batch-outer/round-inner, data stays in registers |
| Software pipelining | 2625 | 56x | Overlap gather latency with hash computation |
| Linear interpolation | 2325 | 64x | Replace gathers for levels 0-2 with math |
| Static scheduler | 1338 | 110x | Flat-list generation with vselect for levels 0-3 |
| **Final tuning** | **1307** | **113x** | Hardcoded params, skip index stores, ALU add |

Key insights:
- **vselect for tree traversal**: Levels 0-3 use preloaded nodes + vselect instead of gathers
- **Static scheduling**: Generate all ops upfront, let scheduler pack across blocks/levels
- **VALU is the bottleneck**: 7267 ops / 6 per cycle = 1211 min cycles (current: 1307)

Files: `perf_takehome.py` (1307 cycles), `AGENT.md` (detailed optimization log)

---

This repo contains a version of Anthropic's original performance take-home, before Claude Opus 4.5 started doing better than humans given only 2 hours.

Now you can try to beat Claude Opus 4.5 given unlimited time!

## Performance benchmarks 

measured in clock cycles from the simulated machine:

- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness

If you optimize below 1487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com with your code (and ideally a resume) so we can be appropriately impressed and perhaps discuss interviewing.

Run `python tests/submission_tests.py` to see which thresholds you pass.
