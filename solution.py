"""
# Anthropic's Original Performance Engineering Take-home (Release version)

RADICALLY DIFFERENT APPROACH: Extend selection-based tree lookup to R0-R5
to minimize expensive gather operations.

Key insight: For rounds 0-5, indices are bounded to small ranges:
- R0: idx=0 (1 node)
- R1: idx∈{1,2} (2 nodes)
- R2: idx∈{3-6} (4 nodes)
- R3: idx∈{7-14} (8 nodes)
- R4: idx∈{15-30} (16 nodes)
- R5: idx∈{31-62} (32 nodes)

We preload these tree values and use binary selection to pick the right one,
trading valu ops for expensive gather loads.
"""

from collections import defaultdict
import random
from problem import (
    Engine,
    DebugInfo,
    HASH_STAGES,
    reference_kernel2,
    Machine,
    Input,
    Tree,
    SLOT_LIMITS,
    VLEN,
    SCRATCH_SIZE,
    build_mem_image,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_ptr = 0
        self.scratch_debug = {}
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr}"
        return addr

    def scratch_const(self, val, name=None, slots=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(f"c_{val}" if name is None else name)
            if slots is None:
                self.add("load", ("const", addr, val))
            else:
                slots.append(("load", ("const", addr, val)))
            self.const_map[val] = addr
        return self.const_map[val]

    def _slot_rw(self, engine, slot):
        reads = set()
        writes = set()
        if engine == "alu":
            if len(slot) == 4:
                _op, dest, a1, a2 = slot
                reads.update([a1, a2])
                writes.add(dest)
            elif len(slot) == 3:
                _op, dest, a1 = slot
                reads.add(a1)
                writes.add(dest)
        elif engine == "valu":
            match slot:
                case ("vbroadcast", dest, src):
                    reads.add(src)
                    for i in range(VLEN): writes.add(dest + i)
                case ("multiply_add", dest, a, b, c):
                    for i in range(VLEN):
                        reads.add(a + i)
                        reads.add(b + i)
                        reads.add(c + i)
                        writes.add(dest + i)
                case (_op, dest, a1, a2):
                    for i in range(VLEN):
                        reads.add(a1 + i)
                        reads.add(a2 + i)
                        writes.add(dest + i)
        elif engine == "load":
            match slot:
                case ("load", dest, addr):
                    reads.add(addr)
                    writes.add(dest)
                case ("load_offset", dest, addr, offset):
                    reads.add(addr + offset)
                    writes.add(dest + offset)
                case ("vload", dest, addr):
                    reads.add(addr)
                    for i in range(VLEN): writes.add(dest + i)
                case ("const", dest, _val):
                    writes.add(dest)
        elif engine == "store":
            match slot:
                case ("store", addr, src):
                    reads.update([addr, src])
                case ("vstore", addr, src):
                    reads.add(addr)
                    for i in range(VLEN): reads.add(src + i)
        elif engine == "flow":
            match slot:
                case ("select", dest, cond, a, b):
                    reads.update([cond, a, b])
                    writes.add(dest)
                case ("add_imm", dest, a, _imm):
                    reads.add(a)
                    writes.add(dest)
                case ("vselect", dest, cond, a, b):
                    for i in range(VLEN):
                        reads.add(cond + i)
                        reads.add(a + i)
                        reads.add(b + i)
                        writes.add(dest + i)
                case ("cond_jump", cond, addr):
                    reads.update([cond, addr])
                case ("cond_jump_rel", cond, _offset):
                    reads.add(cond)
                case ("jump", addr):
                    reads.add(addr)
                case ("jump_indirect", addr):
                    reads.add(addr)
                case ("coreid", dest):
                    writes.add(dest)
                case ("trace_write", val):
                    reads.add(val)
                case ("pause",):
                    pass
                case ("halt",):
                    pass
        return reads, writes

    def _pack_slots(self, slots: list[tuple[Engine, tuple]]):
        instrs = []
        bundle = {}
        bundle_writes = set()
        engine_counts = defaultdict(int)

        def flush_bundle():
            nonlocal bundle, bundle_writes, engine_counts
            if bundle:
                instrs.append(bundle)
                bundle = {}
                bundle_writes = set()
                engine_counts = defaultdict(int)

        for engine, slot in slots:
            reads, writes = self._slot_rw(engine, slot)
            if (
                engine_counts[engine] >= SLOT_LIMITS[engine]
                or reads & bundle_writes
                or writes & bundle_writes
            ):
                flush_bundle()
            bundle.setdefault(engine, []).append(slot)
            engine_counts[engine] += 1
            bundle_writes.update(writes)

        flush_bundle()
        return instrs

    def _pack_overlap(self, addr_ops, load_ops, valu_ops):
        """Build bundles with guaranteed load/valu overlap.

        Phase 1: Emit addr_ops bundles (must complete before loads)
        Phase 2: Pack valu_ops respecting hazards, inject loads into bundles

        This ensures addr calc completes before loads while respecting
        intra-valu hazards and achieving load/valu overlap.
        """
        instrs = []

        # Phase 1: emit addr calc using _pack_slots (respects hazards)
        if addr_ops:
            instrs.extend(self._pack_slots(addr_ops))

        # Phase 2: pack valu_ops respecting hazards
        valu_bundles = self._pack_slots(valu_ops) if valu_ops else []

        # Inject loads into valu bundles (up to 2 per bundle)
        li = 0
        for bundle in valu_bundles:
            loads_to_add = []
            while len(loads_to_add) < 2 and li < len(load_ops):
                loads_to_add.append(load_ops[li][1])
                li += 1
            if loads_to_add:
                bundle["load"] = loads_to_add

        # Add valu bundles (now with loads injected)
        instrs.extend(valu_bundles)

        # If more loads remain, add bundles for them (2 loads per bundle)
        while li < len(load_ops):
            bundle = {"load": []}
            while len(bundle["load"]) < 2 and li < len(load_ops):
                bundle["load"].append(load_ops[li][1])
                li += 1
            instrs.append(bundle)

        return instrs

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        init_slots = []
        NVECS = 8  # Process 8 vectors at a time (64 elements) - limited by scratch

        # Constants
        zero_const = self.scratch_const(0, "zero", init_slots)
        one_const = self.scratch_const(1, "one", init_slots)
        two_const = self.scratch_const(2, "two", init_slots)
        vlen_const = self.scratch_const(VLEN, "vlen", init_slots)
        vlen_nvecs_const = self.scratch_const(VLEN * NVECS, "vlen_nvecs", init_slots)
        n_nodes_const = self.scratch_const(n_nodes, "n_nodes", init_slots)
        batch_end = self.scratch_const(batch_size, "batch_end", init_slots)

        forest_values_p = self.scratch_const(7, "forest_values_p", init_slots)
        inp_indices_p = self.scratch_const(7 + n_nodes, "inp_indices_p", init_slots)
        inp_values_p = self.scratch_const(7 + n_nodes + batch_size, "inp_values_p", init_slots)

        batch_counter = self.alloc_scratch("batch_counter")

        # Vectors
        idx_vecs = [self.alloc_scratch(f"idx_vec{i}", VLEN) for i in range(NVECS)]
        val_vecs = [self.alloc_scratch(f"val_vec{i}", VLEN) for i in range(NVECS)]
        addr_vecs = [self.alloc_scratch(f"addr_vec{i}", VLEN) for i in range(NVECS)]
        node_vecs = [self.alloc_scratch(f"node_vec{i}", VLEN) for i in range(NVECS)]

        # Constant vectors
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        forest_base_vec = self.alloc_scratch("forest_base_vec", VLEN)

        init_slots.append(("valu", ("vbroadcast", one_vec, one_const)))
        init_slots.append(("valu", ("vbroadcast", two_vec, two_const)))
        init_slots.append(("valu", ("vbroadcast", n_nodes_vec, n_nodes_const)))
        init_slots.append(("valu", ("vbroadcast", forest_base_vec, forest_values_p)))

        # Hash constant vectors (precompute all 6 stages)
        # Some stages can use multiply_add: when op1=='+' and op2=='+' and op3=='<<'
        # val = val * (1 + (1 << val3)) + val1
        hash_const_vecs = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.scratch_const(val1, slots=init_slots)
            v1 = self.alloc_scratch(length=VLEN)
            init_slots.append(("valu", ("vbroadcast", v1, c1)))

            if op1 == '+' and op2 == '+' and op3 == '<<':
                # Can use single multiply_add: val = val * mul + c1
                mul = (1 + (1 << val3)) % (2**32)
                mul_c = self.scratch_const(mul, slots=init_slots)
                mul_v = self.alloc_scratch(length=VLEN)
                init_slots.append(("valu", ("vbroadcast", mul_v, mul_c)))
                hash_const_vecs.append(('multiply_add', None, None, v1, mul_v))
            else:
                c3 = self.scratch_const(val3, slots=init_slots)
                v3 = self.alloc_scratch(length=VLEN)
                init_slots.append(("valu", ("vbroadcast", v3, c3)))
                hash_const_vecs.append((op1, op2, op3, v1, v3))

        # Preload tree nodes for R0-R2 (7 nodes total: tree[0] to tree[6])
        # This avoids expensive gathers for the first 3 rounds!
        tree_scalars = [self.alloc_scratch(f"tree_{i}") for i in range(7)]
        tree_vecs = [self.alloc_scratch(f"tree_v{i}", VLEN) for i in range(7)]

        # Load tree[0:7] and broadcast to vectors
        for i in range(7):
            addr_const = self.scratch_const(7 + i, slots=init_slots)  # forest starts at offset 7
            init_slots.append(("load", ("load", tree_scalars[i], addr_const)))
        for i in range(7):
            init_slots.append(("valu", ("vbroadcast", tree_vecs[i], tree_scalars[i])))

        # Precompute R1/R2 diffs (constant across all batches)
        diff_1_2 = self.alloc_scratch("diff_1_2", VLEN)
        diff_3_4 = self.alloc_scratch("diff_3_4", VLEN)
        diff_5_6 = self.alloc_scratch("diff_5_6", VLEN)
        three_vec = self.alloc_scratch("three_vec", VLEN)
        three_const = self.scratch_const(3, "three", init_slots)
        init_slots.append(("valu", ("vbroadcast", three_vec, three_const)))
        # These depend on tree_vecs which are broadcast above, so order is correct
        init_slots.append(("valu", ("-", diff_1_2, tree_vecs[2], tree_vecs[1])))
        init_slots.append(("valu", ("-", diff_3_4, tree_vecs[4], tree_vecs[3])))
        init_slots.append(("valu", ("-", diff_5_6, tree_vecs[6], tree_vecs[5])))

        # Pointer offset constants
        ptr_offsets = [self.scratch_const(i * VLEN, slots=init_slots) for i in range(NVECS)]
        idx_base_offsets = [self.alloc_scratch(f"idx_base_{i}") for i in range(NVECS)]
        val_base_offsets = [self.alloc_scratch(f"val_base_{i}") for i in range(NVECS)]
        for i in range(NVECS):
            init_slots.append(("alu", ("+", idx_base_offsets[i], inp_indices_p, ptr_offsets[i])))
            init_slots.append(("alu", ("+", val_base_offsets[i], inp_values_p, ptr_offsets[i])))

        self.instrs.extend(self._pack_slots(init_slots))
        print(f"After init: {len(self.instrs)} bundles")
        self.add("load", ("const", batch_counter, 0))

        # BATCH LOOP
        batch_loop_start = len(self.instrs)

        ptrs_i = [self.alloc_scratch() for _ in range(NVECS)]
        ptrs_v = [self.alloc_scratch() for _ in range(NVECS)]

        # Setup pointers
        setup_ptrs = []
        for i in range(NVECS):
            setup_ptrs.append(("alu", ("+", ptrs_i[i], idx_base_offsets[i], batch_counter)))
            setup_ptrs.append(("alu", ("+", ptrs_v[i], val_base_offsets[i], batch_counter)))
        self.instrs.extend(self._pack_slots(setup_ptrs))

        # Load initial values (indices start at 0, so we just load values)
        load_ops = []
        for i in range(NVECS):
            load_ops.append(("load", ("vload", val_vecs[i], ptrs_v[i])))
        self.instrs.extend(self._pack_slots(load_ops))

        # Per-vector temps for R0-R2 (8 vectors total, 2 temps each)
        early_temps = [[self.alloc_scratch(f"et_{v}_{i}", VLEN) for i in range(2)] for v in range(NVECS)]

        # Helper functions
        def gen_hash(vec_subset):
            """Hash all vectors using per-vector temps for better packing.

            Reorders 3-op stages to emit all op1, then all op3, then all op2
            to avoid RAW hazards (op2 reads from t0/t1 written by op1/op3).
            """
            ops = []
            for op1, op2, op3, c1, c3_or_mul in hash_const_vecs:
                if op1 == 'multiply_add':
                    # Optimized: single multiply_add per vector
                    for i, val in enumerate(vec_subset):
                        ops.append(("valu", ("multiply_add", val, val, c3_or_mul, c1)))
                else:
                    # Reordered: all op1, then all op3, then all op2
                    for i, val in enumerate(vec_subset):
                        t0, _ = early_temps[i]
                        ops.append(("valu", (op1, t0, val, c1)))
                    for i, val in enumerate(vec_subset):
                        _, t1 = early_temps[i]
                        ops.append(("valu", (op3, t1, val, c3_or_mul)))
                    for i, val in enumerate(vec_subset):
                        t0, t1 = early_temps[i]
                        ops.append(("valu", (op2, val, t0, t1)))
            return ops

        def gen_index_update(idx_subset, val_subset, with_wrap=True):
            """Update indices using per-vector temps for better packing."""
            ops = []
            # Batch multiply_add (independent)
            for i in range(len(idx_subset)):
                t0, _ = early_temps[i]
                ops.append(("valu", ("multiply_add", t0, idx_subset[i], two_vec, one_vec)))
            # Batch AND (independent)
            for i in range(len(idx_subset)):
                _, t1 = early_temps[i]
                ops.append(("valu", ("&", t1, val_subset[i], one_vec)))
            # Batch ADD (depends on above)
            for i in range(len(idx_subset)):
                t0, t1 = early_temps[i]
                ops.append(("valu", ("+", idx_subset[i], t0, t1)))
            if with_wrap:
                # Batch compare
                for i in range(len(idx_subset)):
                    t0, _ = early_temps[i]
                    ops.append(("valu", ("<", t0, idx_subset[i], n_nodes_vec)))
                # Batch multiply
                for i in range(len(idx_subset)):
                    t0, _ = early_temps[i]
                    ops.append(("valu", ("*", idx_subset[i], idx_subset[i], t0)))
            return ops

        def gen_xor(val_subset, node_subset):
            ops = []
            for i in range(len(val_subset)):
                ops.append(("valu", ("^", val_subset[i], val_subset[i], node_subset[i])))
            return ops

        def gen_gather(idx_subset, addr_subset, node_subset):
            ops = []
            for i in range(len(idx_subset)):
                ops.append(("valu", ("+", addr_subset[i], idx_subset[i], forest_base_vec)))
            for i in range(len(idx_subset)):
                for vi in range(VLEN):
                    ops.append(("load", ("load_offset", node_subset[i], addr_subset[i], vi)))
            return ops

        # ============ ROUND 0: All indices are 0, use tree[0] ============
        r0_ops = []
        for v in range(NVECS):
            r0_ops.append(("valu", ("^", val_vecs[v], val_vecs[v], tree_vecs[0])))
        r0_ops.extend(gen_hash(val_vecs))
        # R0 idx update: idx = 0*2 + 1 + (val & 1) = 1 + (val & 1)
        # Batch ANDs first (all independent)
        for v in range(NVECS):
            r0_ops.append(("valu", ("&", idx_vecs[v], val_vecs[v], one_vec)))
        # Then batch ADDs (each depends on its own AND, but independent of other vectors)
        for v in range(NVECS):
            r0_ops.append(("valu", ("+", idx_vecs[v], idx_vecs[v], one_vec)))
        r0_packed = self._pack_slots(r0_ops)
        # Analyze r0 bundles
        r0_valu_dist = {}
        for b in r0_packed:
            vc = len(b.get("valu", []))
            r0_valu_dist[vc] = r0_valu_dist.get(vc, 0) + 1
        print(f"R0: {len(r0_packed)} bundles, valu dist: {dict(sorted(r0_valu_dist.items()))}")
        self.instrs.extend(r0_packed)
        print(f"After R0: {len(self.instrs)} bundles")

        # ============ ROUND 1: idx ∈ {1,2}, use tree[1] and tree[2] ============
        # node = tree[1] + (tree[2] - tree[1]) * (idx - 1)
        # diff_1_2 precomputed in init_slots
        r1_ops = []
        # Batch subtract ops (independent with per-vector temps)
        for v in range(NVECS):
            t0, _ = early_temps[v]
            r1_ops.append(("valu", ("-", t0, idx_vecs[v], one_vec)))  # sel = idx - 1
        # Batch multiply_add ops (depends on above)
        for v in range(NVECS):
            t0, _ = early_temps[v]
            r1_ops.append(("valu", ("multiply_add", node_vecs[v], diff_1_2, t0, tree_vecs[1])))
        r1_ops.extend(gen_xor(val_vecs, node_vecs))
        r1_ops.extend(gen_hash(val_vecs))
        r1_ops.extend(gen_index_update(idx_vecs, val_vecs, with_wrap=False))
        self.instrs.extend(self._pack_slots(r1_ops))
        print(f"After R1: {len(self.instrs)} bundles")

        # ============ ROUNDS 3-15: Use gathers with A/B pipelining ============
        # Split into two groups for pipelining
        HALF = NVECS // 2
        idx_A, idx_B = idx_vecs[:HALF], idx_vecs[HALF:]
        val_A, val_B = val_vecs[:HALF], val_vecs[HALF:]
        addr_A, addr_B = addr_vecs[:HALF], addr_vecs[HALF:]
        node_A, node_B = node_vecs[:HALF], node_vecs[HALF:]
        early_temps_A = early_temps[:HALF]
        early_temps_B = early_temps[HALF:]

        # Reuse early_temps for the main loop (R0-R2 done, temps free)
        # This reduces scratch pressure and can reduce bundling hazards
        temps_A = early_temps_A
        temps_B = early_temps_B

        # ============ ROUND 2: idx ∈ {3,4,5,6}, use tree[3:7] ============
        # Binary selection: 2 levels using per-vector temps
        # diff_3_4, diff_5_6, three_vec precomputed in init_slots
        # Per-vector scratch for selection (need 4 temps per vector: offset, bit0, bit1, pair results)
        r2_temps = [[self.alloc_scratch(f"r2t_{v}_{i}", VLEN) for i in range(4)] for v in range(NVECS)]
        r2_temps_A = r2_temps[:HALF]
        r2_temps_B = r2_temps[HALF:]

        def gen_r2_selection(vec_indices, vec_nodes, vec_temps, vec_early):
            """Generate R2 selection ops for a subset of vectors."""
            ops = []
            # Batch: compute offset (independent)
            for i, (idx, temps) in enumerate(zip(vec_indices, vec_temps)):
                offset = temps[0]
                ops.append(("valu", ("-", offset, idx, three_vec)))
            # Batch: compute bit0 and bit1 (independent)
            for temps in vec_temps:
                offset, bit0, bit1, _ = temps
                ops.append(("valu", ("&", bit0, offset, one_vec)))
                ops.append(("valu", (">>", bit1, offset, one_vec)))
            # Batch: compute pair0 and pair1 (depend on bit0)
            for i, (temps, early) in enumerate(zip(vec_temps, vec_early)):
                _, bit0, _, pair = temps
                ops.append(("valu", ("multiply_add", pair, diff_3_4, bit0, tree_vecs[3])))
            for i, (temps, early) in enumerate(zip(vec_temps, vec_early)):
                _, bit0, _, _ = temps
                t0, _ = early
                ops.append(("valu", ("multiply_add", t0, diff_5_6, bit0, tree_vecs[5])))
            # Batch: compute diff and final node
            for i, (temps, early, node) in enumerate(zip(vec_temps, vec_early, vec_nodes)):
                _, _, bit1, pair = temps
                t0, t1 = early
                ops.append(("valu", ("-", t1, t0, pair)))
            for i, (temps, early, node) in enumerate(zip(vec_temps, vec_early, vec_nodes)):
                _, _, bit1, pair = temps
                _, t1 = early
                ops.append(("valu", ("multiply_add", node, t1, bit1, pair)))
            return ops

        def gen_xor_half(val_subset, node_subset):
            ops = []
            for val, node in zip(val_subset, node_subset):
                ops.append(("valu", ("^", val, val, node)))
            return ops

        def gen_hash_half(vec_subset, vec_temps):
            """Hash using per-vector temps for better packing.

            Reorders 3-op stages to avoid RAW hazards.
            """
            ops = []
            for op1, op2, op3, c1, c3_or_mul in hash_const_vecs:
                if op1 == 'multiply_add':
                    for val, temps in zip(vec_subset, vec_temps):
                        ops.append(("valu", ("multiply_add", val, val, c3_or_mul, c1)))
                else:
                    # Reordered: all op1, then all op3, then all op2
                    for val, temps in zip(vec_subset, vec_temps):
                        t0, _ = temps
                        ops.append(("valu", (op1, t0, val, c1)))
                    for val, temps in zip(vec_subset, vec_temps):
                        _, t1 = temps
                        ops.append(("valu", (op3, t1, val, c3_or_mul)))
                    for val, temps in zip(vec_subset, vec_temps):
                        t0, t1 = temps
                        ops.append(("valu", (op2, val, t0, t1)))
            return ops

        def gen_index_update_half(idx_subset, val_subset, vec_temps, with_wrap=True):
            """Index update using per-vector temps for better packing."""
            ops = []
            for idx, temps in zip(idx_subset, vec_temps):
                t0, _ = temps
                ops.append(("valu", ("multiply_add", t0, idx, two_vec, one_vec)))
            for val, temps in zip(val_subset, vec_temps):
                _, t1 = temps
                ops.append(("valu", ("&", t1, val, one_vec)))
            for idx, temps in zip(idx_subset, vec_temps):
                t0, t1 = temps
                ops.append(("valu", ("+", idx, t0, t1)))
            if with_wrap:
                for idx, temps in zip(idx_subset, vec_temps):
                    t0, _ = temps
                    ops.append(("valu", ("<", t0, idx, n_nodes_vec)))
                for idx, temps in zip(idx_subset, vec_temps):
                    t0, _ = temps
                    ops.append(("valu", ("*", idx, idx, t0)))
            return ops

        # R2 for A half
        r2_A_ops = gen_r2_selection(idx_A, node_A, r2_temps_A, early_temps_A)
        r2_A_ops.extend(gen_xor_half(val_A, node_A))
        r2_A_ops.extend(gen_hash_half(val_A, early_temps_A))
        r2_A_ops.extend(gen_index_update_half(idx_A, val_A, early_temps_A, with_wrap=False))
        self.instrs.extend(self._pack_slots(r2_A_ops))
        print(f"After R2-A: {len(self.instrs)} bundles")

        def gen_hash_group(vec_subset, group_temps):
            """Hash using per-vector temps for better packing.

            Reorders 3-op stages to avoid RAW hazards.
            """
            ops = []
            for op1, op2, op3, c1, c3_or_mul in hash_const_vecs:
                if op1 == 'multiply_add':
                    for i, val in enumerate(vec_subset):
                        ops.append(("valu", ("multiply_add", val, val, c3_or_mul, c1)))
                else:
                    # Reordered: all op1, then all op3, then all op2
                    for i, val in enumerate(vec_subset):
                        t0, _ = group_temps[i]
                        ops.append(("valu", (op1, t0, val, c1)))
                    for i, val in enumerate(vec_subset):
                        _, t1 = group_temps[i]
                        ops.append(("valu", (op3, t1, val, c3_or_mul)))
                    for i, val in enumerate(vec_subset):
                        t0, t1 = group_temps[i]
                        ops.append(("valu", (op2, val, t0, t1)))
            return ops

        def gen_index_update_group(idx_subset, val_subset, group_temps, with_wrap=True):
            """Index update using per-vector temps for better packing."""
            ops = []
            # Batch multiply_add ops (independent)
            for i in range(len(idx_subset)):
                t0, _ = group_temps[i]
                ops.append(("valu", ("multiply_add", t0, idx_subset[i], two_vec, one_vec)))
            # Batch AND ops (independent)
            for i in range(len(idx_subset)):
                _, t1 = group_temps[i]
                ops.append(("valu", ("&", t1, val_subset[i], one_vec)))
            # Batch ADD ops (depends on above)
            for i in range(len(idx_subset)):
                t0, t1 = group_temps[i]
                ops.append(("valu", ("+", idx_subset[i], t0, t1)))
            if with_wrap:
                # Batch compare ops (independent)
                for i in range(len(idx_subset)):
                    t0, _ = group_temps[i]
                    ops.append(("valu", ("<", t0, idx_subset[i], n_nodes_vec)))
                # Batch multiply ops (depends on above)
                for i in range(len(idx_subset)):
                    t0, _ = group_temps[i]
                    ops.append(("valu", ("*", idx_subset[i], idx_subset[i], t0)))
            return ops

        def gen_gather_group(idx_subset, addr_subset, node_subset):
            ops = []
            for i in range(len(idx_subset)):
                ops.append(("valu", ("+", addr_subset[i], idx_subset[i], forest_base_vec)))
            for i in range(len(idx_subset)):
                for vi in range(VLEN):
                    ops.append(("load", ("load_offset", node_subset[i], addr_subset[i], vi)))
            return ops

        def interleave_ops(load_ops, valu_ops, valu_chunk=6, load_chunk=2):
            """Interleave load and valu ops for better packing.
            Default ratio: 2 loads per 6 valu (matches slot limits)."""
            result = []
            li, vi = 0, 0
            while li < len(load_ops) or vi < len(valu_ops):
                # Emit valu ops
                for _ in range(valu_chunk):
                    if vi < len(valu_ops):
                        result.append(valu_ops[vi])
                        vi += 1
                # Then loads
                for _ in range(load_chunk):
                    if li < len(load_ops):
                        result.append(load_ops[li])
                        li += 1
            return result

        # R2 for B half + Prime gather A (overlapped)
        r2_B_ops = gen_r2_selection(idx_B, node_B, r2_temps_B, early_temps_B)
        r2_B_ops.extend(gen_xor_half(val_B, node_B))
        r2_B_ops.extend(gen_hash_half(val_B, early_temps_B))
        r2_B_ops.extend(gen_index_update_half(idx_B, val_B, early_temps_B, with_wrap=False))

        prime_ops = gen_gather_group(idx_A, addr_A, node_A)
        prime_addr = prime_ops[:HALF]  # addr calc (valu)
        prime_loads = prime_ops[HALF:]  # loads

        # Overlap prime gather loads with R2 B-half valu work
        prime_combined = prime_addr + interleave_ops(prime_loads, r2_B_ops)
        self.instrs.extend(self._pack_slots(prime_combined))

        # Wrap is needed when max possible idx after round r can exceed n_nodes-1
        # Max idx after round r is 2^(r+2) - 2 (starting from idx=0 at r=0)
        # Wrap needed when 2^(r+2) - 2 >= n_nodes
        # For n_nodes=2047: wrap needed at r >= 10 (2^12-2 = 4094 > 2047)
        import math
        wrap_threshold = max(3, math.ceil(math.log2(n_nodes + 2)) - 2)

        for r in range(3, rounds):
            needs_wrap = (r >= wrap_threshold)
            before_r = len(self.instrs)

            # Step 1: Hash A (from previous gather) | Gather B
            hash_A_ops = gen_xor(val_A, node_A)
            hash_A_ops.extend(gen_hash_group(val_A, temps_A))
            hash_A_ops.extend(gen_index_update_group(idx_A, val_A, temps_A, with_wrap=needs_wrap))

            gather_B_ops = gen_gather_group(idx_B, addr_B, node_B)
            # Separate addr calc (valu) from loads
            gather_B_addr = gather_B_ops[:HALF]  # HALF valu for addr calc
            gather_B_loads = gather_B_ops[HALF:]  # HALF*VLEN loads

            step1_ops = gather_B_addr + interleave_ops(gather_B_loads, hash_A_ops)
            step1_packed = self._pack_slots(step1_ops)
            self.instrs.extend(step1_packed)

            # Step 2: Hash B | Gather A (for next round, unless last round)
            hash_B_ops = gen_xor(val_B, node_B)
            hash_B_ops.extend(gen_hash_group(val_B, temps_B))
            hash_B_ops.extend(gen_index_update_group(idx_B, val_B, temps_B, with_wrap=needs_wrap))

            if r + 1 < rounds:
                gather_A_ops = gen_gather_group(idx_A, addr_A, node_A)
                gather_A_addr = gather_A_ops[:HALF]
                gather_A_loads = gather_A_ops[HALF:]
                step2_ops = gather_A_addr + interleave_ops(gather_A_loads, hash_B_ops)
            else:
                # Last round: no more gathers, just hash B
                step2_ops = hash_B_ops

            step2_packed = self._pack_slots(step2_ops)
            self.instrs.extend(step2_packed)
            if r == 3:
                print(f"R{r}: step1={len(step1_packed)}, step2={len(step2_packed)}, total={len(self.instrs)-before_r}")

        # Store results and loop
        store_ops = []
        for i in range(NVECS):
            store_ops.append(("store", ("vstore", ptrs_i[i], idx_vecs[i])))
            store_ops.append(("store", ("vstore", ptrs_v[i], val_vecs[i])))

        jump_cond = self.alloc_scratch("jump_cond")
        store_ops.append(("alu", ("+", batch_counter, batch_counter, vlen_nvecs_const)))
        store_ops.append(("alu", ("<", jump_cond, batch_counter, batch_end)))
        self.instrs.extend(self._pack_slots(store_ops))

        jump_src = len(self.instrs) + 1
        offset = batch_loop_start - jump_src
        self.add("flow", ("cond_jump_rel", jump_cond, offset))
        self.add("flow", ("pause",))
        print(f"Total kernel bundles: {len(self.instrs)}")


BASELINE = 147734

def do_kernel_test():
    forest_height = 10
    batch_size = 256  # Full test size
    rounds = 16

    random.seed(245)

    forest = Tree.generate(forest_height)
    n_nodes = len(forest.values)
    input_data = Input.generate(forest, batch_size, rounds)

    # Run Reference
    mem_ref = build_mem_image(forest, input_data)
    trace = {}
    ref_gen = reference_kernel2(mem_ref, trace)
    for _ in ref_gen: pass

    # Build Kernel
    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)

    # Run Machine
    mem_machine = build_mem_image(forest, input_data)
    machine = Machine(mem_machine, kb.instrs, kb.debug_info())
    machine.run()

    # Verify
    inp_indices_p = 7 + n_nodes
    inp_values_p = inp_indices_p + batch_size

    ref_indices = mem_ref[inp_indices_p : inp_indices_p + batch_size]
    ref_values = mem_ref[inp_values_p : inp_values_p + batch_size]

    dut_indices = machine.mem[inp_indices_p : inp_indices_p + batch_size]
    dut_values = machine.mem[inp_values_p : inp_values_p + batch_size]

    if ref_indices != dut_indices:
        print("Indices mismatch!")
        for i in range(batch_size):
            if ref_indices[i] != dut_indices[i]:
                print(f"Index[{i}]: Ref {ref_indices[i]} != Dut {dut_indices[i]}")
                break

    if ref_values != dut_values:
        print("Values mismatch!")
        for i in range(batch_size):
            if ref_values[i] != dut_values[i]:
                print(f"Value[{i}]: Ref {ref_values[i]} != Dut {dut_values[i]}")
                break

    if ref_indices == dut_indices and ref_values == dut_values:
        print("Verification PASSED!")
    else:
        print("Verification FAILED!")

    print("CYCLES: ", machine.cycle)
    print("SPEEDUP: {:.2f}x".format(BASELINE / machine.cycle))

if __name__ == "__main__":
    do_kernel_test()
