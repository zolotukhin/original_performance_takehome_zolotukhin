from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.enable_vdebug = False

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_packed(self, bundle):
        """Add a packed instruction bundle. Bundle is dict {engine: [slots]}."""
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        # OPTIMIZED: Use different temp addresses to pack parameter loads
        # Pairs: (rounds,0), (n_nodes,1), (batch_size,2), (forest_height,3), (forest_values_p,4), (inp_indices_p,5), (inp_values_p,6)
        tmp_addrs = [tmp1, tmp2, tmp1, tmp2, tmp1, tmp2, tmp1]  # Alternate between tmp1 and tmp2
        for i in range(0, len(init_vars), 2):
            if i + 1 < len(init_vars):
                # Pack 2 const loads
                self.add_packed({"load": [("const", tmp_addrs[i], i), ("const", tmp_addrs[i+1], i+1)]})
                # Pack 2 memory loads
                self.add_packed({"load": [
                    ("load", self.scratch[init_vars[i]], tmp_addrs[i]),
                    ("load", self.scratch[init_vars[i+1]], tmp_addrs[i+1]),
                ]})
            else:
                # Odd one out - defer and combine with const 0,1,2 loads below
                odd_idx = i
                odd_addr = tmp_addrs[i]
                odd_var = init_vars[i]

        # OPTIMIZED: Pack odd parameter with const 0,1,2 loads
        zero_const = self.alloc_scratch("zero_const")
        one_const = self.alloc_scratch("one_const")
        two_const = self.alloc_scratch("two_const")
        self.const_map[0] = zero_const
        self.const_map[1] = one_const
        self.const_map[2] = two_const
        # Preload tree[0] for round 0 optimization
        tree0_scalar = self.alloc_scratch("tree0_scalar")
        tree0_v = self.alloc_scratch("tree0_v", VLEN)
        # Combine: const for odd param + const 0
        self.add_packed({"load": [("const", odd_addr, odd_idx), ("const", zero_const, 0)]})
        # Combine: load odd param + const 1
        self.add_packed({"load": [("load", self.scratch[odd_var], odd_addr), ("const", one_const, 1)]})
        # Combine: const 2 + tree0_scalar load
        self.add_packed({"load": [("const", two_const, 2), ("load", tree0_scalar, self.scratch["forest_values_p"])]})

        zero_v = self.alloc_scratch("zero_v", VLEN)
        one_v = self.alloc_scratch("one_v", VLEN)
        two_v = self.alloc_scratch("two_v", VLEN)
        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)
        forest_base_v = self.alloc_scratch("forest_base_v", VLEN)

        # EARLY SETUP: Allocate block offset addresses for interleaving
        vector_batch_early = (batch_size // VLEN) * VLEN
        block_offset_values_early = list(range(0, vector_batch_early, VLEN))  # 0, 8, 16, ..., 248
        block_off_addrs = []
        for i in range(len(block_offset_values_early)):
            addr = self.alloc_scratch(f"block_off_{i}")
            block_off_addrs.append(addr)

        # Load only base offsets (every 4th); compute the rest with ALU adds
        base_indices = list(range(0, len(block_offset_values_early), 4))  # 0, 4, 8, 12, 16, 20, 24, 28
        eight_const = self.alloc_scratch("eight_const")
        sixteen_const = self.alloc_scratch("sixteen_const")
        twentyfour_const = self.alloc_scratch("twentyfour_const")

        # Build early block offset loads list
        early_block_loads = []
        for i in base_indices:
            early_block_loads.append((block_off_addrs[i], block_offset_values_early[i]))
        early_block_loads.append((eight_const, 8))
        early_block_loads.append((sixteen_const, 16))
        early_block_loads.append((twentyfour_const, 24))
        early_load_idx = 0

        # OPTIMIZED: Pack 6 vbroadcasts WITH 2 early block offset loads
        vb_bundle = {"valu": [
            ("vbroadcast", zero_v, zero_const),
            ("vbroadcast", one_v, one_const),
            ("vbroadcast", two_v, two_const),
            ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]),
            ("vbroadcast", forest_base_v, self.scratch["forest_values_p"]),
            ("vbroadcast", tree0_v, tree0_scalar),
        ]}
        if early_load_idx + 1 < len(early_block_loads):
            vb_bundle["load"] = [
                ("const", early_block_loads[early_load_idx][0], early_block_loads[early_load_idx][1]),
                ("const", early_block_loads[early_load_idx+1][0], early_block_loads[early_load_idx+1][1]),
            ]
            early_load_idx += 2
        self.add_packed(vb_bundle)

        # Preload tree[1], tree[2] for round 1 optimization (idx in {1,2} after round 0)
        tree1_scalar = self.alloc_scratch("tree1_scalar")
        tree2_scalar = self.alloc_scratch("tree2_scalar")
        tree1_v = self.alloc_scratch("tree1_v", VLEN)
        tree2_v = self.alloc_scratch("tree2_v", VLEN)
        diff_1_2_v = self.alloc_scratch("diff_1_2_v", VLEN)  # tree2 - tree1
        # OPTIMIZED: Allocate const 3-6 early so we can combine ALU ops with loads
        three_const = self.alloc_scratch("three_const")
        four_const = self.alloc_scratch("four_const")
        five_const = self.alloc_scratch("five_const")
        six_const = self.alloc_scratch("six_const")
        self.const_map[3] = three_const
        self.const_map[4] = four_const
        self.const_map[5] = five_const
        self.const_map[6] = six_const
        # OPTIMIZED: Pack tree1/tree2 ALU with const 3,4 loads (saves 1 cycle)
        self.add_packed({
            "alu": [
                ("+", tree1_scalar, self.scratch["forest_values_p"], one_const),
                ("+", tree2_scalar, self.scratch["forest_values_p"], two_const),
            ],
            "load": [("const", three_const, 3), ("const", four_const, 4)],
        })
        # OPTIMIZED: Pack tree1/tree2 loads together
        self.add_packed({"load": [
            ("load", tree1_scalar, tree1_scalar),
            ("load", tree2_scalar, tree2_scalar),
        ]})
        # OPTIMIZED: Pack const 5,6 loads with tree1/tree2 broadcasts
        self.add_packed({
            "load": [("const", five_const, 5), ("const", six_const, 6)],
            "valu": [
                ("vbroadcast", tree1_v, tree1_scalar),
                ("vbroadcast", tree2_v, tree2_scalar),
            ],
        })

        # Preload tree[3..6] for round 2 optimization (idx in {3,4,5,6} after round 1)
        three_v = self.alloc_scratch("three_v", VLEN)
        tree3_scalar = self.alloc_scratch("tree3_scalar")
        tree4_scalar = self.alloc_scratch("tree4_scalar")
        tree5_scalar = self.alloc_scratch("tree5_scalar")
        tree6_scalar = self.alloc_scratch("tree6_scalar")
        tree3_v = self.alloc_scratch("tree3_v", VLEN)
        tree4_v = self.alloc_scratch("tree4_v", VLEN)
        tree5_v = self.alloc_scratch("tree5_v", VLEN)
        tree6_v = self.alloc_scratch("tree6_v", VLEN)
        diff_3_4_v = self.alloc_scratch("diff_3_4_v", VLEN)  # tree4 - tree3
        diff_5_6_v = self.alloc_scratch("diff_5_6_v", VLEN)  # tree6 - tree5

        # OPTIMIZED: Pack diff_1_2/three_v with tree3-6 ALU ops AND 2 early block offset loads
        tree_alu_bundle = {
            "valu": [
                ("-", diff_1_2_v, tree2_v, tree1_v),
                ("vbroadcast", three_v, three_const),
            ],
            "alu": [
                ("+", tree3_scalar, self.scratch["forest_values_p"], three_const),
                ("+", tree4_scalar, self.scratch["forest_values_p"], four_const),
                ("+", tree5_scalar, self.scratch["forest_values_p"], five_const),
                ("+", tree6_scalar, self.scratch["forest_values_p"], six_const),
            ],
        }
        if early_load_idx + 1 < len(early_block_loads):
            tree_alu_bundle["load"] = [
                ("const", early_block_loads[early_load_idx][0], early_block_loads[early_load_idx][1]),
                ("const", early_block_loads[early_load_idx+1][0], early_block_loads[early_load_idx+1][1]),
            ]
            early_load_idx += 2
        self.add_packed(tree_alu_bundle)
        # OPTIMIZED: Pack loads 2 per cycle
        # OPTIMIZED: Overlap loads with broadcasts - load tree3,4, then load tree5,6 with broadcast tree3,4
        self.add_packed({"load": [
            ("load", tree3_scalar, tree3_scalar),
            ("load", tree4_scalar, tree4_scalar),
        ]})
        self.add_packed({
            "load": [
                ("load", tree5_scalar, tree5_scalar),
                ("load", tree6_scalar, tree6_scalar),
            ],
            "valu": [
                ("vbroadcast", tree3_v, tree3_scalar),
                ("vbroadcast", tree4_v, tree4_scalar),
            ],
        })
        # Broadcast tree5,6 and compute diff_3_4 (all VALU) WITH 2 early block offset loads
        tree56_bundle = {"valu": [
            ("vbroadcast", tree5_v, tree5_scalar),
            ("vbroadcast", tree6_v, tree6_scalar),
            ("-", diff_3_4_v, tree4_v, tree3_v),
        ]}
        if early_load_idx + 1 < len(early_block_loads):
            tree56_bundle["load"] = [
                ("const", early_block_loads[early_load_idx][0], early_block_loads[early_load_idx][1]),
                ("const", early_block_loads[early_load_idx+1][0], early_block_loads[early_load_idx+1][1]),
            ]
            early_load_idx += 2
        self.add_packed(tree56_bundle)
        # diff_5_6 will be deferred to combine with first hash const load below
        deferred_diff_5_6 = ("-", diff_5_6_v, tree6_v, tree5_v)

        # OPTIMIZED: Load all hash constants first, then pack vbroadcasts
        hash_c1_v = []
        hash_c3_v = []
        hash_c3_s = []
        hash_mul_v = []

        # Phase 1: Allocate and load all scalar constants (pack const loads 2 per cycle)
        c1_scalars = []
        c3_scalars = []
        mul_scalars = []

        # Collect all const values and addresses first
        const_loads = []  # List of (addr, value)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # c1 const
            if val1 not in self.const_map:
                addr = self.alloc_scratch(f"hash_c1_s_{hi}")
                self.const_map[val1] = addr
                const_loads.append((addr, val1))
            c1_scalars.append(self.const_map[val1])

            # c3 const
            if val3 not in self.const_map:
                addr = self.alloc_scratch(f"hash_c3_s_{hi}")
                self.const_map[val3] = addr
                const_loads.append((addr, val3))
            c3_scalars.append(self.const_map[val3])
            hash_c3_s.append(self.const_map[val3])

            # mul const for fusible stages
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul = (1 + (1 << val3)) % (2**32)
                if mul not in self.const_map:
                    addr = self.alloc_scratch(f"hash_mul_s_{hi}")
                    self.const_map[mul] = addr
                    const_loads.append((addr, mul))
                mul_scalars.append((hi, self.const_map[mul]))
            else:
                mul_scalars.append((hi, None))

        # Pack const loads 2 per cycle, combine first batch with deferred diff_5_6
        for i in range(0, len(const_loads), 2):
            if i + 1 < len(const_loads):
                bundle = {"load": [
                    ("const", const_loads[i][0], const_loads[i][1]),
                    ("const", const_loads[i + 1][0], const_loads[i + 1][1]),
                ]}
                # Combine diff_5_6 with first batch of const loads (saves 1 cycle)
                if i == 0 and deferred_diff_5_6:
                    bundle["valu"] = [deferred_diff_5_6]
                self.add_packed(bundle)
            else:
                self.add("load", ("const", const_loads[i][0], const_loads[i][1]))

        # Phase 2: Allocate vector destinations
        for hi in range(len(HASH_STAGES)):
            c1_v = self.alloc_scratch(f"hash_c1_v_{hi}", VLEN)
            c3_v = self.alloc_scratch(f"hash_c3_v_{hi}", VLEN)
            hash_c1_v.append(c1_v)
            hash_c3_v.append(c3_v)

        for hi, mul_scalar in mul_scalars:
            if mul_scalar is not None:
                mul_v = self.alloc_scratch(f"hash_mul_v_{hi}", VLEN)
                hash_mul_v.append(mul_v)
            else:
                hash_mul_v.append(None)

        # Phase 3: Interleave vbroadcasts with REMAINING block offset loads
        # Some block offset loads already done in early interleaving above
        # Build remaining loads list from early_load_idx onwards
        remaining_block_loads = early_block_loads[early_load_idx:]

        # Collect all hash vbroadcasts
        all_broadcasts = []
        for i in range(len(HASH_STAGES)):
            all_broadcasts.append(("vbroadcast", hash_c1_v[i], c1_scalars[i]))
        for i in range(len(HASH_STAGES)):
            all_broadcasts.append(("vbroadcast", hash_c3_v[i], c3_scalars[i]))
        for hi in range(len(HASH_STAGES)):
            if hash_mul_v[hi] is not None:
                all_broadcasts.append(("vbroadcast", hash_mul_v[hi], mul_scalars[hi][1]))

        # Pack: 2 const loads + 6 vbroadcasts per cycle while we have both
        bc_idx = 0
        rem_idx = 0
        while bc_idx < len(all_broadcasts) or rem_idx < len(remaining_block_loads):
            bundle = {}

            # Add up to 6 vbroadcasts
            valu_ops = []
            while len(valu_ops) < 6 and bc_idx < len(all_broadcasts):
                valu_ops.append(all_broadcasts[bc_idx])
                bc_idx += 1
            if valu_ops:
                bundle["valu"] = valu_ops

            # Add up to 2 const loads from remaining
            load_ops = []
            while len(load_ops) < 2 and rem_idx < len(remaining_block_loads):
                addr, val = remaining_block_loads[rem_idx]
                load_ops.append(("const", addr, val))
                rem_idx += 1
            if load_ops:
                bundle["load"] = load_ops

            if bundle:
                self.add_packed(bundle)

        # Compute remaining block offsets using ALU adds from the base offsets.
        # Combine last ALU cycle with pause to save 1 cycle
        offset_alu_ops = []
        last_base_idx = base_indices[-1] if base_indices else None
        for base_idx in base_indices:
            base_addr = block_off_addrs[base_idx]
            offset_alu_ops.append(("+", block_off_addrs[base_idx + 1], base_addr, eight_const))
            offset_alu_ops.append(("+", block_off_addrs[base_idx + 2], base_addr, sixteen_const))
            offset_alu_ops.append(("+", block_off_addrs[base_idx + 3], base_addr, twentyfour_const))
            if len(offset_alu_ops) == SLOT_LIMITS["alu"]:
                # If this is the last flush, combine with pause
                if base_idx == last_base_idx:
                    self.add_packed({"alu": offset_alu_ops, "flow": [("pause",)]})
                else:
                    self.add_packed({"alu": offset_alu_ops})
                offset_alu_ops = []
        if offset_alu_ops:
            # Combine remaining ALU ops with pause
            self.add_packed({"alu": offset_alu_ops, "flow": [("pause",)]})
        body_instrs = []

        buffers = []
        vector_batch = (batch_size // VLEN) * VLEN
        vector_blocks = vector_batch // VLEN
        pipe_buffers = min(13, vector_blocks) if vector_blocks else 0  # optimized from 10
        for bi in range(pipe_buffers):
            buffers.append({
                "idx": self.alloc_scratch(f"idx_v{bi}", VLEN),
                "val": self.alloc_scratch(f"val_v{bi}", VLEN),
                "node": self.alloc_scratch(f"node_val_v{bi}", VLEN),
                "addr": self.alloc_scratch(f"addr_v{bi}", VLEN),
                "tmp1": self.alloc_scratch(f"tmp1_v{bi}", VLEN),
                "tmp2": self.alloc_scratch(f"tmp2_v{bi}", VLEN),
                "cond": self.alloc_scratch(f"cond_v{bi}", VLEN),
                "idx_addr": self.alloc_scratch(f"idx_addr{bi}"),
                "val_addr": self.alloc_scratch(f"val_addr{bi}"),
            })

        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Block offsets already loaded above
        block_offsets = block_off_addrs

        def schedule_all_rounds():
            if vector_blocks == 0:
                return []
            instrs = []
            active = []
            free_bufs = list(range(pipe_buffers))
            next_block = 0

            def start_block():
                nonlocal next_block
                if next_block >= vector_blocks or not free_bufs:
                    return False
                buf_idx = free_bufs.pop(0)
                active.append({
                    "block": next_block,
                    "buf_idx": buf_idx,
                    "buf": buffers[buf_idx],
                    "offset": block_offsets[next_block],
                    "phase": "init_addr",
                    "round": 0,
                    "stage": 0,
                    "gather": 0,
                })
                next_block += 1
                return True

            while free_bufs and next_block < vector_blocks:
                start_block()

            while active or next_block < vector_blocks:
                while free_bufs and next_block < vector_blocks:
                    start_block()

                alu_ops = []
                load_ops = []
                valu_ops = []
                store_ops = []
                flow_ops = []

                alu_slots = SLOT_LIMITS["alu"]
                load_slots = SLOT_LIMITS["load"]
                valu_slots = SLOT_LIMITS["valu"]
                store_slots = SLOT_LIMITS["store"]
                flow_slots = SLOT_LIMITS["flow"]

                scheduled_this_cycle = set()

                # Wrap threshold: only need bounds check after round 9 for n_nodes=2047
                # Max idx after round r is 2^(r+2) - 2
                # After round 9: 2046 < 2047 (no wrap), after round 10: 4094 > 2047 (wrap)
                wrap_threshold = 10

                def next_round_phase(current_round):
                    """Determine next phase after completing a round."""
                    next_r = current_round + 1
                    if next_r >= rounds:
                        return "store_both"

                    # Calculate effective depth
                    # For rounds 0-wrap_threshold, depth = next_r
                    # For rounds after wrap (11+), depth restarts from 0
                    # After wrap at round 10, indices reset to 0, so:
                    #   round 11 = depth 0, round 12 = depth 1, etc.
                    if next_r <= wrap_threshold:
                        depth = next_r
                    else:
                        # After wrap: round 11 = depth 0, round 12 = depth 1, etc.
                        depth = next_r - wrap_threshold - 1

                    # Use selection for depths 0-2 (both before and after wrap).
                    # Depth 3+ selection adds too much VALU - use gather instead.
                    is_after_wrap = next_r > wrap_threshold

                    if depth == 0:
                        # Round 0 never hits this (special-cased in vload)
                        # But round 11 (after wrap) does
                        return "round0_xor"
                    elif depth == 1:
                        return "round1_select"
                    elif depth == 2:
                        return "round2_select1"
                    else:
                        return "addr"  # Gather for depth >= 3

                # Priority 1: Flow operations (vselect for bounds) - only for rounds >= wrap_threshold
                update4_blocks = [b for b in active if b["phase"] == "update4" and b["block"] not in scheduled_this_cycle]
                update4_blocks.sort(key=lambda b: b["round"], reverse=True)
                for block in update4_blocks:
                    if flow_slots == 0:
                        break
                    buf = block["buf"]
                    flow_ops.append(("vselect", buf["idx"], buf["cond"], buf["idx"], zero_v))
                    block["round"] += 1
                    block["stage"] = 0
                    block["gather"] = 0
                    block["next_phase"] = next_round_phase(block["round"] - 1)
                    scheduled_this_cycle.add(block["block"])
                    flow_slots -= 1

                # Priority 2: Stores (2 per cycle)
                for block in active:
                    if store_slots == 0:
                        break
                    if block["phase"] == "store_both" and block["block"] not in scheduled_this_cycle:
                        buf = block["buf"]
                        store_ops.append(("vstore", buf["val_addr"], buf["val"]))
                        block["next_phase"] = "store_idx"
                        scheduled_this_cycle.add(block["block"])
                        store_slots -= 1

                for block in active:
                    if store_slots == 0:
                        break
                    if block["phase"] == "store_idx" and block["block"] not in scheduled_this_cycle:
                        buf = block["buf"]
                        store_ops.append(("vstore", buf["idx_addr"], buf["idx"]))
                        block["next_phase"] = "done"
                        scheduled_this_cycle.add(block["block"])
                        store_slots -= 1

                # Priority 3: Vloads (need 2 load slots)
                for block in active:
                    if load_slots < 2:
                        break
                    if block["phase"] == "vload" and block["block"] not in scheduled_this_cycle:
                        buf = block["buf"]
                        load_ops.append(("vload", buf["idx"], buf["idx_addr"]))
                        load_ops.append(("vload", buf["val"], buf["val_addr"]))
                        # For round 0, skip addr and gather phases (all idx=0, use tree0_v)
                        if block["round"] == 0:
                            block["next_phase"] = "round0_xor"
                        else:
                            block["next_phase"] = "addr"
                        scheduled_this_cycle.add(block["block"])
                        load_slots -= 2

                # Priority 4: Gather loads (fill remaining load slots from multiple blocks)
                # Skip gather for round 0 - use tree0_v directly
                for block in active:
                    if load_slots == 0:
                        break
                    if block["phase"] == "gather":
                        if block["round"] == 0:
                            # Round 0: all idx=0, use preloaded tree[0]
                            buf = block["buf"]
                            # Copy tree0_v to node (this is a valu op, handled below)
                            block["next_phase"] = "round0_xor"
                            scheduled_this_cycle.add(block["block"])
                        else:
                            buf = block["buf"]
                            while load_slots > 0 and block["gather"] < VLEN:
                                lane = block["gather"]
                                load_ops.append(("load_offset", buf["node"], buf["addr"], lane))
                                block["gather"] += 1
                                load_slots -= 1
                            if block["gather"] >= VLEN:
                                block["next_phase"] = "xor"
                                scheduled_this_cycle.add(block["block"])

                # Priority 5: VALU operations - unified scheduling to fill all 6 slots
                # Build a list of all schedulable VALU tasks with their costs and priorities
                valu_tasks = []
                for block in active:
                    if block["block"] in scheduled_this_cycle:
                        continue
                    phase = block["phase"]
                    buf = block["buf"]
                    # Priority 0 = highest (closer to completion)
                    if phase == "update3":
                        valu_tasks.append((0, 1, block, "update3"))
                    elif phase == "update2":
                        valu_tasks.append((1, 1, block, "update2"))
                    elif phase == "update1":
                        valu_tasks.append((2, 2, block, "update1"))
                    elif phase == "hash_op2":
                        valu_tasks.append((5, 1, block, "hash_op2"))
                    elif phase == "hash_mul":
                        valu_tasks.append((3, 1, block, "hash_mul"))
                    elif phase == "hash_op1":
                        valu_tasks.append((5, 1, block, "hash_op1"))
                    elif phase == "xor":
                        valu_tasks.append((7, 1, block, "xor"))
                    elif phase == "round0_xor":
                        valu_tasks.append((7, 1, block, "round0_xor"))
                    elif phase == "round1_select":
                        valu_tasks.append((7, 1, block, "round1_select"))  # optimized from 6 to 7
                    elif phase == "round2_select1":
                        valu_tasks.append((6, 1, block, "round2_select1"))  # offset, priority 6
                    elif phase == "round2_select2":
                        valu_tasks.append((4, 1, block, "round2_select2"))  # sel1, priority 4 (optimized)
                    elif phase == "round2_select3":
                        valu_tasks.append((6, 2, block, "round2_select3"))  # low, high, priority 6
                    elif phase == "round2_select4":
                        valu_tasks.append((6, 1, block, "round2_select4"))  # diff
                    elif phase == "round2_select5":
                        valu_tasks.append((6, 1, block, "round2_select5"))  # node
                    elif phase == "addr":
                        # Optimized priority (5) for better scheduling
                        valu_tasks.append((5, 1, block, "addr"))

                # Sort by priority (lower = higher priority)
                valu_tasks.sort(key=lambda x: x[0])

                # Schedule tasks greedily
                for priority, cost, block, phase in valu_tasks:
                    if block["block"] in scheduled_this_cycle:
                        continue
                    buf = block["buf"]

                    if phase == "hash_op1":
                        hi = block["stage"]
                        op1 = HASH_STAGES[hi][0]
                        op3 = HASH_STAGES[hi][3]
                        # Prefer offloading op3 shift to scalar ALU (8 lanes) when slots allow.
                        if alu_slots >= VLEN and valu_slots >= 1:
                            valu_ops.append((op1, buf["tmp1"], buf["val"], hash_c1_v[hi]))
                            for lane in range(VLEN):
                                alu_ops.append((op3, buf["tmp2"] + lane, buf["val"] + lane, hash_c3_s[hi]))
                            alu_slots -= VLEN
                            valu_slots -= 1
                            block["next_phase"] = "hash_op2"
                            scheduled_this_cycle.add(block["block"])
                            continue
                        if valu_slots >= 2:
                            valu_ops.append((op1, buf["tmp1"], buf["val"], hash_c1_v[hi]))
                            valu_ops.append((op3, buf["tmp2"], buf["val"], hash_c3_v[hi]))
                            valu_slots -= 2
                            block["next_phase"] = "hash_op2"
                            scheduled_this_cycle.add(block["block"])
                            continue
                        continue

                    if valu_slots < cost:
                        continue

                    if phase == "update3":
                        valu_ops.append(("<", buf["cond"], buf["idx"], n_nodes_v))
                        block["next_phase"] = "update4"
                    elif phase == "update2":
                        valu_ops.append(("+", buf["idx"], buf["idx"], buf["tmp1"]))
                        # Skip wrap check for rounds where idx can't exceed n_nodes:
                        # - Rounds 0-9 (before wrap): idx grows but stays < n_nodes
                        # - Rounds 11-15 (after wrap): idx reset to 0 and grows small
                        # Only round 10 (wrap_threshold) needs the wrap check
                        if block["round"] != wrap_threshold:
                            block["round"] += 1
                            block["stage"] = 0
                            block["gather"] = 0
                            block["next_phase"] = next_round_phase(block["round"] - 1)
                        else:
                            block["next_phase"] = "update3"
                    elif phase == "update1":
                        valu_ops.append(("&", buf["tmp1"], buf["val"], one_v))
                        valu_ops.append(("multiply_add", buf["idx"], buf["idx"], two_v, one_v))
                        block["next_phase"] = "update2"
                    elif phase == "hash_op2":
                        hi = block["stage"]
                        op2 = HASH_STAGES[hi][2]
                        valu_ops.append((op2, buf["val"], buf["tmp1"], buf["tmp2"]))
                        if hi + 1 == len(HASH_STAGES):
                            block["next_phase"] = "update1"
                        else:
                            block["stage"] = hi + 1
                            block["next_phase"] = "hash_mul" if hash_mul_v[hi + 1] is not None else "hash_op1"
                    elif phase == "hash_mul":
                        hi = block["stage"]
                        mul_v = hash_mul_v[hi]
                        valu_ops.append(("multiply_add", buf["val"], buf["val"], mul_v, hash_c1_v[hi]))
                        if hi + 1 == len(HASH_STAGES):
                            block["next_phase"] = "update1"
                        else:
                            block["stage"] = hi + 1
                            block["next_phase"] = "hash_mul" if hash_mul_v[hi + 1] is not None else "hash_op1"
                    elif phase == "xor":
                        valu_ops.append(("^", buf["val"], buf["val"], buf["node"]))
                        block["next_phase"] = "hash_mul" if hash_mul_v[0] is not None else "hash_op1"
                    elif phase == "round0_xor":
                        # Round 0: XOR with preloaded tree[0] vector
                        valu_ops.append(("^", buf["val"], buf["val"], tree0_v))
                        block["next_phase"] = "hash_mul" if hash_mul_v[0] is not None else "hash_op1"
                    elif phase == "round1_select":
                        # Round 1: idx in {1,2}. tmp1 already holds parity == idx - 1.
                        # node = tree1 + tmp1 * (tree2 - tree1)
                        valu_ops.append(("multiply_add", buf["node"], diff_1_2_v, buf["tmp1"], tree1_v))
                        block["next_phase"] = "xor"
                    elif phase == "round2_select1":
                        # Round 2: compute offset = idx - 3 into tmp2 (preserve tmp1 parity)
                        # Then compute sel1 = offset >> 1 in next cycle
                        valu_ops.append(("-", buf["tmp2"], buf["idx"], three_v))
                        block["next_phase"] = "round2_select2"
                    elif phase == "round2_select2":
                        # Compute sel1 = offset >> 1
                        # Try to also do low/high computations if we have slots (they don't depend on cond yet)
                        valu_ops.append((">>", buf["cond"], buf["tmp2"], one_v))  # sel1
                        block["next_phase"] = "round2_select3"
                    elif phase == "round2_select3":
                        # Compute low/high; sel0 == parity in tmp1
                        valu_ops.append(("multiply_add", buf["tmp2"], diff_3_4_v, buf["tmp1"], tree3_v))  # low
                        valu_ops.append(("multiply_add", buf["node"], diff_5_6_v, buf["tmp1"], tree5_v))  # high
                        block["next_phase"] = "round2_select4"
                    elif phase == "round2_select4":
                        # Compute diff = high - low, then final selection if we have 2 slots
                        # node = node - tmp2, then node = node * cond + tmp2
                        # These have RAW dependency (second reads node from first), so can't combine
                        valu_ops.append(("-", buf["node"], buf["node"], buf["tmp2"]))
                        block["next_phase"] = "round2_select5"
                    elif phase == "round2_select5":
                        # Final selection: node = low + sel1 * diff
                        valu_ops.append(("multiply_add", buf["node"], buf["node"], buf["cond"], buf["tmp2"]))
                        block["next_phase"] = "xor"
                    elif phase == "addr":
                        valu_ops.append(("+", buf["addr"], buf["idx"], forest_base_v))
                        block["next_phase"] = "gather"

                    scheduled_this_cycle.add(block["block"])
                    valu_slots -= cost

                # Priority 6: ALU for init_addr (2 slots each)
                for block in active:
                    if alu_slots < 2:
                        break
                    if block["phase"] == "init_addr" and block["block"] not in scheduled_this_cycle:
                        buf = block["buf"]
                        alu_ops.append(("+", buf["idx_addr"], self.scratch["inp_indices_p"], block["offset"]))
                        alu_ops.append(("+", buf["val_addr"], self.scratch["inp_values_p"], block["offset"]))
                        block["next_phase"] = "vload"
                        scheduled_this_cycle.add(block["block"])
                        alu_slots -= 2

                if not (alu_ops or load_ops or valu_ops or store_ops or flow_ops):
                    # Check if any block is in gather phase but wasn't fully scheduled
                    stuck = False
                    for block in active:
                        if block["phase"] == "gather" and block["gather"] < VLEN:
                            stuck = True
                            break
                    if not stuck:
                        raise RuntimeError("scheduler made no progress")
                    # Otherwise we need another cycle to continue gather
                    continue

                instr = {}
                if alu_ops:
                    instr["alu"] = alu_ops
                if load_ops:
                    instr["load"] = load_ops
                if valu_ops:
                    instr["valu"] = valu_ops
                if store_ops:
                    instr["store"] = store_ops
                if flow_ops:
                    instr["flow"] = flow_ops
                instrs.append(instr)

                # Apply state transitions
                new_active = []
                for block in active:
                    next_phase = block.pop("next_phase", None)
                    if next_phase:
                        block["phase"] = next_phase
                    if block["phase"] == "done":
                        free_bufs.append(block["buf_idx"])
                    else:
                        new_active.append(block)
                active = new_active

            return instrs

        body_instrs.extend(schedule_all_rounds())

        for round_i in range(rounds):
            for i in range(vector_batch, batch_size):
                tail_slots = []
                i_const = self.scratch_const(i)
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                tail_slots.append(("load", ("load", tmp_idx, tmp_addr)))
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                tail_slots.append(("load", ("load", tmp_val, tmp_addr)))
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                tail_slots.append(("load", ("load", tmp_node_val, tmp_addr)))
                tail_slots.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                tail_slots.extend(self.build_hash(tmp_val, tmp1, tmp2, round_i, i))
                tail_slots.append(("alu", ("%", tmp1, tmp_val, two_const)))
                tail_slots.append(("alu", ("==", tmp1, tmp1, zero_const)))
                tail_slots.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                tail_slots.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                tail_slots.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                tail_slots.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                tail_slots.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                tail_slots.append(("store", ("store", tmp_addr, tmp_idx)))
                tail_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                tail_slots.append(("store", ("store", tmp_addr, tmp_val)))
                body_instrs.extend(self.build(tail_slots))

        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    do_kernel_test(10, 16, 256)
