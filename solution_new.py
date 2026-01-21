
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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        zero_v = self.alloc_scratch("zero_v", VLEN)
        one_v = self.alloc_scratch("one_v", VLEN)
        two_v = self.alloc_scratch("two_v", VLEN)
        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)
        forest_base_v = self.alloc_scratch("forest_base_v", VLEN)
        self.add("valu", ("vbroadcast", zero_v, zero_const))
        self.add("valu", ("vbroadcast", one_v, one_const))
        self.add("valu", ("vbroadcast", two_v, two_const))
        self.add("valu", ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", forest_base_v, self.scratch["forest_values_p"]))

        hash_c1_v = []
        hash_c3_v = []
        hash_mul_v = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1_scalar = self.scratch_const(val1)
            c1_v = self.alloc_scratch(f"hash_c1_v_{hi}", VLEN)
            self.add("valu", ("vbroadcast", c1_v, c1_scalar))
            c3_scalar = self.scratch_const(val3)
            c3_v = self.alloc_scratch(f"hash_c3_v_{hi}", VLEN)
            self.add("valu", ("vbroadcast", c3_v, c3_scalar))
            hash_c1_v.append(c1_v)
            hash_c3_v.append(c3_v)
            mul_v = None
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul = (1 + (1 << val3)) % (2**32)
                mul_scalar = self.scratch_const(mul)
                mul_v = self.alloc_scratch(f"hash_mul_v_{hi}", VLEN)
                self.add("valu", ("vbroadcast", mul_v, mul_scalar))
            hash_mul_v.append(mul_v)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))
        body_instrs = []

        buffers = []
        vector_batch = (batch_size // VLEN) * VLEN
        vector_blocks = vector_batch // VLEN
        pipe_buffers = min(20, vector_blocks) if vector_blocks else 0
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

        block_offsets = [self.scratch_const(i) for i in range(0, vector_batch, VLEN)]

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

                # Priority 1: Flow operations (vselect for bounds) - prioritize blocks in later rounds
                # Sort blocks by round (descending) to complete oldest blocks first
                update4_blocks = [b for b in active if b["phase"] == "update4" and b["block"] not in scheduled_this_cycle]
                update4_blocks.sort(key=lambda b: b["round"], reverse=True)
                for block in update4_blocks:
                    if flow_slots == 0:
                        break
                    buf = block["buf"]
                    flow_ops.append(("vselect", buf["idx"], buf["cond"], buf["idx"], zero_v))
                    if block["round"] + 1 < rounds:
                        block["round"] += 1
                        block["stage"] = 0
                        block["gather"] = 0
                        block["next_phase"] = "addr"
                    else:
                        block["next_phase"] = "store_both"
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
                        block["next_phase"] = "addr"
                        scheduled_this_cycle.add(block["block"])
                        load_slots -= 2

                # Priority 4: Gather loads (fill remaining load slots from multiple blocks)
                for block in active:
                    if load_slots == 0:
                        break
                    if block["phase"] == "gather":
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
                        valu_tasks.append((3, 1, block, "hash_op2"))
                    elif phase == "hash_mul":
                        valu_tasks.append((4, 1, block, "hash_mul"))
                    elif phase == "hash_op1":
                        valu_tasks.append((5, 2, block, "hash_op1"))
                    elif phase == "xor":
                        valu_tasks.append((6, 1, block, "xor"))
                    elif phase == "addr":
                        valu_tasks.append((7, 1, block, "addr"))

                # Sort by priority (lower = higher priority)
                valu_tasks.sort(key=lambda x: x[0])

                # Schedule tasks greedily
                for priority, cost, block, phase in valu_tasks:
                    if valu_slots < cost:
                        continue
                    if block["block"] in scheduled_this_cycle:
                        continue
                    buf = block["buf"]

                    if phase == "update3":
                        valu_ops.append(("<", buf["cond"], buf["idx"], n_nodes_v))
                        block["next_phase"] = "update4"
                    elif phase == "update2":
                        valu_ops.append(("+", buf["idx"], buf["idx"], buf["tmp1"]))
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
                    elif phase == "hash_op1":
                        hi = block["stage"]
                        op1 = HASH_STAGES[hi][0]
                        op3 = HASH_STAGES[hi][3]
                        valu_ops.append((op1, buf["tmp1"], buf["val"], hash_c1_v[hi]))
                        valu_ops.append((op3, buf["tmp2"], buf["val"], hash_c3_v[hi]))
                        block["next_phase"] = "hash_op2"
                    elif phase == "xor":
                        valu_ops.append(("^", buf["val"], buf["val"], buf["node"]))
                        block["next_phase"] = "hash_mul" if hash_mul_v[0] is not None else "hash_op1"
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
