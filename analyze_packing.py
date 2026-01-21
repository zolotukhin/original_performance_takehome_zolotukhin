from solution import KernelBuilder, Tree, Input

def analyze():
    # Parameters matching the test case
    forest_height = 10
    n_nodes = 2**forest_height - 1
    batch_size = 256
    rounds = 16
    
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    
    print(f"Total instructions: {len(kb.instrs)}")
    
    # Analyze slot usage
    # We are looking for the main loop (R3+).
    # Inspect the last 100 instructions or so, or search for the loop body.
    
    for i, bundle in enumerate(kb.instrs):
        counts = {k: len(v) for k, v in bundle.items() if k in ['load', 'valu', 'alu', 'flow', 'store']}
        # Filter for interesting lines (e.g. inside the loop)
        # The loop starts around... well let's just print them all if it's not too huge, 
        # or just summary stats.
        # But per-cycle usage is what we want.
        
        # Let's print the ones that have both load and valu to see how well we mix them.
        if 'load' in counts or 'valu' in counts:
             print(f"{i}: {counts}")

if __name__ == "__main__":
    analyze()
