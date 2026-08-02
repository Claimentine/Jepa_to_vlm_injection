"""
Shared task-category vocabulary for the "does the VLM already know the task"
probe. Category names match the EgoDex part2 folder names (== hdf5 `task`
attribute after underscore-normalization).

Kept in its own module so sampling / inference / scoring all use exactly the
same option order (needed to score deterministically).
"""

# folder name -> short human-readable phrase used as a multiple-choice option
CATEGORY_PHRASES = {
    "assemble_disassemble_furniture_bench_chair": "Assembling or disassembling a chair from a furniture kit",
    "assemble_disassemble_furniture_bench_desk": "Assembling or disassembling a desk from a furniture kit",
    "assemble_disassemble_furniture_bench_drawer": "Assembling or disassembling a drawer from a furniture kit",
    "assemble_disassemble_furniture_bench_lamp": "Assembling or disassembling a lamp and its shade",
    "assemble_disassemble_furniture_bench_square_table": "Assembling or disassembling a square table",
    "assemble_disassemble_furniture_bench_stool": "Assembling or disassembling a stool or bench",
    "basic_fold": "Folding a piece of clothing (e.g. a t-shirt) on a table",
    "basic_pick_place": "Picking up an object and placing it somewhere else",
    "fold_stack_unstack_unfold_cloths": "Folding, stacking, unstacking, or unfolding cloths",
    "fold_unfold_paper_basic": "Folding or unfolding a sheet of paper",
    "fold_unfold_paper_origami": "Folding or unfolding an origami shape",
    "insert_remove_furniture_bench_cabinet": "Inserting or removing an item from a cabinet",
    "insert_remove_furniture_bench_round_table": "Assembling/disassembling or inserting/removing parts of a round table",
}

CATEGORIES = list(CATEGORY_PHRASES.keys())  # fixed order, ground-truth index source

assert len(CATEGORIES) == 13
