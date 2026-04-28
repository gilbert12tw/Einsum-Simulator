"""
Multi-stream path splitting helpers used by the experimental C++ backend.
"""


def _original_two_stream_split(postorder_path, split_contraction):
    split_index = postorder_path.index(split_contraction)
    stream_0 = postorder_path[:split_index]
    stream_1 = postorder_path[split_index:]
    return stream_0, stream_1


def build_multistream_paths(tree, postorder_path, verbose=False):
    left_tree_id = tree.triple_path[-1][0]
    left_tree_node = tree.nodes_dict[left_tree_id]
    left_tree_node_contraction_triple = left_tree_node.contraction_triple

    split_method_used = "original"
    best_split = tree.find_balanced_split_point()
    if not best_split:
        stream_0, stream_1 = _original_two_stream_split(
            postorder_path,
            left_tree_node_contraction_triple,
        )
        return {
            "stream_0": stream_0,
            "stream_1": stream_1,
            "stream_2": [],
            "sync_info": [],
            "split_method_used": split_method_used,
            "best_split": None,
        }

    log = print if verbose else (lambda *args, **kwargs: None)
    log("\n========== Balanced Split ==========")
    log(f"Total intermediate nodes: {best_split['total_intermediate_nodes']}")
    log(f"Best split point (Node {best_split['node_id']}):")
    log(f"  - Left subtree: {best_split['left_size']} nodes")
    log(f"  - Right subtree: {best_split['right_size']} nodes")
    log(f"  - Upper tree: {best_split['upper_size']} nodes")
    log(
        f"Max part ({best_split['max_part_name']}): "
        f"{best_split['max_part_size']} ~= {best_split['other_two_sum']} (other two)"
    )
    log(f"Ratio: {best_split['ideal_ratio']:.3f} (ideal: 0.500)")
    log("=" * 36)

    original_tree_structure = tree.backup_structure()
    subtree_ops, remaining_ops, isolated_size = tree.adjust_and_split(best_split)
    changes = tree.compare_structure_changes(original_tree_structure)
    log(f"Adjusted structure on {len(changes)} nodes")

    if len(remaining_ops) < isolated_size:
        log("Split anomaly detected, falling back to original split method.")
        stream_0, stream_1 = _original_two_stream_split(
            postorder_path,
            left_tree_node_contraction_triple,
        )
        stream_2 = []
        sync_info = []
    else:
        stream_0 = subtree_ops
        stream_1_full = remaining_ops
        stream_1 = stream_1_full[:isolated_size]
        stream_2 = stream_1_full[isolated_size:]
        split_method_used = "tree-adjusted"

        split_node = tree.nodes_dict[best_split["node_id"]]
        sync_info = [split_node.contraction_triple]

        log("\n========== 3-Stream Split ==========")
        log(f"Stream 0 (left + small subtree): {len(stream_0)} ops")
        log(f"Stream 1 (large subtree):        {len(stream_1)} ops")
        log(f"Stream 2 (split point + upper):  {len(stream_2)} ops")
        log(f"Total: {len(stream_0) + len(stream_1) + len(stream_2)} ops")

    if verbose:
        adjusted_contraction_path = tree.postorder_traverse_contractions()
        is_valid_full, _, details = tree.validate_contraction_sequence(
            adjusted_contraction_path
        )
        log("\n========== Validation ==========")
        log(f"Contraction sequence validation: {'Valid' if is_valid_full else 'Invalid'}")
        log(f"Original tensor count: {len(details['original_tensors'])}")
        log(f"Total contraction operations: {details['total_contractions']}")
        log(f"Final tensor: {details.get('final_tensor', 'N/A')}")
        log("=" * 32)

    return {
        "stream_0": stream_0,
        "stream_1": stream_1,
        "stream_2": stream_2,
        "sync_info": sync_info,
        "split_method_used": split_method_used,
        "best_split": best_split,
    }


def extract_multistream_parts(tree, postorder_path, verbose=False):
    split_result = build_multistream_paths(tree, postorder_path, verbose=verbose)
    return (
        split_result["stream_0"],
        split_result["stream_1"],
        split_result["stream_2"],
        split_result["sync_info"],
        split_result["split_method_used"],
    )
