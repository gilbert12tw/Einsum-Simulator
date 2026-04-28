"""
Minimal contraction-tree utilities for the experimental multi-stream backend.

This module keeps only the tree operations required by the QSVM multi-stream
path splitter. It intentionally avoids pulling in the larger research helper
module from the original backend bundle.
"""


class ContractionTree:
    """Represent a contraction path as a binary tree over tensor IDs."""

    def __init__(self, expression, path):
        self.expression = expression
        self.path = path
        self.tree = None
        self.input_tensor_subscripts_list = []
        self.generated_tensor_subscript = ""
        self.triple_path = []
        self.tensor_dict = {}
        self.nodes_dict = {}

        self._parse_expression()
        self._init_input_tensors()
        self._build_triple_path()
        self._build_tree()
        self._index_nodes()

    def _parse_expression(self):
        lhs, rhs = self.expression.split("->", 1)
        self.input_tensor_subscripts_list = lhs.split(",")
        self.generated_tensor_subscript = rhs

    def _init_input_tensors(self):
        for tensor_id, subscripts in enumerate(self.input_tensor_subscripts_list):
            self.tensor_dict[tensor_id] = list(subscripts)

    def _build_triple_path(self):
        dynamic_tensor_ids = list(range(len(self.input_tensor_subscripts_list)))
        dynamic_subscripts = self.input_tensor_subscripts_list.copy()
        next_tensor_id = len(dynamic_tensor_ids)
        all_subscripts = {
            tensor_id: subscript
            for tensor_id, subscript in enumerate(self.input_tensor_subscripts_list)
        }

        for left_pos, right_pos in self.path:
            left_tensor_id = dynamic_tensor_ids[left_pos]
            right_tensor_id = dynamic_tensor_ids[right_pos]

            left_subscript = dynamic_subscripts[left_pos]
            right_subscript = dynamic_subscripts[right_pos]
            result_subscript = "".join(
                sorted(set(left_subscript) ^ set(right_subscript))
            )

            result_tensor_id = next_tensor_id
            next_tensor_id += 1
            all_subscripts[result_tensor_id] = result_subscript
            self.triple_path.append((left_tensor_id, right_tensor_id, result_tensor_id))

            for pos in sorted([left_pos, right_pos], reverse=True):
                del dynamic_tensor_ids[pos]
                del dynamic_subscripts[pos]

            dynamic_tensor_ids.append(result_tensor_id)
            dynamic_subscripts.append(result_subscript)

        for tensor_id, subscript in all_subscripts.items():
            self.tensor_dict.setdefault(tensor_id, list(subscript))

    def _build_tree(self):
        nodes = {}

        for left_id, right_id, parent_id in self.triple_path:
            nodes.setdefault(left_id, TreeNode(left_id))
            nodes.setdefault(right_id, TreeNode(right_id))
            nodes.setdefault(parent_id, TreeNode(parent_id))

        for node_id, node in nodes.items():
            node.edges = self.tensor_dict.get(node_id, [])

        children = set()
        parents = set()
        for left_id, right_id, parent_id in self.triple_path:
            parent = nodes[parent_id]
            parent.left_tensor_id = left_id
            parent.right_tensor_id = right_id
            parent.left_node = nodes[left_id]
            parent.right_node = nodes[right_id]
            parent.contraction_triple = (left_id, right_id, parent_id)
            children.add(left_id)
            children.add(right_id)
            parents.add(parent_id)

        root_candidates = parents - children
        if len(root_candidates) != 1:
            raise ValueError("could not determine a unique contraction-tree root")
        self.tree = nodes[root_candidates.pop()]

    def _index_nodes(self):
        self.nodes_dict = {}
        self._collect_nodes(self.tree)

    def _collect_nodes(self, node):
        if node is None:
            return
        self.nodes_dict[node.tensor_id] = node
        self._collect_nodes(node.left_node)
        self._collect_nodes(node.right_node)

    def postorder_traverse_contractions(self):
        contractions = []
        self._postorder(self.tree, contractions)
        return contractions

    def _postorder(self, node, contractions):
        if node is None or node.is_leaf():
            return
        self._postorder(node.left_node, contractions)
        self._postorder(node.right_node, contractions)
        contractions.append(node.contraction_triple)

    def _calculate_intermediate_subtree_sizes(self, node, subtree_sizes):
        if node is None:
            return 0
        if node.is_leaf():
            subtree_sizes[node.tensor_id] = 0
            return 0

        left_size = self._calculate_intermediate_subtree_sizes(
            node.left_node, subtree_sizes
        )
        right_size = self._calculate_intermediate_subtree_sizes(
            node.right_node, subtree_sizes
        )
        size = left_size + right_size + 1
        subtree_sizes[node.tensor_id] = size
        return size

    def find_balanced_split_point(self):
        subtree_sizes = {}
        total_intermediate_nodes = self._calculate_intermediate_subtree_sizes(
            self.tree, subtree_sizes
        )
        if total_intermediate_nodes <= 1:
            return None

        ideal_max_size = total_intermediate_nodes / 2.0
        best_info = None
        best_score = float("inf")

        for node_id, node in self.nodes_dict.items():
            if node.is_leaf():
                continue

            left_size = (
                subtree_sizes.get(node.left_node.tensor_id, 0)
                if node.left_node is not None
                else 0
            )
            right_size = (
                subtree_sizes.get(node.right_node.tensor_id, 0)
                if node.right_node is not None
                else 0
            )
            current_subtree_size = subtree_sizes[node_id]
            upper_size = total_intermediate_nodes - current_subtree_size
            max_part_size = max(left_size, right_size, upper_size)
            score = abs(max_part_size - ideal_max_size)

            if score < best_score:
                best_score = score
                max_part_name = "left"
                if right_size == max_part_size:
                    max_part_name = "right"
                elif upper_size == max_part_size:
                    max_part_name = "upper"

                best_info = {
                    "node_id": node_id,
                    "node": node,
                    "left_size": left_size,
                    "right_size": right_size,
                    "upper_size": upper_size,
                    "max_part_size": max_part_size,
                    "other_two_sum": total_intermediate_nodes - max_part_size,
                    "score": score,
                    "ideal_ratio": (
                        max_part_size / total_intermediate_nodes
                        if total_intermediate_nodes > 0
                        else 0.0
                    ),
                    "contraction_triple": node.contraction_triple,
                    "total_intermediate_nodes": total_intermediate_nodes,
                    "max_part_name": max_part_name,
                }

        return best_info

    def backup_structure(self):
        backup = {}
        for node_id, node in self.nodes_dict.items():
            if node.is_leaf():
                continue
            backup[node_id] = {
                "left": node.left_node.tensor_id if node.left_node else None,
                "right": node.right_node.tensor_id if node.right_node else None,
            }
        return backup

    def compare_structure_changes(self, original_backup):
        changes = []
        for node_id, original in original_backup.items():
            node = self.nodes_dict[node_id]
            current_left = node.left_node.tensor_id if node.left_node else None
            current_right = node.right_node.tensor_id if node.right_node else None
            if original["left"] != current_left or original["right"] != current_right:
                changes.append(
                    {
                        "node_id": node_id,
                        "original": original,
                        "current": {"left": current_left, "right": current_right},
                    }
                )
        return changes

    def move_larger_subtree_to_right(self, split_node_id, left_size, right_size):
        split_node = self.nodes_dict[split_node_id]

        if left_size > right_size:
            split_node.left_node, split_node.right_node = (
                split_node.right_node,
                split_node.left_node,
            )
            if split_node.contraction_triple is not None:
                left_id, right_id, result_id = split_node.contraction_triple
                split_node.contraction_triple = (right_id, left_id, result_id)
            larger_size = left_size
        else:
            larger_size = right_size

        current_id = split_node_id
        while current_id != self.tree.tensor_id:
            found_parent = False
            for parent_id, parent in self.nodes_dict.items():
                if parent.is_leaf():
                    continue
                if (
                    parent.left_node is not None
                    and parent.left_node.tensor_id == current_id
                ):
                    parent.left_node, parent.right_node = (
                        parent.right_node,
                        parent.left_node,
                    )
                    if parent.contraction_triple is not None:
                        left_id, right_id, result_id = parent.contraction_triple
                        parent.contraction_triple = (right_id, left_id, result_id)
                    current_id = parent_id
                    found_parent = True
                    break
                if (
                    parent.right_node is not None
                    and parent.right_node.tensor_id == current_id
                ):
                    current_id = parent_id
                    found_parent = True
                    break
            if not found_parent:
                break

        return larger_size

    def split_into_two_streams(self, split_node_id, larger_subtree_size):
        all_ops = self.postorder_traverse_contractions()
        split_node = self.nodes_dict[split_node_id]
        if split_node.left_node is None or split_node.right_node is None:
            mid = len(all_ops) // 2
            return all_ops[:mid], all_ops[mid:]

        split_op = split_node.contraction_triple
        split_op_index = -1
        for index, op in enumerate(all_ops):
            if op == split_op:
                split_op_index = index
                break
            if (
                len(op) == 3
                and len(split_op) == 3
                and op[0] == split_op[1]
                and op[1] == split_op[0]
                and op[2] == split_op[2]
            ):
                split_op_index = index
                break

        if split_op_index == -1:
            mid = len(all_ops) // 2
            return all_ops[:mid], all_ops[mid:]

        larger_subtree_start = split_op_index - larger_subtree_size
        stream_0_ops = all_ops[:larger_subtree_start]
        stream_1_ops = all_ops[larger_subtree_start:]
        return stream_0_ops, stream_1_ops

    def adjust_and_split(self, split_info):
        larger_size = self.move_larger_subtree_to_right(
            split_info["node_id"],
            split_info["left_size"],
            split_info["right_size"],
        )
        stream_0_ops, stream_1_ops = self.split_into_two_streams(
            split_info["node_id"], larger_size
        )
        return stream_0_ops, stream_1_ops, larger_size

    def validate_contraction_sequence(self, contraction_sequence):
        available_tensors = {
            node_id
            for node_id, node in self.nodes_dict.items()
            if node.is_leaf()
        }
        original_tensors = available_tensors.copy()
        details = {
            "original_tensors": list(original_tensors),
            "steps": [],
            "total_contractions": len(contraction_sequence),
        }

        for index, contraction in enumerate(contraction_sequence):
            input1, input2, output = contraction
            step_info = {
                "step": index,
                "contraction": contraction,
                "input1": input1,
                "input2": input2,
                "output": output,
                "available_before": list(available_tensors),
            }

            if input1 not in available_tensors:
                step_info["error"] = f"Input tensor {input1} unavailable"
                details["steps"].append(step_info)
                return (
                    False,
                    f"Step {index}: input tensor {input1} is unavailable",
                    details,
                )

            if input2 not in available_tensors:
                step_info["error"] = f"Input tensor {input2} unavailable"
                details["steps"].append(step_info)
                return (
                    False,
                    f"Step {index}: input tensor {input2} is unavailable",
                    details,
                )

            available_tensors.remove(input1)
            available_tensors.remove(input2)
            available_tensors.add(output)

            step_info["available_after"] = list(available_tensors)
            step_info["status"] = "OK"
            details["steps"].append(step_info)

        if len(available_tensors) != 1:
            return (
                False,
                f"remaining tensors after contraction: {len(available_tensors)}",
                details,
            )

        details["final_tensor"] = list(available_tensors)[0]
        return True, "Contraction sequence is valid", details


class TreeNode:
    """Node in the contraction tree."""

    def __init__(self, tensor_id):
        self.tensor_id = tensor_id
        self.edges = []
        self.left_tensor_id = None
        self.right_tensor_id = None
        self.left_node = None
        self.right_node = None
        self.contraction_triple = None

    def is_leaf(self):
        return self.left_node is None and self.right_node is None
