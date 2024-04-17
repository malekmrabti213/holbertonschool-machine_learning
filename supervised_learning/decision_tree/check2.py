#!/usr/bin/env python3

import numpy as np
Node = __import__('2-build_decision_tree').Node
Leaf = __import__('2-build_decision_tree').Leaf
Decision_Tree = __import__('2-build_decision_tree').Decision_Tree

def random_tree(max_depth, n_classes, n_features, seed=0):
    """returns a random decision_tree T together with an explanatory array A"""
    assert max_depth > 0, "max_depth must be a strictly positive integer"
    rng = np.random.default_rng(seed)
    root = Node(is_root=True, depth=0)
    root.lower = {i: -100 for i in range(n_features)}
    root.upper = {i: 100 for i in range(n_features)}

    def build_children(node):
        feat = rng.integers(0, n_features)
        node.feature = feat
        node.threshold = np.round(rng.uniform(0, 1) * (node.upper[feat] - node.lower[feat]) + node.lower[feat], 2)
        if node.depth == max_depth - 1:
            node.left_child = Leaf(depth=max_depth, value=rng.integers(0, n_classes))
            node.right_child = Leaf(depth=max_depth, value=rng.integers(0, n_classes))
        else:
            node.left_child = Node(depth=node.depth + 1)
            node.left_child.lower = node.lower.copy()
            node.left_child.upper = node.upper.copy()
            node.left_child.lower[feat] = node.threshold
            node.right_child = Node(depth=node.depth + 1)
            node.right_child.lower = node.lower.copy()
            node.right_child.upper = node.upper.copy()
            node.right_child.upper[feat] = node.threshold
            build_children(node.left_child)
            build_children(node.right_child)

    T = Decision_Tree(root=root)
    build_children(root)

    A = rng.uniform(0, 1, size=100 * n_features).reshape([100, n_features]) * 200 - 100
    return T, A

def example_0() :
    leaf0         = Leaf(0, depth=1)
    leaf1         = Leaf(0, depth=2)
    leaf2         = Leaf(1, depth=2)
    internal_node = Node( feature=1, threshold=30000, left_child=leaf1, right_child=leaf2,          depth=1 )
    root          = Node( feature=0, threshold=.5   , left_child=leaf0, right_child=internal_node , depth=0 , is_root=True)
    return Decision_Tree(root=root)


def example_1(depth):
    level = [Leaf(i, depth=depth) for i in range(2 ** depth)]
    level.reverse()

    def get_v(node):
        if node.is_leaf:
            return node.value
        else:
            return node.threshold

    for d in range(depth):
        level = [Node(feature=0,
                      threshold=(get_v(level[2 * i]) + get_v(level[2 * i + 1])) / 2,
                      left_child=level[2 * i],
                      right_child=level[2 * i + 1], depth=depth - d - 1) for i in range(2 ** (depth - d - 1))]
    root = level[0]
    root.is_root = True
    return Decision_Tree(root=root)

# needed in the checkers :

def get_list_trees_and_explanatory():
    T = [example_0(), example_1(4)]
    np.random.seed(0)
    A = [np.array([[1, 22000], [0, 22000], [1, 42000], [0, 42000]]),
         np.random.rand(10).reshape(10, 1) * 16]
    T[0].name = "tree_of_the_intro ( example 0) )"
    T[1].name = "rounding_tree     ( example 1(4) )"

    for i in range(4):
        t, a = random_tree(3 + i, 2 + i, 2 + i, seed=i)
        t.name = f"random_tree number {i + 1}"
        T.append(t)
        A.append(a[:10, :])

    return zip(T, A)

# example_0, example_1 and random_tree are defined in the preamble

def checker_2() :
    for T,A in get_list_trees_and_explanatory()  :
        print("\n\n\n------------------------------------------------------------------------------------")
        print(T.name)
        print("------------------------------------------------------------------------------------")
        print(T)

checker_2()