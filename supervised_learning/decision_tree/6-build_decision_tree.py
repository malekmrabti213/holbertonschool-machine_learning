#!/usr/bin/env python3
"""
Task 6 - Decision Tree
"""

import numpy as np


class Node:
    """
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        """
        if self.is_leaf:
            return self.depth
        else:
            left_depth = self.left_child.max_depth_below()
            right_depth = self.right_child.max_depth_below()
            return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        """
        count = 0
        if not only_leaves:
            count += 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def left_child_add_prefix(self, text):
        """
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"

        return new_text

    def right_child_add_prefix(self, text):
        """
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"

        return new_text

    def __str__(self):
        """
        """
        feature = self.feature
        threshold = self.threshold
        if self.is_root:
            a = self.left_child_add_prefix(f"{self.left_child}"[:-1])
            b = self.right_child_add_prefix(f"{self.right_child}"[:-1])
            return f"root [feature={feature}, threshold={threshold}]\n" \
                   f"{a}{b}"
        else:
            a = self.left_child_add_prefix(f"{self.left_child}"[:-1])
            b = self.right_child_add_prefix(f"{self.right_child}"[:-1])
            return f"-> node [feature={feature}, threshold={threshold}]\n" \
                   f"{a}{b}"

    def get_leaves_below(self):
        """
        """
        sleftc = self.left_child.get_leaves_below()
        srightc = self.right_child.get_leaves_below()
        return sleftc + srightc

    def update_bounds_below(self):
        """
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()
        if self.feature in self.left_child.lower.keys():
            self.left_child.lower[self.feature] = \
                max(self.threshold, self.left_child.lower[self.feature])
        else:
            self.left_child.lower[self.feature] = self.threshold
        if self.feature in self.right_child.upper.keys():
            self.right_child.upper[self.feature] = \
                min(self.threshold, self.right_child.upper[self.feature])
        else:
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        """
        def is_large_enough(x):
            """
            """
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                          for key in self.lower]),
                axis=0
            )

        def is_small_enough(x):
            """
            """
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    """

    def __init__(self, value, depth=None):
        super().__init__()
        """
        """
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        """
        return 1

    def __str__(self):
        """
        """
        return f"-> leaf [value={self.value}] "

    def get_leaves_below(self):
        """
        """
        return [self]

    def update_bounds_below(self):
        """
        """
        pass

    def pred(self, x):
        """
        """
        return self.value


class Decision_Tree():
    """
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        """
        self.root.update_bounds_below()

    def pred(self, x):
        """
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        vals = np.array([leaf.value for leaf in leaves])
        self.predict = lambda x: np.array(vals[np.argmax(
            np.array([leaf.indicator(x) for leaf in leaves]), axis=0)])
