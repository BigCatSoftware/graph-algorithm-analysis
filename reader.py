"""
Graph Algorithm Analysis System

Hierarchical data structure analysis with network visualization and performance optimization.
Implements graph traversal algorithms, structural metrics, and visualization tools.
"""

import ast
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional


class HierarchicalGraph:
    """
    Directed acyclic graph for analyzing hierarchical decision structures.

    Provides algorithms for traversal, depth analysis, branching factor computation,
    and network visualization for complex tree-like data structures.
    """

    def __init__(self):
        self.nodes = set()
        self.edges = []
        self.children = defaultdict(list)
        self.parents = defaultdict(list)
        self.root = None

    def add_edge(self, parent: tuple, child: tuple):
        """
        Add directed edge from parent to child node.

        Args:
            parent (tuple): Source node identifier
            child (tuple): Target node identifier
        """
        self.nodes.add(parent)
        self.nodes.add(child)
        self.edges.append((parent, child))
        self.children[parent].append(child)
        self.parents[child].append(parent)

    def find_root(self) -> Optional[tuple]:
        """
        Identify root node using topological analysis.

        Returns:
            tuple: Root node with no incoming edges, None if not found
        """
        for node in self.nodes:
            if not self.parents[node]:
                return node
        return None

    def get_terminal_nodes(self) -> List[tuple]:
        """
        Extract all terminal nodes (leaves) from the graph.

        Returns:
            List[tuple]: Nodes with no outgoing edges
        """
        return [node for node in self.nodes if not self.children[node]]

    def compute_node_depth(self, node: tuple) -> int:
        """
        Calculate shortest path distance from root using breadth-first search.

        Args:
            node (tuple): Target node for depth calculation

        Returns:
            int: Depth level from root, -1 if unreachable
        """
        if node == self.root:
            return 0

        # BFS for shortest path computation
        queue = deque([(self.root, 0)])
        visited = set()

        while queue:
            current, depth = queue.popleft()
            if current == node:
                return depth

            if current in visited:
                continue
            visited.add(current)

            for child in self.children[current]:
                queue.append((child, depth + 1))

        return -1  # Node unreachable from root

    def analyze_structure(self):
        """
        Perform comprehensive structural analysis of the hierarchical graph.
        Computes network metrics including connectivity, depth distribution, and branching patterns.
        """
        if not self.nodes:
            print("Error: Empty graph structure")
            return

        terminal_nodes = self.get_terminal_nodes()
        max_depth = max(self.compute_node_depth(node) for node in
                        self.nodes) if self.nodes else 0

        print("\nHierarchical Graph Analysis")
        print("=" * 50)
        print(f"Network size: {len(self.nodes)} nodes")
        print(f"Edge count: {len(self.edges)}")
        print(f"Root node: {self.root}")
        print(f"Terminal nodes: {len(terminal_nodes)}")
        print(f"Maximum depth: {max_depth}")

        if self.root:
            root_branching = len(self.children[self.root])
            print(f"Root branching factor: {root_branching}")

        # Compute average branching factor
        internal_nodes = [node for node in self.nodes if self.children[node]]
        if internal_nodes:
            avg_branching = sum(
                len(self.children[node]) for node in internal_nodes) / len(
                internal_nodes)
            print(f"Average branching factor: {avg_branching:.2f}")

        print(f"\nTerminal node analysis:")
        if terminal_nodes:
            for i, leaf in enumerate(
                    sorted(terminal_nodes)[:10]):  # Show first 10
                print(f"  {leaf}")
            if len(terminal_nodes) > 10:
                print(
                    f"  ... and {len(terminal_nodes) - 10} more terminal nodes")
        else:
            print("  No terminal nodes detected")

    def visualize_network(self, output_path="network_analysis.png",
                          show_labels=True):
        """
        Generate network visualization with hierarchical layout and color coding.

        Args:
            output_path (str): File path for saved visualization
            show_labels (bool): Whether to display node labels
        """
        G = nx.DiGraph()

        # Build NetworkX graph structure
        for node in self.nodes:
            G.add_node(str(node))

        for parent, child in self.edges:
            G.add_edge(str(parent), str(child))

        # Generate hierarchical layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # Create visualization
        plt.figure(figsize=(16, 12))

        # Color nodes by hierarchy level
        node_colors = []
        terminal_nodes = self.get_terminal_nodes()

        for node_str in G.nodes():
            node = ast.literal_eval(node_str)
            if node == self.root:
                node_colors.append('#ff6b6b')  # Root in red
            elif node in terminal_nodes:
                node_colors.append('#51cf66')  # Terminal nodes in green
            else:
                node_colors.append('#74c0fc')  # Internal nodes in blue

        # Render network components
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=1000, alpha=0.8, edgecolors='black',
                               linewidths=1)

        nx.draw_networkx_edges(G, pos, edge_color='#495057',
                               arrows=True, arrowsize=25, alpha=0.7, width=1.5)

        # Add labels for smaller networks
        if show_labels and len(self.nodes) <= 30:
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold',
                                    font_color='white')

        plt.title(
            "Hierarchical Network Analysis\n(Red=Root, Green=Terminal, Blue=Internal)",
            fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()

        # Export visualization
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"\nNetwork visualization exported: {output_path}")
        plt.show()

    def print_hierarchical_structure(self):
        """Display tree structure in hierarchical text format."""
        if not self.root:
            print("Error: No root node identified")
            return

        def print_subtree(node, depth=0):
            indent = "  " * depth
            print(f"{indent}{node}")
            for child in sorted(self.children[node]):
                print_subtree(child, depth + 1)

        print("Hierarchical Structure:")
        print("=" * 40)
        print_subtree(self.root)


def load_hierarchical_data(data_file_path: str) -> Optional[HierarchicalGraph]:
    """
    Parse structured data file and construct hierarchical graph.

    Supports multiple formats: CSV, tab-separated, space-separated tuple pairs.

    Args:
        data_file_path (str): Path to input data file

    Returns:
        HierarchicalGraph: Constructed graph structure, None on failure
    """
    graph = HierarchicalGraph()

    try:
        with open(data_file_path, 'r') as datafile:
            lines = datafile.readlines()

            # Skip header row if present
            start_line = 0
            if lines and ('parent' in lines[0].lower() or 'source' in lines[
                0].lower()):
                start_line = 1

            for line_num, line in enumerate(lines[start_line:],
                                            start_line + 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    # Handle multiple data formats
                    if '","' in line:
                        # Quoted CSV format
                        parts = line.split('","')
                        if len(parts) == 2:
                            parent_str = parts[0].replace('"', '').strip()
                            child_str = parts[1].replace('"', '').strip()
                        else:
                            continue
                    elif line.count(',') >= 3:
                        # Tuple CSV format with comma separation
                        paren_count = 0
                        split_pos = -1
                        for i, char in enumerate(line):
                            if char == '(':
                                paren_count += 1
                            elif char == ')':
                                paren_count -= 1
                            elif char == ',' and paren_count == 0:
                                split_pos = i
                                break

                        if split_pos > 0:
                            parent_str = line[:split_pos].strip()
                            child_str = line[split_pos + 1:].strip()
                        else:
                            continue
                    elif '\t' in line:
                        # Tab-delimited format
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            parent_str = parts[0].strip()
                            child_str = parts[1].strip()
                        else:
                            continue
                    else:
                        # Space-delimited format
                        parts = line.split(None, 1)
                        if len(parts) >= 2:
                            parent_str = parts[0].strip()
                            child_str = parts[1].strip()
                        else:
                            continue

                    # Parse node identifiers
                    parent = ast.literal_eval(parent_str)
                    child = ast.literal_eval(child_str)
                    graph.add_edge(parent, child)

                except (ValueError, SyntaxError) as e:
                    print(
                        f"Warning: Parse error on line {line_num}: '{line}' - {e}")
                    continue

        # Identify root node
        graph.root = graph.find_root()

        print(
            f"Successfully loaded hierarchical structure from {data_file_path}")
        print(
            f"Graph contains {len(graph.nodes)} nodes and {len(graph.edges)} edges")

        return graph

    except FileNotFoundError:
        print(f"Error: Data file '{data_file_path}' not found")
        return None
    except Exception as e:
        print(f"Error loading data file: {e}")
        return None


def compute_path_analysis(graph: HierarchicalGraph):
    """
    Perform comprehensive path analysis from root to all terminal nodes.

    Args:
        graph (HierarchicalGraph): Input graph structure
    """

    def extract_all_paths(graph):
        """Extract all paths from root to terminal nodes using depth-first search."""
        paths = []

        def dfs_path_extraction(node, current_path):
            current_path.append(node)

            if not graph.children[node]:  # Terminal node reached
                paths.append(current_path.copy())
            else:
                for child in graph.children[node]:
                    dfs_path_extraction(child, current_path)

            current_path.pop()

        if graph.root:
            dfs_path_extraction(graph.root, [])

        return paths

    # Generate path statistics
    all_paths = extract_all_paths(graph)
    print(f"\nPath Analysis Results:")
    print("=" * 40)
    print(f"Total root-to-terminal paths: {len(all_paths)}")

    if all_paths:
        path_lengths = [len(path) for path in all_paths]
        avg_path_length = sum(path_lengths) / len(path_lengths)

        print(f"Average path length: {avg_path_length:.2f}")
        print(f"Shortest path length: {min(path_lengths)}")
        print(f"Longest path length: {max(path_lengths)}")

        print(f"\nSample paths (first 3):")
        for i, path in enumerate(all_paths[:3]):
            path_str = " → ".join(map(str, path))
            print(f"  Path {i + 1}: {path_str}")

    return all_paths


def analyze_state_transitions(graph: HierarchicalGraph):
    """
    Analyze transition patterns between nodes to identify structural patterns.

    Args:
        graph (HierarchicalGraph): Input graph structure
    """
    print(f"\nState Transition Analysis:")
    print("=" * 40)
    print(f"Root state: {graph.root}")

    if graph.root:
        state_dimensions = len(graph.root)
        print(f"State vector dimensions: {state_dimensions}")

        # Analyze which dimensions change most frequently
        dimension_changes = defaultdict(int)
        for parent, child in graph.edges:
            for i in range(min(len(parent), len(child))):
                if parent[i] != child[i]:
                    dimension_changes[f"dimension_{i}"] += 1

        print("Transition frequency by dimension:")
        for dimension, count in sorted(dimension_changes.items()):
            print(f"  {dimension}: {count} transitions")


def main():
    """
    Main analysis pipeline for hierarchical graph processing.
    Performs structure loading, analysis, and visualization.
    """
    print("Hierarchical Graph Analysis System")
    print("=" * 50)

    # Load data structure
    input_file = 'network_data.txt'
    graph = load_hierarchical_data(input_file)

    if graph is None:
        print("Failed to load graph structure. Terminating analysis.")
        return

    # Display structure for small graphs
    if len(graph.nodes) <= 25:
        graph.print_hierarchical_structure()
    else:
        print(
            "Graph too large for complete structure display. Generating summary analysis...")

    # Perform structural analysis
    graph.analyze_structure()

    # Generate network visualization
    try:
        graph.visualize_network(
            output_path="hierarchical_network_analysis.png")
    except ImportError:
        print(
            "Warning: Matplotlib/NetworkX unavailable. Skipping visualization.")
    except Exception as e:
        print(f"Visualization error: {e}")

    # Advanced analysis modules
    print("\nAdvanced Analysis Modules:")
    print("=" * 40)

    # Path analysis
    compute_path_analysis(graph)

    # State transition analysis
    analyze_state_transitions(graph)

    print(
        f"\nAnalysis complete. Network contains {len(graph.nodes)} nodes with maximum depth {max(graph.compute_node_depth(node) for node in graph.nodes) if graph.nodes else 0}")


if __name__ == "__main__":
    main()