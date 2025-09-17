# Graph Algorithm Analysis

Hierarchical network analysis with graph traversal algorithms and performance optimization metrics for complex data structures.

## Overview

This system implements advanced graph algorithms for analyzing hierarchical networks and tree-like data structures. Using breadth-first search, depth-first search, and network topology analysis, the system computes structural metrics, identifies optimization opportunities, and generates comprehensive visualizations for decision support and performance analysis.

## Key Features

- **Graph Traversal Algorithms**: BFS and DFS implementations for shortest path and exhaustive search
- **Topological Analysis**: Root identification and hierarchical structure validation
- **Performance Metrics**: Branching factor analysis, depth distribution, and connectivity measurement
- **Path Optimization**: Complete path enumeration with statistical analysis
- **Network Visualization**: Color-coded hierarchical layouts with NetworkX integration
- **Data Format Flexibility**: Support for CSV, tab-delimited, and space-separated input formats

## Mathematical Foundation

**Graph Theory Algorithms:**
- **Breadth-First Search**: Shortest path computation from root to any node
- **Depth-First Search**: Complete path enumeration for terminal node analysis
- **Topological Sorting**: Root node identification in directed acyclic graphs

**Performance Metrics:**
- **Branching Factor**: Average and maximum child node count per internal node
- **Depth Analysis**: Maximum and average path lengths from root to terminals
- **Connectivity Metrics**: Edge density and structural complexity measures

**State Transition Analysis:**
- **Dimensional Change Tracking**: Frequency analysis of state vector modifications
- **Transition Pattern Recognition**: Systematic analysis of node transformation patterns

## Applications

- **System Architecture Analysis**: Hierarchical system performance optimization
- **Decision Tree Evaluation**: Complex decision structure analysis and simplification
- **Network Topology Assessment**: Communication network efficiency analysis
- **Algorithm Performance Testing**: Comparative analysis of traversal methods

## Usage

### Basic Analysis Pipeline
```python
# Load hierarchical data structure
graph = load_hierarchical_data('network_data.txt')

# Perform comprehensive structural analysis
graph.analyze_structure()

# Generate network visualization
graph.visualize_network('analysis_output.png')

# Execute advanced path analysis
compute_path_analysis(graph)
```

### Command Line Execution
```bash
python graph_analysis.py
```

### Input Data Format
The system accepts hierarchical relationships in multiple formats:
```
# CSV format
"(1,2,3)","(1,2,4)"
"(1,2,4)","(1,3,4)"

# Tab-delimited format  
(1,2,3)    (1,2,4)
(1,2,4)    (1,3,4)

# Space-delimited format
(1,2,3) (1,2,4)
(1,2,4) (1,3,4)
```

## Dataset Structure

```
project/
├── network_data.txt          # Input hierarchical relationship data
├── graph_analysis.py         # Main analysis implementation
├── .gitignore               # Version control configuration
└── README.md                # Project documentation
```

## Dependencies

```python
matplotlib>=3.5.0
networkx>=2.6
numpy>=1.21.0  # For advanced numerical operations
```

## Installation

1. **Clone repository:**
```bash
git clone https://github.com/BigCatSoftware/graph-algorithm-analysis.git
cd graph-algorithm-analysis
```

2. **Install dependencies:**
```bash
pip install matplotlib networkx numpy
```

3. **Run analysis:**
```bash
python graph_analysis.py
```

The system will process `network_data.txt` and generate comprehensive analysis output including network visualizations.

## Output Analysis

The system generates detailed reports including:

### Structural Metrics
- **Network Size**: Total node and edge counts
- **Depth Distribution**: Maximum depth and average path lengths  
- **Branching Patterns**: Average branching factor and structural complexity
- **Terminal Analysis**: Count and distribution of leaf nodes

### Path Analysis
- **Complete Path Enumeration**: All root-to-terminal paths
- **Statistical Summary**: Path length distribution and optimization metrics
- **Transition Analysis**: State vector change frequency and patterns

### Visualization Output
- **Hierarchical Layout**: Color-coded network structure (Red=Root, Green=Terminal, Blue=Internal)
- **High-Resolution Export**: 300 DPI network diagrams for professional presentation
- **Scalable Display**: Automatic label management for networks of varying complexity

## Performance Characteristics

**Algorithmic Complexity:**
- **BFS Traversal**: O(V + E) for shortest path computation
- **DFS Enumeration**: O(V + E) for complete path extraction
- **Visualization**: O(V²) for layout optimization with NetworkX

**Memory Efficiency:**
- **Graph Storage**: Adjacency list representation for sparse networks
- **Path Caching**: Optimized memory usage for large-scale analysis
- **Incremental Processing**: Streaming analysis for memory-constrained environments

## Configuration Options

### Analysis Parameters
- **Visualization Thresholds**: Automatic label hiding for networks >30 nodes
- **Path Display Limits**: Configurable sample path count for large networks
- **Output Resolution**: Customizable DPI settings for publication-quality graphics

### Data Processing
- **Format Detection**: Automatic delimiter recognition for flexible input parsing
- **Error Handling**: Robust parsing with detailed error reporting
- **Header Recognition**: Automatic header row detection and skipping

## Technical Implementation

**Core Algorithm Classes:**
- `HierarchicalGraph`: Main graph data structure with traversal methods
- `load_hierarchical_data()`: Multi-format data parser with error handling
- `compute_path_analysis()`: Statistical path enumeration and analysis
- `analyze_state_transitions()`: Dimensional change pattern recognition

**Visualization Pipeline:**
- NetworkX integration for professional graph layouts
- Matplotlib rendering with customizable styling and color schemes
- Automatic node positioning using spring-force algorithms
- Export capabilities for multiple image formats

## Model Applications

**System Performance Analysis:**
- Hierarchical system bottleneck identification
- Communication network optimization
- Decision tree pruning and simplification

**Algorithm Validation:**
- Comparative traversal method analysis  
- Performance metric validation for optimization algorithms
- Structural complexity assessment for system design

## Limitations and Considerations

- **Directed Acyclic Graphs Only**: System assumes hierarchical structure without cycles
- **Memory Scaling**: Large networks (>10,000 nodes) may require memory optimization
- **Visualization Complexity**: Dense networks may produce cluttered visual output
- **Input Format Dependency**: Requires structured tuple-based node representation

## License

This project is available under the MIT License.
