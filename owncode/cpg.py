import numpy as np
import numpy.typing as npt
import networkx as nx

SEED = 42
RNG = np.random.default_rng(seed=SEED)

# Type Aliases
type ArrayLike = npt.NDArray[np.float64]


def decode_genotype_to_cpg(genotype: np.ndarray, num_neurons: int) -> tuple[dict[int, list[int]], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Decode a 1D genotype vector into CPG parameters.
    """
    gene_cursor = 0

    # Connection genes (binary)
    adj_size = num_neurons * num_neurons
    conn_flat = np.ones(adj_size, dtype=int)
    adjacency_list = {
        i: [j for j in range(num_neurons) if conn_flat[i*num_neurons + j]]
        for i in range(num_neurons)
    }
    gene_cursor += adj_size

    # Frequencies
    omega = 2 * np.pi * (0.5 + genotype[gene_cursor:gene_cursor + num_neurons])
    gene_cursor += num_neurons

    # Amplitudes (scaled to ±π/2)
    A = (np.pi / 2) * genotype[gene_cursor:gene_cursor + num_neurons]
    gene_cursor += num_neurons

    # Coupling strengths
    h_flat = genotype[gene_cursor:gene_cursor + adj_size]
    h_matrix = h_flat.reshape(num_neurons, num_neurons)
    gene_cursor += adj_size

    # Phase differences
    phase_flat = (genotype[gene_cursor:gene_cursor + adj_size] * 2 * np.pi) - np.pi
    phase_diff = phase_flat.reshape(num_neurons, num_neurons)

    return adjacency_list, omega, A, h_matrix, phase_diff

class HopfCPG:
    def __init__(
        self,
        num_neurons: int,
        adjacency_list: dict[int, list[int]],
        dt: float = 0.02,
        h: float = 0.1,
        alpha: float = 1.0,
    ) -> None:
        # --- Inherent parameters --- #
        # Number of neurons
        self.num_neurons = num_neurons

        # Time step
        self.dt = dt

        # Learning rate
        self.alpha = np.ones(num_neurons) * alpha

        # Coupling coefficient
        self.h = h

        # Adjacency list for coupling
        self.adjacency_list = adjacency_list
        if len(adjacency_list) != num_neurons:
            raise ValueError(
                "Adjacency list length must match number of neurons."
            )

        # --- Initialize state variables --- #
        self.init_state = 0.5
        self.x: ArrayLike = RNG.uniform(
            -self.init_state, self.init_state, self.num_neurons
        )
        self.y: ArrayLike = RNG.uniform(
            -self.init_state, self.init_state, self.num_neurons
        )

        # --- Adjustable parameters --- #
        # Angular frequency (1Hz default)
        self.omega = np.ones(num_neurons) * 2 * np.pi

        # Amplitude
        self.A = np.ones(num_neurons) * 1.0

        # Phase differences
        self.phase_diff: ArrayLike = np.zeros((num_neurons, num_neurons))

        # Set up ring topology phase differences (neighbors are π/2 apart)
        for i, conn in adjacency_list.items():
            for j in conn:
                self.phase_diff[i, j] = np.pi / 2

class EvolvableCPG(HopfCPG):
    def __init__(
        self,
        num_neurons: int,
        genotype: np.ndarray | None = None,
        dt: float = 0.02,
        h: float = 0.1,
        alpha: float = 1.0,
    ) -> None:
        
        if genotype is None:
            genotype = self.generate_random_cpg_genotype(num_neurons)

        adjacency_list, omega, A, h_matrix, phase_diff = decode_genotype_to_cpg(genotype, num_neurons)

        super().__init__(num_neurons=num_neurons, adjacency_list=adjacency_list, dt=dt, h=h, alpha=alpha)
        self.omega = omega
        self.A = A
        self.phase_diff = phase_diff
        self.h_matrix = h_matrix  # per-connection coupling
        self.genotype = genotype

    def step(self) -> np.ndarray:
        """
        Advance the CPG by one timestep and return joint updates for each hinge.
        Output shape: (num_hinges,)
        """
        new_x = np.zeros_like(self.x)
        new_y = np.zeros_like(self.y)

        # Compute Hopf dynamics and coupling
        for i, conn in self.adjacency_list.items():
            r_squared = self.x[i] ** 2 + self.y[i] ** 2
            x_dot = self.alpha[i] * (self.A[i] ** 2 - r_squared) * self.x[i] - self.omega[i] * self.y[i]
            y_dot = self.omega[i] * self.x[i] + self.alpha[i] * (self.A[i] ** 2 - r_squared) * self.y[i]

            for j in conn:
                rot = np.array([
                    [np.cos(self.phase_diff[i, j]), -np.sin(self.phase_diff[i, j])],
                    [np.sin(self.phase_diff[i, j]),  np.cos(self.phase_diff[i, j])]
                ])
                coupled = self.h_matrix[i, j] * (rot @ np.array([self.x[j], self.y[j]]))
                x_dot += coupled[0]
                y_dot += coupled[1]

            new_x[i] = self.x[i] + x_dot * self.dt
            new_y[i] = self.y[i] + y_dot * self.dt

        self.x, self.y = new_x, new_y

        # Map neuron states to hinge commands (angle updates)
        # Here we use sine of x to ensure smooth cyclical output
        hinge_updates = np.sin(self.x) * self.A  # shape: (num_hinges,)
        return hinge_updates
    
    @staticmethod
    def generate_random_cpg_genotype(num_neurons: int) -> np.ndarray:
        """
        Generate a random genotype for EvolvableCPG.
        
        Structure:
        - Connection genes: num_neurons^2 binary {0,1}
        - Frequencies (ω): num_neurons floats [0, 1] (later scaled)
        - Amplitudes (A): num_neurons floats [0, 1] (later scaled to ±π/2)
        - Coupling strengths (h): num_neurons^2 floats [0, 1]
        - Phase differences: num_neurons^2 floats [0, 1] (later scaled to [-π, π])
        """
        # Connection genes (0 or 1)
        # conn_genes = RNG.integers(0, 2, size=num_neurons**2)
        conn_genes = np.ones(num_neurons ** 2, dtype=int)

        # Frequencies
        freq_genes = RNG.random(size=num_neurons)

        # Amplitudes
        amp_genes = RNG.random(size=num_neurons)

        # Coupling strengths
        h_genes = RNG.random(size=num_neurons**2)

        # Phase differences
        phase_genes = RNG.random(size=num_neurons**2)

        # Concatenate all genes into a single flat vector
        genotype = np.concatenate([
            conn_genes.astype(np.float32),
            freq_genes.astype(np.float32),
            amp_genes.astype(np.float32),
            h_genes.astype(np.float32),
            phase_genes.astype(np.float32),
        ])
        return genotype
    
    def get_genotype(self) -> np.ndarray:
        return self.genotype
    
    def set_genotype(self, genotype: np.ndarray) -> None:
        if len(genotype) != len(self.genotype):
            raise ValueError("Genotype length does not match.")
        
        self.genotype = genotype
        adjacency_list, omega, A, h_matrix, phase_diff = decode_genotype_to_cpg(genotype, self.num_neurons)
        self.adjacency_list = adjacency_list
        self.omega = omega
        self.A = A
        self.phase_diff = phase_diff
        self.h_matrix = h_matrix

def make_brain_from_body(body_graph: nx.DiGraph) -> nx.DiGraph:
    """
    Construct a brain graph that mirrors the nested body graph structure.
    Each HINGE becomes a motor neuron.
    Each BRICK that connects multiple parts becomes an interneuron.
    Connections follow morphological adjacency recursively.
    """
    brain = nx.DiGraph()

    def add_brain_nodes_recursive(node_id):
        node_data = body_graph.nodes[node_id]
        node_type = node_data.get("type", "NONE")

        # Add neuron for each functional body node
        if node_type in {"HINGE", "BRICK"}:
            brain.add_node(node_id, type=node_type)

        # Traverse all connected nodes (children)
        for child in body_graph.successors(node_id):
            child_data = body_graph.nodes[child]
            child_type = child_data.get("type", "NONE")

            # Recurse first
            add_brain_nodes_recursive(child)

            # Add edge between neurons if both exist
            if node_id in brain.nodes and child in brain.nodes:
                brain.add_edge(node_id, child)

    # Start recursion from morphological roots (no predecessors)
    roots = [n for n in body_graph.nodes if body_graph.in_degree(n) == 0]
    for root in roots:
        add_brain_nodes_recursive(root)

    return brain