import numpy as np

def flatten_weights(W1: np.ndarray, W2: np.ndarray, W3: np.ndarray) -> np.ndarray:
    return np.concatenate([W1.flatten(), W2.flatten(), W3.flatten()])

class NNBrain:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: np.ndarray | None=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1, self.W2, self.W3 = self._randomize_weights()

        if weights is not None:
            self.set_weights(weights)

    def _randomize_weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.random.randn(self.input_size, self.hidden_size) * 0.2,
            np.random.randn(self.hidden_size, self.hidden_size) * 0.2,
            np.random.randn(self.hidden_size, self.output_size) * 0.2,
        )

    def get_random_weights(self) -> np.ndarray:
        return flatten_weights(*self._randomize_weights())

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.tanh(np.dot(x, self.W1))
        x = np.tanh(np.dot(x, self.W2))
        return np.tanh(np.dot(x, self.W3)) * (np.pi / 2)  # Scale outputs to [-pi/2, pi/2]

    def get_weights(self) -> np.ndarray:
        return flatten_weights(self.W1, self.W2, self.W3)

    def get_num_weights(self) -> int:
        return self.W1.size + self.W2.size + self.W3.size

    def set_weights(self, flat_weights: np.ndarray) -> None:
        if len(flat_weights) != self.get_num_weights():
            raise ValueError("Weights size does not match the brain architecture.")

        offset = 0
        for W in [self.W1, self.W2, self.W3]:
            size = W.size
            W[:] = flat_weights[offset:offset + size].reshape(W.shape)
            offset += size

    def get_weight_slice_indices(self) -> dict[str, slice]:
        indices = {}
        offset = 0
        for i, W in enumerate([self.W1, self.W2, self.W3], start=1):
            size = W.size
            indices[f"W{i}"] = slice(offset, offset + size)
            offset += size
        return indices
