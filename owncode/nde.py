import torch

from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding

class NeuralDevelopmentalEncodingWithLoading(NeuralDevelopmentalEncoding):

    def save(self) -> None:
        """Save the model parameters to a file."""
        torch.save(self.state_dict(), 'nde_model.pt')
    
    def load(self, file_path: str = 'nde_model.pt') -> None:
        """Load the model parameters from a file."""
        self.load_state_dict(torch.load(file_path))
    
