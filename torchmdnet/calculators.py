import torch
from torchmdnet.models.model import load_model

class External:
    '''
    The External class is used to calculate the energy and forces of an external potential, such as a neural network. The class is initialized with the path to the neural network
    ckpt, the embeddings, the device on which the neural network should be run and the output_transform argument.  
    The output_transform is used to give a function that transform the energy and the forces. This is could be useful to convert the energy and the forces from the units of the 
    neural network (the ones used to trained it) to the units of the simulation. The output_transform argument should be a string that can be evaluated to a function. 
    The function should take two arguments, the energy and the forces, and return the transformed energy and the transformed forces.
    '''
    
    def __init__(self, netfile, embeddings, device="cpu", output_transform=None):
        self.model = load_model(netfile, device=device, derivative=True)
        self.device = device
        self.n_atoms = embeddings.size(1)
        self.embeddings = embeddings.reshape(-1).to(device)
        self.batch = torch.arange(embeddings.size(0), device=device).repeat_interleave(
            embeddings.size(1)
        )
        self.model.eval()

        if not output_transform:
            self.output_transformer = lambda energy, forces: (energy, forces)  # identity
        else:
            self.output_transformer = eval(output_transform)

    def calculate(self, pos, box):
        pos = pos.to(self.device).type(torch.float32).reshape(-1, 3)
        energy, forces = self.model(self.embeddings, pos, self.batch)

        return self.output_transformer(
            energy.detach(), forces.reshape(-1, self.n_atoms, 3).detach()
        )