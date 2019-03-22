'Loss function'

import torch

import torch.nn as nn
import torch.nn.functional as F


class DistanceFromAnswerLoss(nn.Module):
    """
    Penalizes outputs that predict answers that are far away from the location
    of the true answer.


    Parameters
    ----------
    coefficient : float
        Emphasis to be put to the loss value (higher value is more).

    device : torch.device
        Indicates location of computations and tensor storage.

    normalization : boolean
        Asks whether to normalize mask such that sum of coefficients is 1.

    penalization_type : str in ['square_root', 'linear', 'quadratic']
        Type of penalization function to apply

    reduction : str in ['mean']
        Type of post-treatment on the loss value.

    """
    def __init__(self, coefficient=.1, device=torch.device('cpu'), normalization=True,
                 penalization_type='linear', reduction='mean'):
        'Initialization'
        # Inherit methods of parent class
        super(DistanceFromAnswerLoss, self).__init__()

        # Save parameters
        self.coefficient = coefficient
        self.device = device
        self.normalization = normalization
        self.penalization_type = penalization_type
        self.reduction = reduction

    def forward(self, input, target):
        'Forward pass'
        # Transform log-probabilities to probabilities
        input = input.exp()

        # Create penalization mask
        mask = self._create_mask(input, target).to(self.device)

        # Element-wise multiplication of the two tensors
        out = mask * input

        # Normalize
        out = self._normalize(out)

        # Output loss
        return out

    def _create_mask(self, input, target):
        'Create mask function'
        # Parameters
        batch_size, c_len = input.size()

        # Initialization
        mask = torch.zeros(batch_size, c_len)

        # Loop through batch
        for i in range(batch_size):
            # Fill in beginning of vector
            mask[i] = self._penalization(target[i].item(), c_len)

        return mask

    def _normalize(self, tensor):
        'Normalization'
        # Case mean
        if self.normalization == 'mean':
            tensor = torch.mean(tensor, dim=0)

        return self.coefficient * tensor.sum()

    def _penalization(self, target_index, c_len):
        'Create penalization depending on mode'
        # Case no answer token encountered
        if target_index == 0:
            return torch.zeros(c_len)

        # Convert to float
        target_index, c_len = 1.0*target_index, 1.0*c_len

        # First tensor
        tensor_1 = torch.arange(-target_index,0).abs()

        # Second tensor
        tensor_2 = torch.arange(c_len-target_index)

        # Concatenate
        out = torch.cat([tensor_1, tensor_2])

        # Case square root
        if self.penalization_type == 'square_root':
            out = out.sqrt()

        # Case linear
        elif self.penalization_type == 'linear':
            pass

        # Case quadratic
        elif self.penalization_type == 'quadratic':
            out = out.pow(2)

        # Normalize
        if self.normalization:
            out = F.normalize(out, p=2, dim=0)

        # Return mask
        return out
