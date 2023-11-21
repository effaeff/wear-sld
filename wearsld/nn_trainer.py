"""Trainer class for wear quantification"""

import numpy as np

from pytorchutils.globals import torch, DEVICE
from pytorchutils.basic_trainer import BasicTrainer
from config import INPUT_SIZE

from tqdm import tqdm


class Trainer(BasicTrainer):
    """Wrapper class for training routine"""
    def __init__(self, config, model, dataprocessor):
        BasicTrainer.__init__(self, config, model, dataprocessor)

    def learn_from_epoch(self, epoch_idx, verbose):
        """Training method"""
        epoch_loss = 0
        try:
            batches = self.get_batches_fn()
        except AttributeError:
            print(
                "Error: No nb_batches_fn defined in preprocessor. "
                "This attribute is required by the training routine."
            )

        for idx, __ in enumerate(batches):
            inp = batches[idx, :, :INPUT_SIZE]
            out = batches[idx, :, INPUT_SIZE]

            pred_out = self.predict(inp)

            # pred_out = torch.sigmoid(pred_out)

            batch_loss = self.loss(
                pred_out.flatten(),
                torch.FloatTensor(out).to(DEVICE)
            )

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()
        epoch_loss /= len(batches)

        return epoch_loss

    def predict(self, inp):
        """
        Capsuled prediction method.
        Only single model usage supported for now.
        """
        inp = torch.Tensor(inp).to(DEVICE)
        return self.model(inp)

    def evaluate(self, inp, out):
        """Prediction and error estimation for given input and output"""
        with torch.no_grad():
            # Switch to PyTorch's evaluation mode.
            # Some layers, which are used for regularization, e.g., dropout or batch norm layers,
            # behave differently, i.e., are turnd off, in evaluation mode
            # to prevent influencing the prediction accuracy.
            self.model.eval()

            pred_out = self.predict(inp).cpu().detach().numpy()
            # pred_out = torch.sigmoid(pred_out)
            if out:
                error = np.sqrt((pred_out - out)**2)

                return pred_out, error
            else:
                return pred_out
