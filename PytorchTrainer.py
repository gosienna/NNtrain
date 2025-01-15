from abc import ABC, abstractmethod
class PytorchTrainer(ABC):
    def __init__(self, 
                 model, 
                 optimizer, 
                 criterion, 
                 device,
                 args: dict):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.args = args

    @abstractmethod
    def train(self, train_loader):
        """Train the model for one epoch"""
        pass
        

class DiffusionTrainer(PytorchTrainer):
    def __init__(self, 
                 model, 
                 optimizer, 
                 criterion, 
                 device,
                 args: dict):
        super().__init__(model, optimizer, criterion, device, args)
    
    def train(self, data_loader):
        self.model.train()
        for epoch in range(self.args['num_epochs']):
            for x, y in data_loader:
                # Get some data and prepare the corrupted version
                x = x.to(self.device)  # Data on the GPU
                noise_amount = torch.rand(x.shape[0]).to(self.device)  # Pick random noise amounts
                noisy_x = corrupt(x, noise_amount)  # Create our noisy x

                # Get the model prediction
                pred = self.model(noisy_x)

                # Calculate the loss
                loss = self.criterion(pred, x)  # How close is the output to the true 'clean' x?

                # Backprop and update the params:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Store the loss for later
                losses.append(loss.item())

            # Print our the average of the loss values for this epoch:
            avg_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}")