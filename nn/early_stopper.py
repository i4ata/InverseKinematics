class EarlyStopper:

    def __init__(self, patience: int = 3):
        
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.save_model = False

    def check(self, validation_loss: float) -> bool:
        """If this returns true, stop early. Else, if self.save_model is true, save the model"""

        self.save_model = False
        if validation_loss > self.best_loss:
            self.counter += 1
            if self.counter == self.patience:
                return True
        else:
            self.best_loss = validation_loss
            self.counter = 0
            self.save_model = True
        return False