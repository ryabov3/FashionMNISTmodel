class Earlystopping:
    def __init__(self, attemps):
        self.attemps = attemps
        self.counter = 0
        self.best_loss = float("inf")
    
    def __call__(self, loss_value):
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.counter = 0
            return False
        elif self.counter > self.attemps:
            return True
        
        else:
            self.counter += 1