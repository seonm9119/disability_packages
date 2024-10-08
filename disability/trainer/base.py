import os
import time
from disability.utils import setup_logger
from .utils import train, eval, save


class Trainer:
    def __init__(self, train_config=None):

        if train_config is not None:
            self.set_config(train_config)
            os.makedirs(self.save_path, exist_ok=True)
            self.set_logger(self.save_path)
        
        self.global_tr_losses = []
        self.global_val_losses = []

    def set_config(self, config):
        for key, value in config.items():
            setattr(self, key, value)

    def set_logger(self, save_path):
        self.logger = setup_logger(save_path)

    def train(self, **kwargs):
        return train(**kwargs)
    
    def eval(self, **kwargs):
        return eval(**kwargs)

    def run(self):

        for epoch in range(self.epochs):
            start_time = time.time()

            self.model = self.model.to(self.device)
            tr_losses = self.train(model=self.model,
                                   loader=self.loader['train'],
                                   optimizer=self.optimizer,
                                   criterion=self.criterion,
                                   device=self.device)
            
            val_losses, total, correct = self.eval(model=self.model,
                                                   loader=self.loader['val'],
                                                   criterion=self.criterion,
                                                   device=self.device)

            self.global_tr_losses.append(sum(tr_losses) / len(tr_losses))
            self.global_val_losses.append(sum(val_losses) / len(val_losses))

            self.logger.info(f"Val Progress --- total:{total}, correct:{correct.item()}")
            self.logger.info(f"Val Accuracy of the network on the {total} test images: {(100 * correct / total):.2f}%")

            self.logger.info(f"Epoch [{epoch + 1}/{self.epochs}], "
                        f"TRNLoss: {self.global_tr_losses[-1]:.4f}, "
                        f"VALoss: {self.global_val_losses[-1]:.4f}, "
                        f"Time: {(time.time() - start_time) / 60:.2f}")
            
            if (epoch + 1) % self.save_interval == 0 or epoch == self.epochs - 1:
                file_path = os.path.join(self.save_path, f"epoch_{epoch+1}.pth")
                save(epoch, self.model, self.optimizer, self.scheduler, 
                     self.global_tr_losses, self.global_val_losses, file_path)
                self.logger.info(f"Model saved at epoch {epoch+1} to {file_path}")

            if self.scheduler is not None:
                self.scheduler.step()

            


