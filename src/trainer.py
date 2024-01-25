import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model:nn.Module,
                 epochs:int = 10,
                 train_set=None,
                 val_set=None,
                 test_set=None,
                 optim=None) -> None:
        
        self.model = model
        self.epochs = epochs
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.optim = optim() or None

        pass

    def __train_one_epoch(self) -> None:
        for x, y in self.train_set:
            self.optim.zero_grad()
            output = self.model(x)
            pass

        pass

    def __validate(self) -> None:
        pass

    def __test(self) -> None:
        pass

    def train(self) -> None:
        for e in range(self.epochs):
            print(f"Training the {e + 1}° epoch...")
            self.__train_one_epoch()
            print(f"Validating...")
            self.__validate

            if self.test_set:
                print("Testing...")
                self.__test()
            
            print(f"Finished training for the {e + 1}° epoch")

if __name__ == "__main__":
    print("Hello World!")