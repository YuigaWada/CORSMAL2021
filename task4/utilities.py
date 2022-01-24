import torch


class EarlyStopping():
    def __init__(self, model, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.model = model

    def update(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            return

        if self.best_loss > loss:
            print("[Early-Stopping]: reset counter.")
            print("[Early-Stopping]: saving weights ...")
            self.best_loss = loss
            self.counter = 0
            torch.save(self.model.state_dict(), './task4.pt')

        else:
            self.counter += 1
            print("[Early-Stopping]: {} / {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                print("[Early-Stopping]: STOP ... !")
                return True

        return False
