import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0 
        self.F1=0
        self.F2 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,F1,F2,model,modelname,str):

        score = accs

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.save_checkpoint(val_loss, model,modelname,str)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f}|NR F1: {:.4f}|R F1: {:.4f}"
                      .format(self.accs,self.F1,self.F2))
        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.save_checkpoint(val_loss, model,modelname,str)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, modelname,str):
        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        self.val_loss_min = val_loss
