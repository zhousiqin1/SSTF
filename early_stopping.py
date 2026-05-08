# early_stopping.py
# Early stopping utility for model training.
import os
import torch

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.001, save_dir='./models'):
        self.patience = patience      
        self.verbose = verbose
        self.counter = 0                
        self.best_score = None 
        self.early_stop = False         
        self.delta = delta               
        self.save_dir = save_dir       
        self.best_model_path = None

    def __call__(self, val_loss):
        if self.best_score is None:

            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:

            self.counter += 1
            if self.counter >= self.patience:

                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

        return self.early_stop
    
    def save_checkpoint(self, model, epoch, val_loss, val_acc, save_dir=None):

        if save_dir is None:
            save_dir = self.save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        should_save = False
        if self.best_score is None:
            # First save
            should_save = True
            self.best_score = val_loss
        elif val_loss < self.best_score + self.delta:
            # Validation loss improved
            should_save = True
            self.best_score = val_loss
            self.counter = 0               # Reset counter
        else:
            # No improvement
            self.counter += 1
        
        if should_save:
            model_filename = f"best_model_epoch_{epoch}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.pth"
            model_path = os.path.join(save_dir, model_filename)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_score': self.best_score,
            }, model_path)
            
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    os.remove(self.best_model_path)
                except:
                    pass
            
            self.best_model_path = model_path
            
            if self.verbose:
                if self.best_score is not None:
                    print(f'Validation loss improved ({self.best_score:.6f} --> {val_loss:.6f}). Model saved to {model_path}')
                else:
                    print(f'First model saved (validation loss: {val_loss:.6f}). Path: {model_path}')
        elif self.verbose:
            print(f'Validation loss did not improve ({self.best_score:.6f} vs {val_loss:.6f}), skipping save.')
    
    def load_best_model(self, model):
        """Load the best model checkpoint."""
        if self.best_model_path and os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint
        return None