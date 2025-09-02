
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import matplotlib.pyplot as plt
import os

class PerformanceVisualizationCallback(Callback):
    def __init__(self, validation_data, output_dir="performance_logs"):
        super().__init__()
        self.current_file_name = os.path.basename(__file__).split('.')[0]
        self.validation_data = validation_data

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = os.path.join(output_dir, self.current_file_name)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.aucs = []
        self.log_losses = []
        self.accuracies = []

    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val).flatten()  # Use self.model from Callback base class

        auc = roc_auc_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred)
        acc = accuracy_score(y_val, (y_pred > 0.5).astype(int))

        print(f"\nEpoch {epoch + 1} - AUC: {auc:.4f} - Log Loss: {logloss:.4f} - Accuracy: {acc:.4f}")

        self.aucs.append(auc)
        self.log_losses.append(logloss)
        self.accuracies.append(acc)

        # Plot and save metrics
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.aucs, label='AUC')
        plt.title('Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('Score')

        plt.subplot(1, 3, 2)
        plt.plot(self.log_losses, label='Log Loss', color='orange')
        plt.title('Validation Log Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 3)
        plt.plot(self.accuracies, label='Accuracy', color='green')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'performance_epoch_{epoch + 1}.png'))
        plt.close()
