import matplotlib.pyplot as  plt
import os

def log_train(predictions_valid, logs_path, epoch):
    '''
    Log train results.

    Usage
    ```python
    log_train(predictions_valid, logs_path, epoch)
    ```
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_valid, 'ro')
    plt.title('Dot Diagram of Predictions')
    plt.xlabel('Index')
    plt.ylabel('Prediction')
    plt.grid(True)
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join(logs_path ,f'plots/dot_diagram_epoch_{epoch}.png'))
    plt.close()

