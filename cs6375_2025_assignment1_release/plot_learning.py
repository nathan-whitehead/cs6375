# plot the learning curve of the best model including the training loss and validation accuracy by epoch
import matplotlib.pyplot as plt
import numpy as np

# load the training loss and validation accuracy
training_loss = [
  1.2108354600071907, 0.8438054381906986, 0.6354487210214138, 0.5231910569295287,
  0.44793275017291306, 0.41527720146626235, 0.371101736985147, 0.35257324969396,
  0.340491942808032, 0.31631186514906584
]
validation_accuracy = [
  0.49,
  0.48625,
  0.5425,
  0.48125,
  0.49625,
  0.5275,
  0.515,
  0.535,
  0.48625,
  0.51375,
]

# plot the learning curve

fig, ax1 = plt.subplots()

epochs = np.arange(1, len(training_loss) + 1)

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color='tab:blue')
ax1.plot(epochs, training_loss, label='Training Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Validation Accuracy', color='tab:orange')
ax2.plot(epochs, validation_accuracy, label='Validation Accuracy', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout()
plt.title('Learning Curve')
plt.savefig('learning_curve.png')
plt.show()
