
import matplotlib.pyplot as plt
import numpy as np


history_data = np.load('cnn_history.npz')
tr_acc, tr_loss, val_acc, val_loss = history_data['tr_acc'], history_data['tr_loss'], history_data['val_acc'], history_data['val_loss']
tr_acc = tr_acc[:30]
tr_loss = tr_loss[:30]
val_acc = val_acc[:30]
val_loss = val_loss[:30]

loss_train = min(tr_loss)
accuracy_train = max(tr_acc)

print('Log Loss and Accuracy on Train Dataset:')
print("Loss: {}".format(loss_train))
print("Accuracy: {}".format(accuracy_train))
print()

loss_test = min(val_loss)
accuracy_test = max(val_acc)

print('\nLog Loss and Accuracy on Test Dataset:')
print("Loss: {}".format(loss_test))
print("Accuracy: {}".format(accuracy_test))
print()

plt.clf()
plt.plot(tr_acc, label='training accuracy')
plt.plot(val_acc, label='testing accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.ylim(ymin=0.6, ymax=1.1)
plt.savefig("cnn_accuracy.png", type="png", dpi=300)

plt.clf()
plt.plot(tr_loss, label='training loss')
plt.plot(val_loss, label='testing loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig("cnn_loss.png", type="png", dpi=300)

plt.clf()
