"""
CAP 5404 Deep Learning for Computer Graphics
Project II. Neural Networks & Computer Graphics

Pranath Reddy Kumbam (UFID: 8512-0977)

Plot training loss
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
sns.set_theme(style="darkgrid")


# Import loss arrays
loss1 = np.load('CNN_Regressor_train_loss_Faces_ReLU.npy',allow_pickle=True)
loss2 = np.load('CNN_Regressor_train_loss_Faces_Tanh.npy',allow_pickle=True)
loss3 = np.load('CNN_Regressor_train_loss_NCD_ReLU.npy',allow_pickle=True)
loss4 = np.load('CNN_Regressor_train_loss_NCD_Tanh.npy',allow_pickle=True)
loss5 = np.load('DCA_train_loss_Faces_Sigmoid.npy',allow_pickle=True)
loss6 = np.load('DCA_train_loss_Faces_Tanh.npy',allow_pickle=True)
loss7 = np.load('DCA_train_loss_NCD_Sigmoid.npy',allow_pickle=True)
loss8 = np.load('DCA_train_loss_NCD_Tanh.npy',allow_pickle=True)
loss9 = np.load('MLP_Regressor_train_loss_Faces.npy',allow_pickle=True)

# Plots
epochs1 = [i for i in range(1, loss1.shape[0]+1)]
sns.lineplot(epochs1, loss1, label='ReLU')
plt.ylim(0,loss1[0])
plt.xlabel('Training Epochs')
plt.ylabel('MSE Loss')
plt.title('Regressor (Faces Dataset) ReLU')
plt.savefig('./Reg_Face_Loss1.png', format='png', dpi=300)
plt.close()

epochs2 = [i for i in range(1, loss2.shape[0]+1)]
sns.lineplot(epochs2, loss2, label='Tanh')
plt.ylim(0,loss2[0])
plt.xlabel('Training Epochs')
plt.ylabel('MSE Loss')
plt.title('Regressor (Faces Dataset) Tanh')
plt.savefig('./Reg_Face_Loss2.png', format='png', dpi=300)
plt.close()

epochs3 = [i for i in range(1, loss3.shape[0]+1)]
sns.lineplot(epochs3, loss3, label='ReLU')
plt.xlabel('Training Epochs')
plt.ylabel('MSE Loss')
plt.title('Regressor (NCD Dataset) ReLU')
plt.savefig('./Reg_NCD_Loss1.png', format='png', dpi=300)
plt.close()

epochs4 = [i for i in range(1, loss4.shape[0]+1)]
sns.lineplot(epochs4, loss4, label='Tanh')
plt.xlabel('Training Epochs')
plt.ylabel('MSE Loss')
plt.title('Regressor (NCD Dataset) Tanh')
plt.savefig('./Reg_NCD_Loss2.png', format='png', dpi=300)
plt.close()

epochs5 = [i for i in range(1, loss5.shape[0]+1)]
sns.lineplot(epochs5, loss5, label='Sigmoid')
plt.xlabel('Training Epochs')
plt.ylabel('MSE Loss')
plt.title('Colorizer (Faces Dataset) Sigmoid')
plt.savefig('./Color_Face_Loss1.png', format='png', dpi=300)
plt.close()

epochs6 = [i for i in range(1, loss6.shape[0]+1)]
sns.lineplot(epochs6, loss6, label='Tanh')
plt.xlabel('Training Epochs')
plt.ylabel('MSE Loss')
plt.title('Colorizer (Faces Dataset) Tanh')
plt.savefig('./Color_Face_Loss2.png', format='png', dpi=300)
plt.close()

epochs7 = [i for i in range(1, loss7.shape[0]+1)]
sns.lineplot(epochs7, loss7, label='Sigmoid')
plt.xlabel('Training Epochs')
plt.ylabel('MSE Loss')
plt.title('Colorizer (NCD Dataset) Sigmoid')
plt.savefig('./Color_NCD_Loss1.png', format='png', dpi=300)
plt.close()

epochs8 = [i for i in range(1, loss8.shape[0]+1)]
sns.lineplot(epochs8, loss8, label='Tanh')
plt.xlabel('Training Epochs')
plt.ylabel('MSE Loss')
plt.title('Colorizer (NCD Dataset) Tanh')
plt.savefig('./Color_NCD_Loss2.png', format='png', dpi=300)
plt.close()

epochs9 = [i for i in range(1, loss9.shape[0]+1)]
sns.lineplot(epochs9, loss9, label='MLP')
plt.xlabel('Training Epochs')
plt.ylabel('MSE Loss')
plt.title('Regression Fully Connected Model (Faces Dataset)')
plt.savefig('./Reg_Face_Loss_MLP.png', format='png', dpi=300)
plt.close()
