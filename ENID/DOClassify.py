import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

def network_train(network, X_train, Y_train, batch_size, num_epochs, evaluate, version_directory):
    
    # Initialisation
    optimizer = optim.Adam(network.parameters(), lr=0.01)
    
    epoch_loss = []

    # Training
    for epoch in range(num_epochs+1):

        permutation = torch.randperm(X_train.size()[0])
        batch_loss = 0
        count = 0

        for i in range(0,X_train.size()[0], batch_size):

            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]

            batch_x, batch_y = X_train[indices], Y_train[indices]

            outputs = network(batch_x.float())

            loss = evaluate(outputs, batch_y.float())

            batch_loss += loss.item()
            count += 1

            loss.backward()

            optimizer.step()

        epoch_loss.append(batch_loss/count)
        torch.save(network.state_dict(), version_directory+'/Weights_'+str(epoch))

        if epoch % 5 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, batch_loss/count))
        elif epoch == 69:
            print("Epoch: %d, Nice, loss: %1.5f" % (epoch, batch_loss/count))

#    plt.figure()
#    plt.plot(epoch_loss, linewidth=3)
#    plt.xlabel('Epoch')
#    plt.ylabel('Cross-Entropy Loss')
#    plt.title('Loss Evolution During Training')

#    plt.savefig(version_directory+'/TrainingLoss.png')
    
    return network, epoch_loss

def network_evaluation(network, X_test, Y_test, train_loss, evaluate, version_directory, train_time):
    
    y_true = np.argmax(Y_test.detach().numpy(), axis = 1)
    f1_epoch = []
     
    for i in range(len(train_loss)):
        
        network.load_state_dict(torch.load(version_directory+'/Weights_'+str(i)))

        outputs = network(X_test.float())
        y_pred = np.argmax(outputs.detach().numpy(), axis = 1)
        
        f1_epoch.append(f1_score(y_true, y_pred, average='weighted'))

    index = f1_epoch.index(max(f1_epoch))
    print(index)
    network.load_state_dict(torch.load(version_directory+'/Weights_'+str(index)))
    os.rename(version_directory+'/Weights_'+str(index), version_directory+'/Weights_FINAL')
    
    outputs = network(X_test.float())
    y_pred = np.argmax(outputs.detach().numpy(), axis = 1)
        
    f1 = f1_score(y_true, y_pred, average='weighted')
      
    loss = evaluate(outputs, Y_test).item()
    accuracy = accuracy_score(y_true, y_pred)
    
    C = confusion_matrix(y_true, y_pred, normalize='true')

    print('Train Loss : ' + str(train_loss[index]))
    print('Test Loss : ' + str(loss))
    print('Global Accuracy : ' + str(round(accuracy*100,2)) + '%')
    print('F1 Measure : ' + str(round(f1,2)))

    df = pd.DataFrame({'Metrics': ['Train Loss', 'Test Loss', 'F1 Measure', 'Global Accuracy', 'Training Time'], 'Values': np.array([train_loss[index], loss, round(accuracy*100,2), round(f1,2), train_time])})
    df.to_csv(version_directory+'/Metrics')
    
    C = np.round(C, 2)

    plt.figure(figsize=(15,10))
    ax = plt.subplot()
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels', fontsize=20);
    ax.set_ylabel('True labels', fontsize=20); 
    ax.set_title('Confusion Matrix for Predictions on Test Data', fontsize=25); 
    ax.xaxis.set_ticklabels(['CV', 'SN II', 'SN Ia', 'SLSN-I', 'SN IIn', 'SN Ib/c'], fontsize=15); 
    ax.yaxis.set_ticklabels(['CV', 'SN II', 'SN Ia', 'SLSN-I', 'SN IIn', 'SN Ib/c'], fontsize=15);

    plt.savefig(version_directory+'/ConfusionMatrix.png')

    plt.figure(figsize=(10,6))
    plt.plot(np.arange(0, len(train_loss)), train_loss, linewidth=3, label='Loss Evolution')
    plt.axvline(index, 0, 100, linewidth=3, color='r', label='Selected Model')
    plt.legend()
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Cross-Entropy Loss', fontsize=20)
    plt.title('Loss Evolution During Training', fontsize=25)

    plt.savefig(version_directory+'/TrainingLoss.png')
