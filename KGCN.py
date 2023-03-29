import random
import pickle

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from model import KGCN
from data_loader import DataLoader


class KGCNDataset(torch.utils.data.Dataset):
    """
    Dataset class: dataset object from the given dataframe
    """
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label

def prepare_arguments(arguments=None):
    """
    prepare arguments (hyperparameters)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use, sum, concat, or neighbor')
    parser.add_argument('--mixer', type=str, default='attention', help='which mixer to use, attention or transe?')
    parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed')
    arguments = arguments.split() if arguments else None
    return parser.parse_args(arguments)

def experiment(arguments=None):
    """
    Conduct an experiment by arguments passing from function call or from CLI
    :return: loss_list, test_loss_list, auc_score_list
    """
    # Initialization
    args = prepare_arguments(arguments)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Data Preparation: build dataset and knowledge graph
    data_loader = DataLoader(args.dataset)
    kg = data_loader.load_kg()
    df_dataset = data_loader.load_dataset()

    # Dataset Splitting
    x_train, x_test, y_train, y_test = train_test_split(
    df_dataset, df_dataset['label'],  # x are triplets; a triplet is (user, item, label)
    test_size=1 - args.ratio, shuffle=False, random_state=999)
    train_dataset = KGCNDataset(x_train)
    test_dataset = KGCNDataset(x_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    # Model Preparation: network, loss function, optimizer
    num_user, num_entity, num_relation = data_loader.get_num()
    user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    criterion = torch.nn.BCELoss()  # binary cross entropy loss
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    print(f'[Info] Using {device}.')

    # Results
    loss_list = []
    test_loss_list = []
    auc_score_list = []

    # Training
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset the gradients of all the parameters in the optimizer to zero
            outputs = net(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()  # Calculate the gradients of the parameters
            
            optimizer.step()  # Update the parameters using the gradients

            running_loss += loss.item()
        
        # Print train loss
        print('[Epoch {}]train_loss: '.format(epoch+1), running_loss / len(train_loader))
        loss_list.append(running_loss / len(train_loader))
            
        # Evaluation
        with torch.no_grad():  # Disabling gradient calculation
            test_loss = 0
            total_roc = 0
            for user_ids, item_ids, labels in test_loader:  # For each row in the data
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids)
                test_loss += criterion(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            print('[Epoch {}]test_loss: '.format(epoch+1), test_loss / len(test_loader))
            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))
    
    return loss_list, test_loss_list, auc_score_list

if __name__ == '__main__':
    arguments = '--dataset product --mixer transe --n_epochs 10 --batch_size=16 --l2_weight 1e-4'
    loss_list, test_loss_list, auc_score_list = experiment(arguments)

    # plot losses / scores
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,6))
    ax1.plot(loss_list)
    ax1.plot(test_loss_list)
    ax2.plot(auc_score_list)
    plt.tight_layout()
    plt.show()

    # store loss_list, test_loss_list, auc_score_list
    result = [loss_list, test_loss_list, auc_score_list]
    with open('result.pkl', 'wb') as f:  # open a text file
        pickle.dump(result, f) # serialize the result