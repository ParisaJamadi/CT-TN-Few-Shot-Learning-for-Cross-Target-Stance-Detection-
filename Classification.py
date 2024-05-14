#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import csv
import copy
import os
import shutil
import torch
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict
from gensim.models import KeyedVectors
from imblearn.over_sampling import RandomOverSampler
from pylab import rcParams
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
from utils import *


# In[2]:


#region Setup

def get_run_modes():
    ##---------------------ï´¾ Run Mode ï´¿---------------------##
    '''available modes:
        -â˜… 'gr'       for only graph embedding :: [graph]
        -â˜… 'friend-gr-ta'    for friends graph embedding + sentence embedding (only target) :: [graph] + [target]
        -â˜… 'like-gr-ta'    for likes graph embedding + sentence embedding (only target) :: [graph] + [target]
        -â˜… 'follower-gr-ta'    for followers graph embedding + sentence embedding (only target) :: [graph] + [target]
        -â˜… 'gr-tw'    for graph embedding + sentence embedding (only tweet) :: [graph] + [tweet]
        -â˜… 'tw'       for sentence embedding (only tweet) :: [tweet] => Sentiment Analysis
        -â˜… 'tw-ta'    for sentence embedding (tweet + target) :: [tweet+target]
        -â˜… 'gr-tw-ta' for graph embedding + sentence embedding (tweet + target) :: [graph] + [tweet+target]
        -â˜… 'like-tw-ta' for like graph embedding + sentence embedding (tweet + target) :: [graph] + [tweet+target]
        -â˜… 'friend-tw-ta' for freind graph embedding + sentence embedding (tweet + target) :: [graph] + [tweet+target]
        -â˜… 'friend-like-tw-ta' for like+ freind graph embedding + sentence embedding (tweet + target) :: [graph] + [tweet+target]
    '''
    run_modes = ['like-gr-ta', 'friend-gr-ta', 'follower-gr-ta', 'tw-ta']


    return run_modes


def define_configuration_general():
    ##--------------------ï´¾ General ï´¿--------------------##
    cfg_gn = {
        'number_of_classes': 2,    # either 2 (to remove 'NONE') or 3 (to include 'NONE')
        'data_source': 'gdrive',   # 'local' or 'gdrive'
        'data_path_gdrive': '/P_stance/',
        'data_path_local':  './data/',
        'data_file_tweet':  'Multi_data_TweetID-UserID.csv',
        'data_file_ge':     'Pstance_friends_model.npz'
        #'data_file_like_ge':  'Multi_likes_embeddings.npz'
    }

    return cfg_gn


def define_configuration_language_model():
    ##-----------------ï´¾ Language Model ï´¿-----------------##
    '''parameters:
        -â˜… model         :: str:     'bert' | 'roberta' | 'roberta-large'
        -â˜… token_length  :: int/str: 50 | 'avg' | 'max' | 'min'
        -â˜… case_sensitive:: bool:    True | False
        -â˜… return_dict   :: bool:    True | False
    '''
    cfg_lm = retrieve_lm(model='roberta', token_length=90)
    # cfg_lm = retrieve_lm(model='roberta-large', token_length='max')
    #cfg_lm = retrieve_lm(model='bert', token_length=90, return_dict=False)

    return cfg_lm


def define_configuration_data_split():
    ##-----------------ï´¾ Data Splitting ï´¿-----------------##
    cfg_ds = {
        'in-target':        False, # whether to consider in-target approach for data splitting & sampling
        'representative':   True,  # whether to pick representative test split from data
        'oversampling'    : False,  # whether to oversample minority classes to mitigate class imbalance in train data (mutually exclusive with 'weighted_sampler')
        'weighted_sampler': False, # whether to use weighted sampler for data sampling to mitigate class imbalance in train data (mutually exclusive with 'oversampling')
        'ratio_test':       0.2,   # ratio of test/train data (0.2 = 20% test, 80% train) :: only used if in-target=True
        'ratio_valid':      0.2    # ratio of valid/train data (0.2 = 20% validation, 80% train)
    }

    return cfg_ds


def define_configuration_neural_network():
    ##-----------ï´¾ Neural Network Architecture ï´¿-----------##
    cfg_nn = {
        'n_hidden_layers': 1, # number of hidden layers in the network (the larger the number, the more complex the model)
        'use_normalization': True,
        'use_dropout': True,
        'dropout_val': 0.2,
        'use_activation_fn': True,
        'activation_fn': nn.ReLU()    # nn.LeakyReLU(), nn.SELU(), ...
    }

    return cfg_nn


def define_tweet_preprocessing():
    ##--------------ï´¾ Tweet Preprocessing ï´¿--------------##
    cfg_tw_preprocessing = {
        'enabled': False, # whether to preprocess tweet text
        'options': {
            'mentions': {
                'enabled':    True, # whether to remove mentions from tweets :: (sth @username sth-else -> sth username sth-else) or (sth @username sth-else -> sth sth-else)
                'keep_after': False # whether to keep the phrase that comes after '@' sign
            },
            'hashtags': {
                'enabled':    True, # whether to handle hashtags in tweets :: example: (... #hashtag -> ... hashtag) or (... #hashtag -> ...)
                'keep_after': True  # whether to keep the phrase that comes after '#' sign
            },
            'brackets': {
                'enabled':     True, # whether to handle brackets in tweets :: example: ()... this [sth] has ... -> ... this sth has ...) or (... this [sth] has ... -> ... this has ...)
                'keep_inside': True  # whether to keep the phrase that is inside brackets
            },
            'emojis': {
                'enabled': True,  # whether to handle emojis in tweets :: example: (... ðŸ˜€ ... -> ... (grinning face) ...) or (... ðŸ˜€ ... -> ... ...)
                'replace': True,  # whether to replace emoji with its description (if False, emoji is removed from tweet)
            },
            'hyperlinks':  True,  # whether to remove hyperlinks from tweets :: example: https://t.co/ -> ...
            'lowercase':   True, # whether to lowercase tweets :: example: 'Hello World!' -> 'hello world!'
            'punctuation': False, # whether to remove punctuation from tweets :: example: ... ! -> ...
            'numbers': {
                'enabled': False,  # whether to handle numbers in tweets :: example: (... 1,000,000 ... -> ... one million ...) or (... 1,000,000 ... -> ... ...)
                'replace': False,  # whether to replace numbers with their word form (if False, numbers are removed from tweet)
            },
            'replacements': {
                'enabled': True,
                'dict': {         # replace words with other words :: example: 'hello' -> 'hi'
                    'hilary': 'hillary',
                    'hillaryclinton': 'hillary clinton',
                    'Hilary': 'Hillary',
                    'HillaryClinton': 'Hillary Clinton',
                    'donaldtrump': 'donald trump',
                    'DonaldTrump': 'Donald Trump',
                    ' RT ': '',
                    'RT ': '',
                    ' rt ': '',
                    'rt ': '',
                    ' @ ': ' at ',
                    ' # ': ' number '
                }
            }
        }
    }

    return cfg_tw_preprocessing


def define_training_parameters(run_mode):
    ##---------------ï´¾ Training Parameters ï´¿---------------##
    cfg_train = {}

    if run_mode == 'tw-ta':
        cfg_train = {
            'batch_size': 128,
            'epochs': 10,
            'optimizer': 'adamw',  # 'adam' | 'adamw' | 'sgd'
            'lr': 3e-5,            # learning rate: 1e-3/1e-5/3e-5:          for bert:  3e-4, 1e-4, 5e-5, 3e-5
            'momentum': 0.9,       # momentum for SGD
            'weight_decay': 1e-4,      #1e-4|1e-6
            'eps': 1e-8,
            'loss_fn': nn.CrossEntropyLoss(), # use CrossEntropy loss, given we have a classification problem
            'num_workers': 0,
        }
    else:
        cfg_train = {
            'batch_size': 128,
            'epochs': 95,
            'optimizer': 'sgd',  # 'adam' | 'adamw' | 'sgd'
            'lr': 1e-2,            # learning rate: 1e-3/1e-5/3e-5:          for bert:  3e-4, 1e-4, 5e-5, 3e-5
            'momentum': 0.9,       # momentum for SGD
            'weight_decay': 1e-4,      #1e-4|1e-6
            'eps': 1e-8,
            'loss_fn': nn.CrossEntropyLoss(), # use CrossEntropy loss, given we have a classification problem
            'num_workers': 0,
        }
        
    return cfg_train


# In[3]:


def set_determinism(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def setup_tokenizer(cfg_lm, cfg_tw_preprocessing, df):
    sample_tokenizer = cfg_lm['tokenizer']

    max_len = 0
    min_len = float('inf')
    sum_len = 0

    all_tweets = df['tweet_cleaned'] if cfg_tw_preprocessing['enabled'] else df['tweet']
    for text in all_tweets:
        # tokenize the text and add special tokens i.e `[CLS]` and `[SEP]`
        input_ids = sample_tokenizer.encode(text, add_special_tokens=True)
        input_ids_len = len(input_ids)

        # update the maximum & minimum sentence length
        max_len = max(max_len, input_ids_len)
        min_len = min(min_len, input_ids_len)
        sum_len += input_ids_len


    tokens_avg = round(sum_len / len(df), 2)
    tokens_min = min_len
    tokens_max = max_len

    print(f'Min length for input tokens: {tokens_min}')
    print(f'Max length for input tokens: {tokens_max}')
    print(f'Avg length for input tokens: {tokens_avg}')

    if type(cfg_lm['token_length']) == int:
        token_max_len = cfg_lm['token_length']
    else:
        if cfg_lm['token_length'] == 'max':
            token_max_len = tokens_max
        elif cfg_lm['token_length'] == 'min':
            token_max_len = tokens_min
        elif cfg_lm['token_length'] == 'avg':
            token_max_len = int(tokens_avg)
        else:
            raise ValueError(f'Error: Invalid value for token_length: {cfg_lm["token_length"]}')


    cfg_sentence_embedding = {
        'tokenizer': cfg_lm['tokenizer'],
        'tokenizer_max_len': token_max_len,
        'embedding_size': cfg_lm['embedding_size'],
        'tweet_column_name': 'tweet_cleaned' if cfg_tw_preprocessing['enabled'] else 'tweet'
    }

    return cfg_sentence_embedding

#endregion

#region StanceDataset Class and Related Functions

class StanceDataset(Dataset):
    def __init__(self, dataframe, sentence_embedding_settings, friend_graph_embedding_settings, like_graph_embedding_settings , follower_graph_embedding_settings ):
    
        self.tokenizer = sentence_embedding_settings['tokenizer']
        self.tokenizer_max_len = sentence_embedding_settings['tokenizer_max_len']

        self.friend_graph_embeddings = friend_graph_embedding_settings['graph_embeddings']
        self.friend_graph_keys = friend_graph_embedding_settings['graph_key_indices_dict']

        self.like_graph_embeddings = like_graph_embedding_settings['graph_embeddings']
        self.like_graph_keys = like_graph_embedding_settings['graph_key_indices_dict']
        
        self.follower_graph_embeddings = follower_graph_embedding_settings['graph_embeddings']


        self.tweets   = dataframe[sentence_embedding_settings['tweet_column_name']].to_numpy()
        self.targets  = dataframe['target'].to_numpy()
        self.stances  = dataframe['stance'].to_numpy()
        self.user_ids = dataframe['user_id'].to_numpy()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, item):
        tweet  = str(self.tweets[item])
        #target = str(self.targets[item]).lower()
        target = str(self.targets[item])
        friend_g_embedding = self.friend_graph_embeddings[self.friend_graph_keys[f'{int(self.user_ids[item])}']]
        like_g_embedding = self.like_graph_embeddings[self.like_graph_keys[f'{int(self.user_ids[item])}']]
        follower_g_embedding = self.follower_graph_embeddings[f'{int(self.user_ids[item])}']

        
        stance = self.stances[item]

        tweet_encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.tokenizer_max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        target_encoding = self.tokenizer.encode_plus(
            target,
            add_special_tokens=True,
            max_length=self.tokenizer_max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        mixed_encoding = self.tokenizer.encode_plus(
            tweet,
            target,
            add_special_tokens=True,
            max_length=self.tokenizer_max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation='only_first',
            return_attention_mask=True,
            return_tensors='pt',
        )


        item_input = {
            'tweet_token_ids': tweet_encoding['input_ids'].flatten(),
            'tweet_attention_mask': tweet_encoding['attention_mask'].flatten(),

            'target_token_ids': target_encoding['input_ids'].flatten(),
            'target_attention_mask': target_encoding['attention_mask'].flatten(),

            'mixed_token_ids': mixed_encoding['input_ids'].flatten(),
            'mixed_attention_mask': mixed_encoding['attention_mask'].flatten(),

            'friend_g_embedding': torch.tensor(friend_g_embedding, dtype=torch.float),
            'like_g_embedding': torch.tensor(like_g_embedding, dtype=torch.float),
            'follower_g_embedding': torch.tensor(follower_g_embedding, dtype=torch.float)
        }
        item_output = torch.tensor(stance, dtype=torch.long)

        return (item_input, item_output)


# In[5]:


# data loader function
def create_data_loader(dataframe, se_settings, friend_ge_settings, like_ge_settings, follower_ge_settings, batch_size=32, shuffle=False, sampler=None, num_workers=0):
    ds = StanceDataset(
        dataframe=dataframe,
        sentence_embedding_settings=se_settings,
        friend_graph_embedding_settings=friend_ge_settings,
        like_graph_embedding_settings=like_ge_settings,
        follower_graph_embedding_settings=follower_ge_settings
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers
    )

#endregion


#region StanceClassifier Class

class StanceClassifier(nn.Module):
    def __init__(self, n_classes, mode, lm_options, friend_ge_options, like_ge_options, follower_ge_options, nn_options):
        super(StanceClassifier, self).__init__()

        self.n_classes = n_classes
        self.mode = mode

        self.lm = lm_options['model']
        self.lm.eval()  # important: evaluation mode must be enabled for language model, to prevent re-training model on each call

        self.ge_size = friend_ge_options['embedding_size'] # graph embedding size
        self.se_size = lm_options['embedding_size'] # sentence embedding size


        self.mx_size = self.ge_size + self.se_size  # mixed size (graph embedding + sentence embedding)
        self.all_mx_size = self.ge_size + self.ge_size + self.se_size # mixed size (graph embedding + sentence embedding)
        
        self.all_3graphs_size = self.ge_size + self.ge_size + self.ge_size+ self.se_size # mixed size (3 graph embeddings: friend,like, and followers

        self.normalizer_enabled  = nn_options['use_normalization']
        self.normalizer_graph    = nn.LayerNorm(self.ge_size)
        self.normalizer_sentence = nn.LayerNorm(self.se_size)

        self.n_hidden_layers = nn_options['n_hidden_layers']
        self.use_dropout = nn_options['use_dropout']
        self.dropout_val = nn_options['dropout_val']
        self.use_activation_fn = nn_options['use_activation_fn']
        self.activation_fn = nn_options['activation_fn']


        self.nn_only_graph = nn.Sequential(
            # hidden layer(s)
            self.generate_hidden_unit(self.n_hidden_layers, self.ge_size),
            # output layer
            nn.Linear(self.ge_size, self.n_classes),
        )

        self.nn_only_sentence = nn.Sequential(
            # hidden layer(s)
            self.generate_hidden_unit(self.n_hidden_layers, self.se_size),
            # output layer
            nn.Linear(self.se_size, self.n_classes),
        )

        self.nn_mixed = nn.Sequential(
            # hidden layer(s)
            self.generate_hidden_unit(self.n_hidden_layers, self.mx_size),
            # output layer
            nn.Linear(self.mx_size, self.n_classes),
        )
        self.nn_all_mixed = nn.Sequential(
            # hidden layer(s)
            self.generate_hidden_unit(self.n_hidden_layers, self.all_mx_size),
            # output layer
            nn.Linear(self.all_mx_size, self.n_classes),
        )

        self.nn_graph_mixed = nn.Sequential(
            # hidden layer(s)
            self.generate_hidden_unit(self.n_hidden_layers, self.all_3graphs_size),
            # output layer
            nn.Linear(self.all_3graphs_size, self.n_classes),
        )
    # generate hidden unit
    def generate_hidden_unit(self, n_layers, layer_size):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(layer_size, layer_size))
            if self.use_activation_fn:
                layers.append(self.activation_fn)
            if self.use_dropout:
                layers.append(nn.Dropout(self.dropout_val))
        block = nn.Sequential(*layers)
        return block


    def forward(self, x):
        friend_graph_embedding = x['friend_g_embedding']
        like_graph_embedding = x['like_g_embedding']
        follower_graph_embedding = x['follower_g_embedding']
        
        if self.normalizer_enabled: friend_graph_embedding = self.normalizer_graph(friend_graph_embedding)
        if self.normalizer_enabled: like_graph_embedding = self.normalizer_graph(like_graph_embedding)
        if self.normalizer_enabled: follower_graph_embedding = self.normalizer_graph(follower_graph_embedding)

        # if self.mode == 'gr':
        #     fc = self.nn_only_graph
        #     input_vector = graph_embedding

        if self.mode == 'like-gr-ta':
            _, sentence_embedding = self.lm(
                input_ids=x['target_token_ids'],
                attention_mask=x['target_attention_mask']
            )
            if self.normalizer_enabled: 
                sentence_embedding = self.normalizer_sentence(sentence_embedding)
                fc = self.nn_mixed
                input_vector = torch.cat((like_graph_embedding, sentence_embedding), dim=1)
        
        elif self.mode == 'friend-gr-ta':
            _, sentence_embedding = self.lm(
                input_ids=x['target_token_ids'],
                attention_mask=x['target_attention_mask']
            )
            if self.normalizer_enabled: 
                sentence_embedding = self.normalizer_sentence(sentence_embedding)
                fc = self.nn_mixed
                input_vector = torch.cat((friend_graph_embedding, sentence_embedding), dim=1)
            
        elif self.mode == 'follower-gr-ta':
            _, sentence_embedding = self.lm(
                input_ids=x['target_token_ids'],
                attention_mask=x['target_attention_mask']
            )
            if self.normalizer_enabled: 
                sentence_embedding = self.normalizer_sentence(sentence_embedding)
                fc = self.nn_mixed
                input_vector = torch.cat((follower_graph_embedding, sentence_embedding), dim=1)
            
        
        # elif self.mode == 'gr-tw':
        #     _, sentence_embedding = self.lm(
        #         input_ids=x['tweet_token_ids'],
        #         attention_mask=x['tweet_attention_mask']
        #     )
        #     if self.normalizer_enabled: 
        #         sentence_embedding = self.normalizer_sentence(sentence_embedding)
        #         fc = self.nn_mixed
        #         input_vector = torch.cat((graph_embedding, sentence_embedding), dim=1)

        elif self.mode == 'tw':
            _, sentence_embedding = self.lm(
                input_ids=x['tweet_token_ids'],
                attention_mask=x['tweet_attention_mask']
            )
            if self.normalizer_enabled: 
                sentence_embedding = self.normalizer_sentence(sentence_embedding)
                fc = self.nn_only_sentence
                input_vector = sentence_embedding

        elif self.mode == 'tw-ta':
            _, sentence_embedding = self.lm(
                input_ids=x['mixed_token_ids'],
                attention_mask=x['mixed_attention_mask']
            )
            if self.normalizer_enabled: 
                sentence_embedding = self.normalizer_sentence(sentence_embedding)
                fc = self.nn_only_sentence
                input_vector = sentence_embedding

        elif self.mode == 'like-tw-ta':
            _, sentence_embedding = self.lm(
                input_ids=x['mixed_token_ids'],
                attention_mask=x['mixed_attention_mask']
            )
            if self.normalizer_enabled: 
                sentence_embedding = self.normalizer_sentence(sentence_embedding)
                fc = self.nn_mixed
                input_vector = torch.cat((like_graph_embedding, sentence_embedding), dim=1)

        elif self.mode == 'friend-tw-ta':
            _, sentence_embedding = self.lm(
                input_ids=x['mixed_token_ids'],
                attention_mask=x['mixed_attention_mask']
            )
            if self.normalizer_enabled: 
                sentence_embedding = self.normalizer_sentence(sentence_embedding)
                fc = self.nn_mixed
                input_vector = torch.cat((friend_graph_embedding, sentence_embedding), dim=1)

        elif self.mode == 'friend-like-tw-ta':
            # _, sentence_embedding = self.lm(
            #     input_ids=x['tweet_token_ids'],
            #     attention_mask=x['tweet_attention_mask']
            # )
            _, sentence_embedding = self.lm(
                input_ids=x['mixed_token_ids'],
                attention_mask=x['mixed_attention_mask']
            )
            # _, target_embedding = self.lm(
            #     input_ids=x['target_token_ids'],
            #     attention_mask=x['target_attention_mask']
            # )
            if self.normalizer_enabled: 
                sentence_embedding = self.normalizer_sentence(sentence_embedding)
                target_embedding = self.normalizer_sentence(target_embedding)
                fc = self.nn_all_mixed
                input_vector = torch.cat((friend_graph_embedding, like_graph_embedding , sentence_embedding), dim=1)
            
        elif self.mode == 'friend-like-follower':
            _, sentence_embedding = self.lm(
                input_ids=x['target_token_ids'],
                attention_mask=x['target_attention_mask']
            )
            if self.normalizer_enabled: sentence_embedding = self.normalizer_sentence(sentence_embedding)
            fc = self.nn_graph_mixed
            input_vector = torch.cat((friend_graph_embedding, like_graph_embedding , follower_graph_embedding ,sentence_embedding), dim=1)

        else:
            raise ValueError('Wrong mode provided. Please check mode value.')


        output = fc(input_vector)
        return output

#endregion


# In[6]:


#region Model Functions and Related Functions
    
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


# In[7]:


"""## Model Train Step"""

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()

    losses = []
    predictions = []
    labels = []

    for i, (train_inputs, train_labels) in enumerate(data_loader):
        for input_tensor in train_inputs:
            train_inputs[input_tensor] = train_inputs[input_tensor].to(device)
        train_labels = train_labels.to(device)

        optimizer.zero_grad()
        output = model(train_inputs)

        loss = loss_fn(output, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)  # prevent exploding gradients
        optimizer.step()

        _, preds = torch.max(output, dim=1)
        losses.append(loss.item())
        predictions.append(preds.cpu().numpy())
        labels.append(train_labels.data.cpu().numpy())

    # calculate accuracy & f1 score
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)

    return np.mean(losses), acc, f1

"""## Model Evaluation Step"""

def eval_model(model, data_loader, loss_fn, device):
    model.eval()

    losses = []
    predictions = []
    labels = []

    with torch.no_grad():
        for i, (test_inputs, test_labels) in enumerate(data_loader):
            for input_tensor in test_inputs:
                test_inputs[input_tensor] = test_inputs[input_tensor].to(device)
            test_labels = test_labels.to(device)

            output = model(test_inputs)
            loss = loss_fn(output, test_labels)

            _, preds = torch.max(output, dim=1)
            losses.append(loss.item())
            predictions.append(preds.cpu().numpy())
            labels.append(test_labels.data.cpu().numpy())

    # calculate accuracy & f1 score
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)

    return np.mean(losses), acc, f1


def get_predictions(device, model, data_loader):
    model.eval()
    
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for i, (test_inputs, test_labels) in enumerate(data_loader):
            for input_tensor in test_inputs:
                test_inputs[input_tensor] = test_inputs[input_tensor].to(device)
            test_labels = test_labels.to(device)

            output = model(test_inputs)
            _, preds = torch.max(output, dim=1)
            # probs = F.softmax(outputs, dim=1)
            
            predictions.extend(preds)
            prediction_probs.extend(output)
            real_values.extend(test_labels)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, prediction_probs, real_values

#endregion


# In[8]:


def main():
    # Folder paths
    data_folder_location = "/CT-TN_Project/"
    output_folder_location = "/output_job/"

    # PARAMETERS #
    target_pairs = ["Trump_Sanders", "Trump_Biden", "Sanders_Trump", "Sanders_Biden", "Biden_Trump"]
    class_name_strings = {"Trump": "Donald Trump", "Sanders": "Bernie Sanders", "Biden": "Joe Biden"}
    run_modes = get_run_modes()
    shot_numbers = {
        "Trump_Sanders": {100: 250, 200: 430, 300: 620, 400: 870},
        "Trump_Biden": {100: 230, 200: 410, 300: 640, 400: 870},
        "Sanders_Trump": {100: 230, 200: 420, 300: 630, 400: 850},
        "Sanders_Biden": {100: 240, 200: 420, 300: 640, 400: 860},
        "Biden_Trump": {100: 250, 200: 430, 300: 670, 400: 870}
    }
    sample_sizes = {
        "Trump_Sanders": {100: 430, 200: 430, 300: 430, 400: 340},
        "Trump_Biden": {100: 450, 200: 450, 300: 450, 400: 370},
        "Sanders_Trump": {100: 450, 200: 450, 300: 330, 400: 230},
        "Sanders_Biden": {100: 450, 200: 450, 300: 450, 400: 380},
        "Biden_Trump": {100: 450, 200: 450, 300: 380, 400: 280}
    }
    random_seeds = [24, 524, 1024, 1524, 2024]

    # Fixed parameters
    cfg_gn = define_configuration_general()
    cfg_lm = define_configuration_language_model()
    cfg_ds = define_configuration_data_split()
    cfg_nn = define_configuration_neural_network()
    cfg_tw_preprocessing = define_tweet_preprocessing()
    print()  # Print line for readability

    # Read in likes embeddings
    print(f"Getting likes embeddings")

    like_npz = np.load(f"{data_folder_location}likes_model.npz" , allow_pickle=True)
    like_embeds = {item: like_npz[item] for item in like_npz.files}
    like_word2vec_object = like_embeds['arr_0'].tolist()
    like_graph_keys_str_list = like_word2vec_object.index_to_key
    like_graph_keys = [int(item) for item in like_graph_keys_str_list]
    print(f"\tNumber of graph keys: {len(like_graph_keys)}")

    like_graph_key_indices_dict = like_word2vec_object.key_to_index
    like_graph_embeddings = like_word2vec_object.vectors
    print(f"\tNumber of graph embeddings: {len(like_graph_embeddings)}")

    # Read in friends embeddings
    print(f"Getting friends embeddings")

    friend_npz = np.load(f"{data_folder_location}friends_model.npz" , allow_pickle=True)
    friend_embeds = {item: friend_npz[item] for item in friend_npz.files}
    friend_word2vec_object = friend_embeds['arr_0'].tolist()
    friend_graph_keys_str_list = friend_word2vec_object.index_to_key
    friend_graph_keys = [int(item) for item in friend_graph_keys_str_list]
    print(f"\tNumber of graph keys: {len(friend_graph_keys)}")

    friend_graph_key_indices_dict = friend_word2vec_object.key_to_index
    friend_graph_embeddings = friend_word2vec_object.vectors
    print(f"\tNumber of graph embeddings: {len(friend_graph_embeddings)}")

    # Read in followers embeddings
    print(f"Getting followers embeddings")

    followers_vectors = KeyedVectors.load(f"{data_folder_location}followers_model.pickle")
    followers_graph_keys = [int(item) for item in followers_vectors.index_to_key[:]]
    print(f"\tNumber of graph keys: {len(followers_graph_keys)}")
    print(f"\tNumber of graph embeddings: {len(followers_vectors)}")


    # Check missing keys
    print(f"Checking missing keys")

    #userIds_in_df = [int(item) for item in df['user_id'].unique().tolist()]
    #print(f"\tNumber of users in dataframe: {len(userIds_in_df)}")

    graph_keys = list(
        set(friend_graph_keys) & 
        set(like_graph_keys) & 
        set(followers_graph_keys) 
    )
    print(f"\tNumber of users in embeddings: {len(graph_keys)}")

    #missing_user_ids = list(set(userIds_in_df) - set(graph_keys))
    #print(f"\tNumber of missing users: {len(missing_user_ids)}")

    # Organise embeddings
    friend_cfg_graph_embedding = {
        'graph_embeddings': friend_graph_embeddings,
        'graph_key_indices_dict': friend_graph_key_indices_dict,
        'embedding_size': friend_graph_embeddings.shape[1]
    }

    like_cfg_graph_embedding = {
        'graph_embeddings': like_graph_embeddings,
        'graph_key_indices_dict': like_graph_key_indices_dict,
        'embedding_size': like_graph_embeddings.shape[1]
    }

    follower_cfg_graph_embedding = {
        'graph_embeddings': followers_vectors
        #'graph_key_indices_dict': like_graph_key_indices_dict,
        #'embedding_size': followers_vectors.shape[1]
    }
    


    # CODE START #
    # For each target pair
    for target_pair in target_pairs:
    
        print(f"\nNew target pair: {target_pair}")

        first_target, second_target = target_pair.split('_')
        class_names = [class_name_strings[first_target], class_name_strings[second_target]]

        # Read data
        print(f"\tReading in data")

        df_train_one =pd.read_csv(data_folder_location+f"Train_{first_target}_with_UserIds.csv")
        df_train_two= pd.read_csv(data_folder_location+f"Val_{first_target}_with_UserIds.csv")
        df_train_three= pd.read_csv(data_folder_location+f"Test_{first_target}_with_UserIds.csv")

        df_test_one= pd.read_csv(data_folder_location+f"Train_{second_target}_with_UserIds.csv")
        df_test_two= pd.read_csv(data_folder_location+f"Val_{second_target}_with_UserIds.csv")
        df_test_three= pd.read_csv(data_folder_location+f"Test_{second_target}_with_UserIds.csv")

        # Concatenate data
        print(f"\tConcatenating data")
        df_train_main_name = f"df_train_{target_pair}"
        globals()[df_train_main_name]  = pd.concat([df_train_one, df_train_two, df_train_three])
        globals()[df_train_main_name].reset_index(drop=True, inplace=True)
        df_test_main_name = f"df_test_{target_pair}"
        globals()[df_test_main_name] = pd.concat([df_test_one, df_test_two, df_test_three])
        globals()[df_test_main_name].reset_index(drop=True, inplace=True)

        # Remap columns
        print(f"\tRemapping data")
        column_name_mapping = {globals()[df_train_main_name].columns[1]:'tweet',globals()[df_train_main_name].columns[2]:'target',globals()[df_train_main_name].columns[3]:'stance', globals()[df_train_main_name].columns[4]: 'Tweet_ID', globals()[df_train_main_name].columns[5]:'user_ID'}
        globals()[df_train_main_name] = globals()[df_train_main_name].rename(columns=column_name_mapping)
        column_name_mapping = {globals()[df_test_main_name].columns[1]:'tweet',globals()[df_test_main_name].columns[2]:'target',globals()[df_test_main_name].columns[3]:'stance', globals()[df_test_main_name].columns[4]: 'Tweet_ID', globals()[df_test_main_name].columns[5]:'user_ID'}
        globals()[df_test_main_name] = globals()[df_test_main_name].rename(columns=column_name_mapping)

        # Rename columns
        globals()[df_train_main_name].drop('Unnamed: 0', axis=1, inplace=True)
        globals()[df_test_main_name].drop('Unnamed: 0', axis=1, inplace=True)
        globals()[df_train_main_name].columns = globals()[df_train_main_name].columns.str.lower().str.replace(' ', '_')
        globals()[df_test_main_name].columns = globals()[df_test_main_name].columns.str.lower().str.replace(' ', '_')


        # For each mode
        for cfg_run_mode in run_modes:
            print(f"\t\tNew run mode: {cfg_run_mode}")
            cfg_train = define_training_parameters(cfg_run_mode)


            # For each shot
            for shot_number in shot_numbers[target_pair]:
                print(f"\t\tNew shot: {shot_number}")

                df = None
                
                injection_number =shot_numbers[target_pair][shot_number]
                df_train_add_name = f"df_train_add_{target_pair}{shot_number}"
                df_train_name = f"{df_train_main_name}{shot_number}"
                globals()[df_train_add_name] = globals()[df_test_main_name].iloc[0:injection_number, :]
                globals()[df_train_name] = pd.concat([globals()[df_train_main_name], globals()[df_train_add_name]])
                globals()[df_train_name].reset_index(drop=True, inplace=True)


          
                df_test_name = f"df_test_{target_pair}{shot_number}"
                globals()[df_test_name] = globals()[df_test_main_name].iloc[injection_number:, :]
                globals()[df_test_name].reset_index(drop=True, inplace=True)


                globals()[df_train_name] = globals()[df_train_name].assign(TrainOrTest ='train')
                globals()[df_test_name] = globals()[df_test_name].assign(TrainOrTest ='test')


                df_name = f"df_{target_pair}{shot_number}"
                globals()[df_name] = pd.concat([globals()[df_train_name], globals()[df_test_name]])
                globals()[df_name].reset_index(drop=True, inplace=True)
                df=globals()[df_name]
                print(f'df{cfg_run_mode}{target_pair}',df)

                # Get value counts
                value_counts_target_initial = df.groupby(['TrainOrTest', 'target']).size().unstack(fill_value=0)
                print(f"\t\t\tValue counts initial: {value_counts_target_initial}")

                # Remap stance values (string to int)
                if cfg_gn['number_of_classes'] == 2:
                    df = df.drop(df[df['stance'] == 'NONE'].index).reset_index(drop=True)
                    stance_map_dict = {
                        'AGAINST': 0,
                        'FAVOR': 1
                    }
                else:
                    stance_map_dict = {
                        'AGAINST': 0,
                        'NONE': 1,
                        'FAVOR': 2
                    }
                df['stance'] = df['stance'].map(stance_map_dict)

                # Create a column containing cleaned tweets
                if cfg_tw_preprocessing['enabled']:
                    df['tweet_cleaned'] = df['tweet'].apply(lambda x: preprocess_tweet(x, cfg_tw_preprocessing['options']))

                # Lower the target columns if lowercase option is enabled
                if cfg_tw_preprocessing['enabled'] and cfg_tw_preprocessing['options']['lowercase']:
                    df['target'] = df['target'].str.lower()
                    df['target_two'] = df['target_two'].str.lower()

                # Get value counts
                value_counts_target = df.groupby(['TrainOrTest', 'target']).size().unstack(fill_value=0)
                print(f"\t\t\tValue counts target: {value_counts_target}")

                value_counts_stance = df.groupby(['TrainOrTest', 'stance']).size().unstack(fill_value=0)
                print(f"\t\t\tValue counts stance: {value_counts_stance}")

                value_counts_target_stance = df.groupby(['TrainOrTest', 'target', 'stance']).size().unstack(fill_value=0)
                print(f"\t\t\tValue counts target stance: {value_counts_target_stance}")

                # Remove samples which have a user id not present in graph embedding
                ######################################################################
                print(f"\t\tRemoving missing user")
                userIds_in_df = [int(item) for item in df['user_id'].unique().tolist()]
                print(f"\tNumber of users in dataframe: {len(userIds_in_df)}")
                
                missing_user_ids = list(set(userIds_in_df) - set(graph_keys))
                ######################################################################
                df = df[~df['user_id'].isin(missing_user_ids)]
                df = df.reset_index(drop=True)

                new_userIds_in_df = [int(item) for item in df['user_id'].unique().tolist()]
                print(f"\t\t\tNumber new user IDs: {len(new_userIds_in_df)}")

                value_counts_target_stance_no_missing = df.groupby(['TrainOrTest', 'target', 'stance']).size().unstack(fill_value=0)
                print(f"\t\t\tValue counts target stance: {value_counts_target_stance_no_missing}")

                # For each seed
                for seed in random_seeds:
                    print(f"\t\t\tNew seed: {seed}")

                    # Set seed
                    device = set_determinism(seed)

                    # Setup tokinzer
                    cfg_sentence_embedding = setup_tokenizer(cfg_lm, cfg_tw_preprocessing, df)

                    # Split data
                    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle dataframe
                    #print(df)
                    #print('target_pair shot seed', target_pair, shot_number, seed)

                    if df.TrainOrTest[0] == 'train':
                        train = df['TrainOrTest'].unique()[0]
                        test = df['TrainOrTest'].unique()[1]
                    elif df.TrainOrTest[0] == 'test':
                        train = df['TrainOrTest'].unique()[1]
                        test = df['TrainOrTest'].unique()[0]
                    
                    ## IMPORTANT: Notice the target distribution difference when comparing sample counts for "train" and "test"
                    print(f"\t\t\t\tStance value distribution for test:', {df[df['TrainOrTest'] == test]['stance'].value_counts(normalize=True, sort=False)}")
                    print(f"\t\t\t\tStance value distribution for train:', {df[df['TrainOrTest'] == train]['stance'].value_counts(normalize=True, sort=False)}")

                    # Split dataframe into representative train, validation and test sets
                    df_train_full = df[df['TrainOrTest'] == 'train']
                    df_test_full  = df[df['TrainOrTest'] == 'test']
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    train_indices, valid_indices = next(skf.split(X=df_train_full, y=df_train_full['stance']), 0)  # make sure target distribution remains the same by utilizing StratifiedKFold data split
                    df_train = df_train_full.iloc[train_indices]
                    df_valid = df_train_full.iloc[valid_indices]
                    df_test = None

                    if cfg_ds['representative']:
                        #sample_size = len(df_test_full)  # Set sample_size to the minimum value between length of df_test_full and 400
                        df_test  = data_sampler(df=df_test_full, sample_size=sample_sizes[target_pair][shot_number],
                                                distribution_column=df_train['stance'], target_column_name='stance',
                                                shuffle=True, random_state=seed)
                    else:
                        df_test  = df_test_full.sample(n=len(df_test_full), random_state=seed)  # take a random subset from test data matching the counts of validation samples
                        
                    print(f"\t\t\t\tLength df_train_full: {len(df_train_full)}")
                    print(f"\t\t\t\tLength df_test_full: {len(df_test_full)}")
                    print(f"\t\t\t\tLength df_train: {len(df_train)}")
                    print(f"\t\t\t\tLength df_valid: {len(df_valid)}")

                    # NOTE: lines 796 to 866 omitted

                    # Setup Stance Dataset
                    ds = StanceDataset(
                        dataframe=df,
                        sentence_embedding_settings=cfg_sentence_embedding,
                        friend_graph_embedding_settings=friend_cfg_graph_embedding,
                        like_graph_embedding_settings=like_cfg_graph_embedding,
                        follower_graph_embedding_settings=follower_cfg_graph_embedding
                    )

                    # Deal with class imbalance
                    if cfg_ds['oversampling']:
                        ros = RandomOverSampler(random_state=seed)
                        df_train_resampled, df_train_labels_resampled = ros.fit_resample(df_train, df_train['stance'])
                        print(f"\t\t\t\tTrain Data Resampled: {df_train_resampled['stance'].value_counts(normalize=True)}")

                    # approach 2: weighted sampling
                    # calculate class weights for weighted sampling in each batch, to equalize the class distribution
                    if cfg_ds['weighted_sampler']:
                        class_counts  = []
                        sample_counts = len(df_train['stance'])
                        labels = df_train['stance'].to_list()
                        
                        for i, label in enumerate(class_names):
                            class_counts.append(df_train[df_train['stance'] == i].shape[0])

                        class_weights = [sample_counts/class_counts[i] for i in range(len(class_counts))]
                        weights = [class_weights[labels[i]] for i in range(int(sample_counts))]
                        train_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(sample_counts))


                    # create data loaders
                    if cfg_ds['oversampling']:
                        train_data_loader = create_data_loader(df_train_resampled, cfg_sentence_embedding, friend_cfg_graph_embedding, like_cfg_graph_embedding , follower_cfg_graph_embedding, batch_size=cfg_train['batch_size'], shuffle=True, num_workers=cfg_train['num_workers'])
                    elif cfg_ds['weighted_sampler']:
                        train_data_loader = create_data_loader(df_train, cfg_sentence_embedding, friend_cfg_graph_embedding, like_cfg_graph_embedding,follower_cfg_graph_embedding,  batch_size=cfg_train['batch_size'], shuffle=False, sampler=train_sampler, num_workers=cfg_train['num_workers'])
                    else:
                        train_data_loader = create_data_loader(df_train, cfg_sentence_embedding, friend_cfg_graph_embedding, like_cfg_graph_embedding,follower_cfg_graph_embedding, batch_size=cfg_train['batch_size'], shuffle=True, num_workers=cfg_train['num_workers'])
                        # print('len train_data_loader',len(train_data_loader))

                    valid_data_loader = create_data_loader(
                        df_valid,
                        cfg_sentence_embedding, 
                        friend_cfg_graph_embedding, 
                        like_cfg_graph_embedding,
                        follower_cfg_graph_embedding,
                        batch_size=cfg_train['batch_size'], 
                        num_workers=cfg_train['num_workers']
                    )
                    print(f"\t\t\t\tLength of valid_data_loader {len(valid_data_loader)}")

                    test_data_loader = create_data_loader(
                        df_test,
                        cfg_sentence_embedding, 
                        friend_cfg_graph_embedding, 
                        like_cfg_graph_embedding,
                        follower_cfg_graph_embedding,
                        batch_size=cfg_train['batch_size'], 
                        num_workers=cfg_train['num_workers']
                    )
                    print(f"\t\t\t\tLength of test_data_loader: {len(test_data_loader)}")

                    data = next(iter(train_data_loader))
                    sample_input, sample_output = data[0], data[1]
                    sample_input.keys()

                    # print(sample_input['mixed_token_ids'].shape)
                    # print(sample_input['mixed_attention_mask'].shape)
                    # print(sample_input['friend_g_embedding'].shape)
                    # print(sample_input['like_g_embedding'].shape)
                    # print(sample_output.shape)

                    # Create model
                    model = StanceClassifier(
                        len(class_names), 
                        mode=cfg_run_mode, 
                        lm_options=cfg_lm, 
                        friend_ge_options=friend_cfg_graph_embedding, 
                        like_ge_options=like_cfg_graph_embedding, 
                        follower_ge_options=follower_cfg_graph_embedding,
                        nn_options=cfg_nn
                    )
                    model = model.to(device)

                    # Optimizer
                    optimizer = None

                    if cfg_train['optimizer'] == 'adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train['lr'], weight_decay=cfg_train['weight_decay'], eps=cfg_train['eps'])
                    elif cfg_train['optimizer'] == 'adamw':
                        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train['lr'], weight_decay=cfg_train['weight_decay'])
                    elif cfg_train['optimizer'] == 'sgd':
                        optimizer = optim.SGD(model.parameters(), lr=cfg_train['lr'], momentum=cfg_train['momentum'])
                    else:
                        raise ValueError('Wrong optimizer provided. Please check optimizer value.')

                    # Scheduler
                    total_steps = len(train_data_loader) * cfg_train['epochs']
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=0,
                        num_training_steps=total_steps
                    )

                    # Loss function
                    loss_fn = cfg_train['loss_fn'].to(device)
            
                    # Model Training
                    history = defaultdict(list)
                    best_epoch = 1
                    best_loss = float('inf')
                    best_loss_acc = 0.0
                    best_loss_f1 = 0.0
                    best_wts = copy.deepcopy(model.state_dict())


                    # Training loop
                    print(f"\t\t\t\tStarting training")

                    for epoch in range(cfg_train['epochs']):
                        print(f"Epoch {epoch + 1}/{cfg_train['epochs']}")
                        print('-' * 10)

                        train_loss, train_acc, train_f1 = train_epoch(model, train_data_loader, loss_fn, optimizer, device)
                        print(f'Train Loss: {round(train_loss, 6)}, Accuracy: {round(train_acc, 4)}, F1: {round(train_f1, 4)}')

                        val_loss, val_acc, val_f1 = eval_model(model, valid_data_loader, loss_fn, device)
                        print(f'Valid Loss: {round(val_loss, 6)}, Accuracy: {round(val_acc, 4)}, F1: {round(val_f1, 4)}')
                        print()

                        history['train_loss'].append(train_loss)
                        history['train_acc'].append(train_acc)
                        history['train_f1'].append(train_f1)
                        history['val_loss'].append(val_loss)
                        history['val_acc'].append(val_acc)
                        history['val_f1'].append(val_f1)

                        scheduler.step()
                        
                        # create checkpoint variable and add important data
                        checkpoint = {
                            'epoch': epoch + 1,
                            'valid_loss_min': val_loss,
                            'state_dict': model.state_dict()
                                #'optimizer': optimizer.state_dict(),
                        }
                        
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_loss_acc = val_acc
                            best_loss_f1  = val_f1
                            best_wts = copy.deepcopy(model.state_dict())
                            #best_wts =checkpoint['state_dict']
                            best_epoch = (epoch + 1)
                            #torch.save(model.state_dict(), 'best_model_state.bin')
                            #torch.save(model.state_dict() , PATH + f'best-{cfg_run_mode}-{shot}shot-{RANDOM_SEED}seed_sample_size400.pth')
                            #torch.save(best_wts , PATH + f'Best-{cfg_run_mode}-{RANDOM_SEED}seed_sample_size400.pth')

                    # Load best model and predictions
                    model.load_state_dict(best_wts)

                    y_pred_valid, y_pred_probs_valid, y_valid = get_predictions(device, model, valid_data_loader)
                    y_pred_test,  y_pred_probs_test,  y_test  = get_predictions(device, model, test_data_loader)

                    # Format Predictions
                    y_pred=y_pred_test.tolist()
                    df_test[f'{cfg_run_mode}'] = y_pred
                    y_test=y_test.tolist()

                    # print(f'Validation:{cfg_run_mode}_for_{shot_number}shot\n', classification_report(y_valid, y_pred_valid, target_names=class_names))
                    # print(f'Test:{cfg_run_mode}_for_{shot_number}shot\n', classification_report(y_test, y_pred_test, target_names=class_names))
                    # print("\n Test DataFrame:\n", df_test)
                    # print("\n Valid DataFrame:\n", df_valid)
                    # print("\n Train DataFrame:\n", df_train)

                    # Save Predictions
                    #output_filepath = f"{output_folder_location}feature_preds_{target_pair}_{cfg_run_mode}_{seed}_{shot_number}.csv"
                    output_filepath = f"{output_folder_location}feature_preds_{target_pair}_{shot_number}_{seed}.csv"
                    print(f"\t\t\t\tSaving best predictions to: {output_filepath}")

                    # Check if the file already exists
                    if os.path.exists(output_filepath):
                        # Load the existing CSV file into a DataFrame
                        df_exist = pd.read_csv(output_filepath)

                        # Add new columns
                        df_exist[f'{cfg_run_mode}'] = y_pred
                        df_exist['y_test'] = y_test

                        # Save the updated DataFrame to the same CSV file
                        df_exist.to_csv(output_filepath, index=False)
                    else:
                        # If the file doesn't exist, create a new DataFrame and save it
                        df_test.to_csv(output_filepath, index=False)

                    print(f"\t\t\t\tFINISHED!")


# In[ ]:


if __name__ == "__main__":
    main()

