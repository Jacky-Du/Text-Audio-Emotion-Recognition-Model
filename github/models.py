from sklearn.metrics import confusion_matrix
import torch

import os
import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score,f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
from function import CMD,DiffLoss,MSE
from transformers_encoder.transformer import TransformerEncoder
from collections import defaultdict, OrderedDict
import seaborn as sn

import json
import pickle
import LoadData
from LoadData import label_classes

torch.set_printoptions(threshold=np.inf)
dropout = 0.2
batch_size = 20
epochs = 150
audio_feature_Dimension = 100
audio_Linear = 100
text_embedding_Dimension = 100
Bert_text_embedding_Dimension = 768
text_Linear = 100
gru_hidden = 50
attention_weight_num = 100
attention_head_num = 4
bidirectional = 2  # 2表示双向LSTM,1表示单向

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.set_printoptions(precision=5, suppress=True)
import matplotlib.pyplot as plt


def plot_matrix(matrix):
    '''
    matrix: confusion matrix
    '''
    labels_order = ['hap', 'sad', 'neu', 'ang']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels_order)
    ax.set_yticklabels([''] + labels_order)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    for x in range(len(matrix)):
        for y in range(len(matrix)):
            plt.annotate(matrix[y, x], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    return plt


def weighted_accuracy(list_y_true, list_y_pred):
    '''
    list_y_true: a list of groundtruth labels.
    list_y_pred: a list of labels predicted by the model.
    '''
    assert (len(list_y_true) == len(list_y_pred))

    y_true = np.array(list_y_true)
    y_pred = np.array(list_y_pred)

    w = np.ones(y_true.shape[0])
    for idx, i in enumerate(np.bincount(y_true)):
        w[y_true == idx] = float(1 / i)

    return accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=w)

class After_fusion(nn.Module):
    def __init__(self, fusion="ADD"):
        super(After_fusion, self).__init__()
        self.fusion = fusion
        self.loss_cmd_func = CMD()               
        # 并联concat
        self.Concat_Linear = torch.nn.Linear(label_classes, label_classes,       ##label_classes = 4
                                              bias=False)
        self.Omega_f = torch.normal(mean=torch.full((label_classes, 1), 0.0),
                                    std=torch.full((label_classes, 1), 0.01)).to(device)
        # 特征直接相加

        self.Bert_Linear_text = torch.nn.Linear(Bert_text_embedding_Dimension, text_Linear, bias=True)
        self.Linear_audio = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Linear_text = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Linear_fusion = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Classify_Linear_audio = torch.nn.Linear(100, label_classes, bias=True)
        self.Classify_Linear_text = torch.nn.Linear(100, label_classes, bias=True)
        self.Classify_Linear = torch.nn.Linear(100, label_classes, bias=True)
        self.Softmax = torch.nn.Softmax(dim=2)
        # self.LN=torch.nn.functional.layer_norm([batch_size,])
        self.GRU_audio = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                      bidirectional=True)
        self.GRU_text = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                     bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.Attention_audio = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                           bias=True)
        self.Attention_text = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                          bias=True)



        # private encoders
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=attention_weight_num, out_features=attention_weight_num))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_2', nn.Linear(in_features=attention_weight_num, out_features=attention_weight_num))
        self.private_a.add_module('private_a_activation_2', nn.Sigmoid())
        


        # shared encoder
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=attention_weight_num, out_features=attention_weight_num))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        # reconstruct
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=attention_weight_num, out_features=attention_weight_num))

        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=attention_weight_num, out_features=attention_weight_num))



       #cross transfomer
        self.trans_t_with_a = TransformerEncoder(embed_dim=text_embedding_Dimension ,
                                  num_heads=attention_head_num,
                                  layers=4,
                                  attn_dropout=0.2,
                                  relu_dropout=0.0,
                                  res_dropout=0.0,
                                  embed_dropout=0.2,
                                  attn_mask=True)
        
        self.trans_a_with_t = TransformerEncoder(embed_dim=text_embedding_Dimension ,
                                  num_heads=attention_head_num,
                                  layers=4,
                                  attn_dropout=0.2,
                                  relu_dropout=0.0,
                                  res_dropout=0.0,
                                  embed_dropout=0.2,
                                  attn_mask=True)


        #fusion
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=100*4, out_features=100*2))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(0.2))
        self.fusion.add_module('fusion_layer_1_activation', nn.Sigmoid())
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=100*2, out_features= label_classes))



    def forward(self, Audio_Features, Texts_Features, Seqlen, Mask):
        '''
        Audio_Features: 100 dimensions audio feature which affined from IS13.
                        shape is (B, L, D)
        Texts_Features: 768 dimensions text feature    
                        shape is (B, L, D)    
        Seqlen:         Every sample length in the batch.   
                        shape is (B)
        Mask:           Sample mask. '1' indicates that the corresponding position is allowed to attend.
                        shape is (B, L)
        Where is :
            B: Batch size
            L: Dialogue lengths
            D: Feature dimensions
        '''
        # Convert text to 100 dimensions
        input_text = self.Bert_Linear_text(Texts_Features)
        input_audio = Audio_Features

        # audio flow
        Audio_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_audio, Seqlen,
                                                                batch_first=True, enforce_sorted=False)
            # Audio_GRU_Out shape is (L, B, D)
        Audio_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_audio(Audio_Padding)[0])      
        Audio_MinMask = Mask[:, :Audio_GRU_Out.shape[0]]
        Audio_GRU_Out = self.dropout(Audio_GRU_Out)
            # Audio_Attention_Out shape is (L, B, D)
        Audio_Attention_Out, Audio_Attention_Weight = self.Attention_audio(Audio_GRU_Out, Audio_GRU_Out,
                                                                          Audio_GRU_Out,
                                                                          key_padding_mask=(~Audio_MinMask.to(torch.bool)),
                                                                          need_weights=True)                                                            
        Audio_Dense1 = torch.relu(self.Linear_audio(Audio_Attention_Out.permute([1, 0, 2])))
            # Audio_Dropouted_Dense1 shape is (B, L, D) 
        Audio_Dropouted_Dense1 = self.dropout(Audio_Dense1 * Audio_MinMask[:, :, None])      
            # Audio_Emotion_Output shape is (L, B, D)
        Audio_Emotion_Output = self.Classify_Linear_audio(Audio_Dropouted_Dense1.permute([1, 0, 2]))

        # text flow 
        Text_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_text, Seqlen,
                                                               batch_first=True, enforce_sorted=False)
            # Text_GRU_Out shape is (L, B, D)
        Text_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_text(Text_Padding)[0])
        Text_MinMask = Mask[:, :Text_GRU_Out.shape[0]]
        Text_GRU_Out = self.dropout(Text_GRU_Out)
            # Text_Attention_Out shape is (L, B, D) D = 100
        Text_Attention_Out, Text_Attention_Weight = self.Attention_text(Text_GRU_Out, Text_GRU_Out,
                                                                       Text_GRU_Out,
                                                                       key_padding_mask=(~Text_MinMask.to(torch.bool)),
                                                                       need_weights=True)
        Text_Dense1 = torch.relu(self.Linear_text(Text_Attention_Out.permute([1, 0, 2])))
            # Text_Dropouted_Dense1 shape is (B, L, D)
        Text_Dropouted_Dense1 = self.dropout(Text_Dense1 * Text_MinMask[:, :, None])
            # Text_Emotion_Output shape is (L, B, D)
        Text_Emotion_Output = self.Classify_Linear_text(Text_Dropouted_Dense1.permute([1, 0, 2]))

        
        self.T_feature = Text_Attention_Out  # (L, B, D)
        self.A_feature = Audio_Attention_Out # (L, B, D)
        
        self.share_T = self.shared(self.T_feature)
        self.share_A = self.shared(self.A_feature)

        self.private_T = self.private_t(self.T_feature)
        self.private_A = self.private_a(self.A_feature)
        
        self.utt_t = (self.share_T + self.private_T)
        self.utt_a = (self.share_A + self.private_A)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_a_recon = self.recon_a(self.utt_a)

        self.utt_t_cyc = self.private_t(self.utt_t_recon)
        self.utt_a_cyc = self.private_a(self.utt_a_recon)

        #cross attention
        att_private_T = self.trans_t_with_a(self.private_T, self.private_A, self.private_A) # Q K V
        att_private_A = self.trans_a_with_t(self.private_A, self.private_T, self.private_T)

        h = torch.cat((att_private_T, att_private_A, self.share_T,self.share_A), dim=-1)

        Emotion_Output = self.fusion(h) # (L, B, D)
        Emotion_Predict = self.Softmax(Emotion_Output)
        return Emotion_Predict, Text_Emotion_Output, Audio_Emotion_Output,self.share_A,self.share_T,\
        self.private_A,self.private_T,self.T_feature,self.A_feature, self.utt_t_recon,self.utt_a_recon,\
        self.utt_t_cyc,self.utt_a_cyc




def train_and_test_afterfusion(train_loader, test_loader, model, optimizer, num_epochs, loss_num,
                               savefile=None):
    '''
    train_loader:   Torch dataloader in trainset.
    test_loader:    Torch dataloader in testset.
    model:          Torch model.
    optimizer:      Optimizer for training model.
    num_epochs:     Train epochs.
    loss_num:       '1' means the single loss and 3 indicates the Perspective Loss Function.
    savefile:       Where to save confusion matrix and WA.
    '''
    Best_WA = 0
    Best_F1 = 0
    Loss_Function = nn.CrossEntropyLoss(reduction='none').to(device)
    loss_CMD = CMD().to(device)
    loss_DIFF = DiffLoss().to(device)
    loss_MSE = MSE().to(device)

    for epoch in range(num_epochs):
        confusion_Ypre = []
        confusion_Ylabel = []
        confusion_TrainYlabel = []
        text_confusion_Ypre = []
        audio_confusion_Ypre = []
        model.train()
        model_sne = defaultdict(list)
        max_length = 0
        # train model
        for i, features in enumerate(train_loader):
            audio_train, text_train, train_mask, train_label, seqlen_train= features
            train_mask = train_mask.to(torch.int).to(device)
            audio_train = audio_train.to(device)
            train_label = train_label.to(device)
            text_train = text_train.to(device)
            # inference
            outputs, text_outputs, audio_outputs,share_A,share_T,private_A,private_T,\
            t_feature,a_feature,utt_t_recon,utt_a_recon,utt_t_cyc,utt_a_cyc= \
                model.forward(audio_train, text_train, seqlen_train,train_mask)
            

            text_outputs = torch.nn.functional.softmax(text_outputs, dim=2)
            audio_outputs = torch.nn.functional.softmax(audio_outputs, dim=2)

            train_label = train_label[:, 0:outputs.shape[0]]
            outputs = outputs.permute([1, 2, 0])
            train_label = train_label.permute([0, 2, 1])
            Loss_Label = torch.argmax(train_label, dim=1)

            # calculate loss and update model parameters
            optimizer.zero_grad()

            loss = Loss_Function(outputs, Loss_Label)
            audio_outputs = audio_outputs.permute([1, 2, 0])
            text_outputs = text_outputs.permute([1, 2, 0])
            loss_audio = Loss_Function(audio_outputs, Loss_Label)
            loss_text = Loss_Function(text_outputs, Loss_Label)
            loss_cmd = loss_CMD(share_A,share_T,5)
            loss_dif =loss_DIFF(share_A,private_A) + loss_DIFF(share_T,private_T) + loss_DIFF(private_A,private_T)
            loss_recon = loss_MSE(t_feature,utt_t_recon)+loss_MSE(a_feature,utt_a_recon)
            loss_cyc = loss_MSE(private_A,utt_a_cyc)+loss_MSE(private_T,utt_t_cyc)
            loss_space = loss_cyc+loss_recon+0.1*loss_cmd+0.1*loss_dif
            if loss_num == '1':
                # Single Loss Function
                total_loss_ = loss
            elif loss_num == '3':
                # Perspective Loss Function
                total_loss_ = loss + 0*loss_audio + 0*loss_text + 0.1*loss_space
            True_loss = total_loss_ * train_mask[:, :loss.shape[1]]
            total_loss = torch.sum(True_loss, dtype=torch.float)
            total_loss.backward()
            optimizer.step()
                
        # test model
        with torch.no_grad():
            model.eval()
            correct = 0
            text_correct = 0
            audio_correct = 0
            total = 0
            text_scale=0
            audio_scale=0
            for i, features in enumerate(test_loader):
                
                audio_test, text_test, test_mask, test_label, seqlen_test = features
                test_mask = test_mask.to(torch.int).to(device)
                audio_test = audio_test.to(device)
                test_label = test_label.to(device)
                text_test = text_test.to(device)
                # inference
                outputs, text_outputs, audio_outputs,share_A,share_T,private_A,private_T,\
                 t_feature,a_feature,utt_t_recon,utt_a_recon,utt_t_cyc,utt_a_cyc= \
                model.forward(audio_test, text_test, seqlen_test,test_mask)



                predict = torch.max(outputs, 2)[1].permute([1, 0])
                text_predict = torch.max(text_outputs, 2)[1].permute([1, 0])
                audio_predict = torch.max(audio_outputs, 2)[1].permute([1, 0])
                test_label = test_label[:, :predict.shape[1]]

                # different sample has various length, so we need to mask redundant position
                test_mask = test_mask[:, :predict.shape[1]]
                predict = predict * test_mask
                text_predict = text_predict * test_mask
                audio_predict = audio_predict * test_mask
                test_label = torch.argmax(test_label, dim=2)
                test_label = test_label * test_mask
                
                # count numbers
                total += test_mask.sum()
                correct += ((predict == test_label) * test_mask).sum()

                # record confusion matrix
                for i in range(predict.shape[0]):
                    confusion_Ypre.extend(predict[i][:seqlen_test[i]].cpu().numpy())
                    text_confusion_Ypre.extend(text_predict[i][:seqlen_test[i]].cpu().numpy())
                    audio_confusion_Ypre.extend(audio_predict[i][:seqlen_test[i]].cpu().numpy())
                    confusion_Ylabel.extend(test_label[i][:seqlen_test[i]].cpu().numpy())
                    
            WA=weighted_accuracy(confusion_Ylabel,confusion_Ypre)
            text_WA=weighted_accuracy(confusion_Ylabel,text_confusion_Ypre)
            audio_WA=weighted_accuracy(confusion_Ylabel,audio_confusion_Ypre)
            if WA > Best_WA:
                # fusion confusion matrix
                Best_WA = WA
                matrix = confusion_matrix(confusion_Ylabel, confusion_Ypre)
                total_num = np.sum(matrix, axis=1)
                acc_matrix = np.round(matrix / total_num[:, None], decimals=4)   
                # text confusion matrix
                text_Best_WA = text_WA
                matrix = confusion_matrix(confusion_Ylabel, text_confusion_Ypre)
                total_num = np.sum(matrix, axis=1)
                text_acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
                # audio confusion matrix
                audio_Best_WA = audio_WA
                matrix = confusion_matrix(confusion_Ylabel, audio_confusion_Ypre)
                total_num = np.sum(matrix, axis=1)
                audio_acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
                torch.save(model, "best.pt")       
            F1=f1_score(confusion_Ylabel,confusion_Ypre,average='macro')
            text_F1=f1_score(confusion_Ylabel,text_confusion_Ypre,average='macro')
            audio_F1=f1_score(confusion_Ylabel,audio_confusion_Ypre,average='macro')
            if F1 > Best_F1:
                # fusion confusion matrix
                Best_F1 = F1  
                # text confusion matrix
                text_Best_F1 = text_F1
                # audio confusion matrix
                audio_Best_F1 = audio_F1

        print(
            'Epoch: %d/%d; total utterance: %d ; correct utterance: %d ; WA: %.2f%%; AudioWA: %.2f%%; TextWA: %.2f%%' % (
            epoch + 1, num_epochs, total.item(), correct.item(), WA*100, 100 * audio_WA, 100 * text_WA))

    print("Best Valid WA: %0.2f%%" % (100 * Best_WA))
    print("Best Text Valid WA: %0.2f%%" % (100 * text_Best_WA ))
    print("Best Audio Valid WA: %0.2f%%" % (100 * audio_Best_WA))
    print("Best Valid F1: %0.2f%%" % (100 * Best_F1))
    print("Best Text Valid F1: %0.2f%%" % (100 * text_Best_F1))
    print("Best Audio Valid F1: %0.2f%%" % (100 * audio_Best_F1))
    print(acc_matrix)
    print(audio_acc_matrix)
    print(text_acc_matrix)
    if savefile != None:
        np.savez(savefile, matrix=acc_matrix, ACC=Best_F1, text_matrix=text_acc_matrix, text_ACC=text_F1,
                 audio_matrix=audio_acc_matrix, audio_ACC=audio_F1)

