import numpy as np
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
# from pytorchtools import EarlyStopping
import time
from preprocess import get_data
import csv
# import sys
# sys.path.append('/home/chengxiangxin/mieeg') # 添加模块所在的文件夹路径
# import multi_head as mh




class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        # nn.utils.clip_grad_norm_(self.depthwise.parameters(), max_norm=1.0)  
        return out
    
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        nn.init.kaiming_uniform_(self.conv1d.weight, nonlinearity='linear')

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv1d(x)
    

class TCN_block(nn.Module):
    def __init__(self, depth=2):
        super(TCN_block, self).__init__()
        self.depth = depth


        self.Activation_1 = nn.ELU()
        self.TCN_Residual_1 = nn.Sequential(
            #可能问题的所在
            CausalConv1d(32, 32, 4, dilation=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            CausalConv1d(32, 32, 4, dilation=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        
        self.TCN_Residual = nn.ModuleList()
        self.Activation = nn.ModuleList()
        for i in range(depth-1):
            TCN_Residual_n = nn.Sequential(
            CausalConv1d(32, 32, 4, dilation=2**(i+1)),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            CausalConv1d(32, 32, 4, dilation=2**(i+1)),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
        )
            self.TCN_Residual.append(TCN_Residual_n)
            self.Activation.append(nn.ELU())   
        
    def forward(self, x):
        block = self.TCN_Residual_1(x)
        # print(block.shape)
        block += x
        block = self.Activation_1(block)
        
        for i in range(self.depth-1):
            block_o = block
            block = self.TCN_Residual[i](block)
            block += block_o
            # block = torch.add(block_o,block)
            block = self.Activation[i](block)
        return block

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = input_size//(2*num_heads)
        self.num_heads = num_heads
        self.d_k = self.d_v = input_size // (num_heads * 2)
        
        self.W_Q = nn.Linear(input_size, self.hidden_size * num_heads)
        self.W_K = nn.Linear(input_size, self.hidden_size * num_heads)
        self.W_V = nn.Linear(input_size, self.hidden_size * num_heads)
        self.W_O = nn.Linear(self.hidden_size * num_heads, self.input_size)
        
        nn.init.normal_(self.W_Q.weight, mean=0.0, std=self.d_k ** -0.5)
        nn.init.normal_(self.W_K.weight, mean=0.0, std=self.d_k ** -0.5)
        nn.init.normal_(self.W_V.weight, mean=0.0, std=self.d_v ** -0.5)
        nn.init.normal_(self.W_O.weight, mean=0.0, std=self.d_v ** -0.5)

        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 计算Q、K、V
        Q = self.W_Q(x)   # (batch_size, seq_len, hidden_size * num_heads)
        K = self.W_K(x)   # (batch_size, seq_len, hidden_size * num_heads)
        V = self.W_V(x)   # (batch_size, seq_len, hidden_size * num_heads)
        # print(Q)

        # 将Q、K、V按头数进行切分
        Q = Q.view(batch_size, seq_len, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, hidden_size)
        K = K.view(batch_size, seq_len, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, hidden_size)
        V = V.view(batch_size, seq_len, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, hidden_size)
        # print('切分',Q)
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.hidden_size ** 0.5)   # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores.softmax(dim=-1)

        attn_scores = self.dropout(attn_scores)
        # 计算注意力加权后的值
        attn_output = torch.matmul(attn_scores, V)   # (batch_size, num_heads, seq_len, hidden_size)
        # 将头拼接起来
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)   # (batch_size, seq_len, hidden_size * num_heads)
        # 计算输出
        output = self.W_O(attn_output)   # (batch_size, seq_len, hidden_size)
        return output, attn_scores


class attention_block(nn.Module):
    def __init__(self,):
        super(attention_block,self).__init__()
        self.LayerNorm = nn.LayerNorm(normalized_shape=32,eps=1e-06)
        # self.mha = nn.MultiheadAttention(32, 2,dropout=0.5, batch_first=True)
        self.mha = mh.MultiHeadAttention(2,32,0.5)
        # self.mha = MultiHeadAttention(32, 2)
        self.drop = nn.Dropout(0.2)
    
    def forward(self,x):
        #问题
        x = x.permute(2, 0, 1)
        # x = self.LayerNorm(x)
        # att_out,_ = self.mha(x,x,x)
        att_out = self.mha(query=x, key=x, value=x)
        att_out = self.drop(att_out)
        output = att_out.permute(1, 2, 0) + x.permute(1, 2, 0)
        return output

class conv_block(nn.Module):
    def __init__(self,):
        super(conv_block,self).__init__()
        self.conv_block_1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1,64), bias=False,padding='same'),
                nn.BatchNorm2d(16),
                # problem,
            )
        self.depthwise = nn.Conv2d(16, 16*2, (22,1), stride=1, padding=0, dilation=1, groups=16, bias=False)
        self.pointwise = nn.Conv2d(16*2, 16*2, 1, 1, 0, 1, 1, bias=False)
        self.conv_block_2 = nn.Sequential(
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.AvgPool2d(kernel_size=(1,8)),
                nn.Conv2d(32, 32, kernel_size=(1,16), bias=False,padding='same'),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 7)),
                nn.Dropout(0.5),
            )
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        out = self.conv_block_2(x)
        # nn.utils.clip_grad_norm_(self.depthwise.parameters(), max_norm=1.0)  
        return out

class ATCNet(nn.Module):

    def __init__(self, ):
        super(ATCNet, self).__init__()
        #conv模块
        # self.conv_block = nn.Sequential(
        #         nn.Conv2d(1, 16, kernel_size=(64,1), bias=False,padding='same'),
        #         nn.BatchNorm2d(16),
        #         # problem,
        #         DepthwiseConv2d(in_channels=16, out_channels=16*2, kernel_size=(1,22), stride=1, padding=0,dilation=1),
        #         nn.BatchNorm2d(32),
        #         nn.ELU(),
        #         nn.Dropout(0.3),
        #         nn.AvgPool2d(kernel_size=(8,1)),
        #         nn.Conv2d(32, 32, kernel_size=(16,1), bias=False,padding='same'),
        #         nn.BatchNorm2d(32),
        #         nn.ELU(),
        #         nn.AvgPool2d(kernel_size=(7, 1)),
        #         nn.Dropout(0.3),
        #     )
        # 没问题的模块
        # self.conv_block = nn.Sequential(
        #         nn.Conv2d(1, 16, kernel_size=(1,64), bias=False,padding='same'),
        #         nn.BatchNorm2d(16),
        #         # problem,
        #         DepthwiseConv2d(in_channels=16, out_channels=16*2, kernel_size=(22,1), stride=1, padding=0,dilation=1),
        #         nn.BatchNorm2d(32),
        #         nn.ELU(),
        #         nn.Dropout(0.5),
        #         nn.AvgPool2d(kernel_size=(1,8)),
        #         nn.Conv2d(32, 32, kernel_size=(1,16), bias=False,padding='same'),
        #         nn.BatchNorm2d(32),
        #         nn.ELU(),
        #         nn.AvgPool2d(kernel_size=(1, 7)),
        #         nn.Dropout(0.5),
        #     )
        self.conv_block = conv_block()
        #self-attention input_size,hidden_size,num_head
        self.attention_list = nn.ModuleList()
        self.TCN_list = nn.ModuleList()
        self.slideOut_list = nn.ModuleList()
        self.layerNorm_list = nn.ModuleList()
        for i in range(5):
            self.layerNorm_list.append(nn.LayerNorm(normalized_shape=32,eps=1e-06))
            self.attention_list.append(attention_block())
            self.TCN_list.append(TCN_block())
            self.slideOut_list.append(nn.Linear(32,4))

        # self.layerNormalization = nn.LayerNorm(normalized_shape=16,eps=1e-06 )
        # self.multihead_attn = attention_block()
        # self.TCN_block = TCN_block()
        # self.out_1 = nn.Linear(32,4)


        self.out_2 = nn.Linear(160,4)
        self.cv_out = nn.Linear(640,4)

    def forward(self, x):
        #64,1,22,1125
        # x = x.permute(0, 1, 3, 2)
        block1 = self.conv_block(x)
        #64,32,1,20
        # block1 = block1[:,:, -1,:]
        block1 = block1.squeeze(2)

        # block2 = self.multihead_attn(block1)
        # return 1
        # block2 = self.TCN_block(block2)

        fuse = 'average'
        n_windows = 5
        sw_concat = []
        for i in range(n_windows):
            # print(block1.shape)
            # print(i)
            st = i
            end = block1.shape[2]-n_windows+i+1 #在时间窗口上滑动
            # print(end)
            block2 = block1[:,:, st:end]  #获得时间窗口内的数据
            

            # block2 = self.layerNorm_list[i](block2.permute(0,2,1)).permute(0,2,1)

            # Attention_model
            # if attention is not None:
            # block2 = attention_block(block2) 
            block2 = self.attention_list[i](block2)

            # Temporal convolutional network (TCN)
            block3 = self.TCN_list[i](block2)
            # Get feature maps of the last sequence
            # 64,32,16
            block3 = block3[:,:, -1]
            # block3 = torch.functional.F.normalize(block3)
            
            # Outputs of sliding window: Average_after_dense or concatenate_then_dense
            if(fuse == 'average'):
                # block3 = block3.view(block3.size(0), -1)
                sw_concat.append(self.slideOut_list[i](block3))
            elif(fuse == 'concat'):
                if i == 0:
                    sw_concat = block3
                else:
                    sw_concat = torch.cat((sw_concat, block3), axis=1)

        if(fuse == 'average'):
            if len(sw_concat) > 1: # more than one window
                sw_concat = torch.stack(sw_concat).permute(1,0,2)
                # print(sw_concat[0])
                sw_concat = torch.mean(sw_concat, dim=1)
            else: # one window (# windows = 1)
                sw_concat = sw_concat[0]
        elif(fuse == 'concat'):
            sw_concat = self.out_2(sw_concat)

        # sw_concat = self.cv_out(block1.view(block1.size(0), -1))

        return sw_concat
    


def train():
    data_path ='E:/MyBci/mi-data/BCICIV_2a_gdf_mat/'
    dataset_conf = { 'n_classes': 4, 'n_sub': 9, 'n_channels': 22, 'data_path': data_path,
                'isStandard': True, 'LOSO': False}
    #Get dataset paramters
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')

    device = "cuda" #if torch.cuda.is_available() else "cpu"

    criterion = nn.CrossEntropyLoss() 



    for i in range(n_sub):
        # if i != 1:
        #     continue
        #BCIC2a 9人；SMR_BCI1 14人；OpenBMI 54人；
        print(f' Subject = {i:.1f}')

        model = ATCNet()
        #加载数据集
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(data_path, i, LOSO, isStandard)
        eeg_train = torch.from_numpy(X_train)
        label_train = torch.from_numpy(y_train_onehot)
        eeg_test = torch.from_numpy(X_test)
        label_test = torch.from_numpy(y_test_onehot)
        train_dataset=TensorDataset(eeg_train,label_train)
        test_dataset=TensorDataset(eeg_test,label_test)
        # val_dataset=TensorDataset(eeg_val,label_val)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=300, shuffle=False)
        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=32, shuffle=True)

        # The number of training epochs.
        n_epochs = 5000
        # patience = 300	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
        # early_stopping = EarlyStopping(patience, verbose=True)	
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)
        TL = []
        TA = []
        VL = []
        VA = []
        #开始迭代训练
        for epoch in range(n_epochs):
            # ---------- Training ----------
            model.train()
            train_loss = []
            train_accs = []

        # Iterate the training set by batches.
            for k,batch in enumerate(train_loader):
                t1 = time.time()
            # A batch consists of image data and corresponding labels.
                eeg,label = batch
                # eeg = eeg.transpose(0,2,1)
                eeg = eeg.to(torch.float32)
                # label = label.to(torch.float32)
            # Forward the data. (Make sure data and model are on the same device.)
                pred = model(eeg.to(device))
                loss = criterion(pred.cpu(), label)
                optimizer.zero_grad()
            # Compute the gradients for parameters.
                loss.backward()

                #裁剪
                nn.utils.clip_grad_norm_(model.conv_block.depthwise.parameters(), 1.0)
                for j in range(5):
                    nn.utils.clip_grad_norm_(model.slideOut_list[j].parameters(), 0.25)
                # nn.utils.clip_grad_norm_(model.out_2.parameters(), 0.25)
            # Update the parameters with computed gradients.
                optimizer.step()
                train_loss.append(loss.item())
                acc = (pred.argmax(dim=-1) == label.to(device).argmax(dim=-1)).float().mean()
                train_accs.append(acc)

            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            TL.append(train_loss)
            TA.append(train_acc.cpu().item())
            t2 = time.time()
            print(t2-t1)
            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
            model.eval()
            valid_loss = []
            valid_accs = []
            #开始验证数据集
            # Iterate the validation set by batches.
            for k,batch in enumerate(test_loader):
                eeg,label = batch
                eeg = eeg.to(torch.float32)
                # label = label.to(torch.float32)

                # print(eeg.shape)
                with torch.no_grad():
                    logits = model(eeg.to(device))
                    loss = criterion(logits.cpu(), label)
            # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == label.to(device).argmax(dim=-1)).float().mean()

            # Record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc)
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)
            VL.append(valid_loss)
            VA.append(valid_acc.cpu().item())
        # Print the information.
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


            #绘制曲线
        plt.plot(TA)
        plt.plot(VA)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'val'], loc='upper left')
        plt.show()
        plt.plot(TL)
        plt.plot(VL)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'val'], loc='upper left')
        plt.show()
        plt.close()

        print('the Test Acc Best is {}, index is {}'.format(max(VA),VA.index (max(VA))))
        with open('loss'+str(i)+'.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([TL,VL])
        with open('acc'+str(i)+'.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([TA,VA])

            # early_stopping(valid_loss, model)

            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
        break


if __name__ == "__main__":
    train()





