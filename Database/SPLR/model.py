import torch
import torch.nn as nn
import torch.nn.functional as F
SUB_WEIGHT_COEF = 3.0
CLS_WEIGHT_COEF_1 = [0.1,2.0]
CLS_WEIGHT_COEF_2 = [0.1,0.2]
CLS_WEIGHT_COEF_3 = [0.1, 0.2]
CLS_WEIGHT_COEF_4 = [0.1, 0.5]
CLS_WEIGHT_COEF_5 = [0.1, 0.5]
CLS_WEIGHT_COEF_6 = [0.1, 1.5]
CLS_WEIGHT_COEF_7 = [0.1, 2.0]
CLS_WEIGHT_COEF_8 = [0.1, 2.0]
CLS_WEIGHT_COEF_9 = [0.1, 0.3]
class RModel(nn.Module):
    def __init__(self, BERT, rela_num):
        super().__init__()
        self.bert = BERT
        for p in self.bert.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout(p=0.4)
        self.pred_S = nn.Sequential(nn.Linear(769, 8))

        self.preds = nn.ModuleList([
            nn.Sequential(nn.Linear(770, rela_num)) for _ in range(8)
        ])
        self.pred_tail = nn.Sequential(nn.Linear(768, 1))

        self.lstm_list = nn.ModuleList([
            nn.LSTM(770, 385, 1, batch_first=True, bidirectional=True) for _ in range(8)
        ])
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.conv_list = nn.ModuleList([
            nn.Conv1d(768, 768, kernel_size=k, padding=0) for k in range(1, 9)
        ])

        self.convs = list(zip(self.conv_list, range(1, 9)))

    def get_encoded_text(self, input_ids, mask):
        bert_out1, bert_out = self.bert(input_ids, attention_mask=mask, return_dict=False)
        return bert_out1, bert_out

    def get_objs_for_specific_sub(self, bert):

        B, L, H = bert.size()
        x = bert.transpose(1, 2)

        tail = self.pred_tail(bert)
        pred_tail = self.sigmoid(tail)
        new_bert = torch.cat([bert, tail], dim=-1)
        S = self.sigmoid(self.pred_S(new_bert))
        S_list = [S[..., i:i + 1] for i in range(S.shape[-1])]
        outputs = []
        for conv, k in self.convs:
            if L < k:
                y = bert
            else:
                y = conv(x).transpose(1, 2)
                pad_right = k - 1
                y = F.pad(y, (0, 0, 0, pad_right), value=0.0)
            outputs.append(y)

        # E_1 用 bert，其余用 outputs[1]~outputs[7]
        E_list = [torch.cat([bert, S_list[0], S_list[0]], dim=-1)] + [
            torch.cat([outputs[i], S_list[i], S_list[i]], dim=-1) for i in range(1, 8)
        ]

        lstm_outs = [self.lstm_list[i](E_list[i])[0] for i in range(8)]
        preds = [self.sigmoid(self.preds[i](lstm_outs[i])) for i in range(8)]

        return (pred_tail, S, *preds)
    def clal_loss_1(self, y, p, mask):
        true = y.float()
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_1[1], CLS_WEIGHT_COEF_1[0])
        loss = F.binary_cross_entropy(p, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    def clal_loss_2(self, y, p, mask):
        true = y.float()
        # pred.shape (b, c, 1) -> (b, c)
        #pred = p.squeeze(-1)
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_2[1], CLS_WEIGHT_COEF_2[0])
        loss = F.binary_cross_entropy(p, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    def clal_loss_3(self, y, p, mask):
        true = y.float()
        # pred.shape (b, c, 1) -> (b, c)
        pred = p.squeeze(-1)
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_3[1], CLS_WEIGHT_COEF_3[0])
        loss = F.binary_cross_entropy(pred, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    def clal_loss_4(self, y, p, mask):
        true = y.float()
        # pred.shape (b, c, 1) -> (b, c)
        pred = p.squeeze(-1)
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_4[1], CLS_WEIGHT_COEF_4[0])
        loss = F.binary_cross_entropy(pred, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    def clal_loss_5(self, y, p, mask):
        true = y.float()
        # pred.shape (b, c, 1) -> (b, c)
        pred = p.squeeze(-1)
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_5[1], CLS_WEIGHT_COEF_5[0])
        loss = F.binary_cross_entropy(pred, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    def clal_loss_6(self, y, p, mask):
        true = y.float()
        # pred.shape (b, c, 1) -> (b, c)
        pred = p.squeeze(-1)
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_6[1], CLS_WEIGHT_COEF_6[0])
        loss = F.binary_cross_entropy(pred, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    def clal_loss_7(self, y, p, mask):
        true = y.float()
        # pred.shape (b, c, 1) -> (b, c)
        pred = p.squeeze(-1)
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_7[1], CLS_WEIGHT_COEF_7[0])
        loss = F.binary_cross_entropy(pred, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    def clal_loss_8(self, y, p, mask):
        true = y.float()
        # pred.shape (b, c, 1) -> (b, c)
        pred = p.squeeze(-1)
        num_ones = (true == 1).sum().item()
        num_zero = true.shape[1]
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_8[1], CLS_WEIGHT_COEF_8[0])
        loss = F.binary_cross_entropy(pred, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    def clal_loss_tail(self, y, p, mask):
        true = y.float()
        # pred.shape (b, c, 1) -> (b, c)
        #pred = p.squeeze(-1)
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_9[1], CLS_WEIGHT_COEF_2[0])
        loss = F.binary_cross_entropy(p, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)

    def clal_loss_S(self, y, p, mask):
        true = y.float()
        # pred.shape (b, c, 1) -> (b, c)
        # pred = p.squeeze(-1)
        weight = torch.where(true > 0, CLS_WEIGHT_COEF_9[1], CLS_WEIGHT_COEF_2[0])
        loss = F.binary_cross_entropy(p, true, weight=weight, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    def forward(self, input, input_mask):

        encoded_text,_ = self.get_encoded_text(input, input_mask)
        pred = self.get_objs_for_specific_sub(
            encoded_text)
        return encoded_text,pred
    def loss_fun(self, true_y, pred_y, mask):
        p_tail,S,p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8= pred_y
        t_tail,t_S,t_1,t_2,t_3,t_4,t_5,t_6,t_7,t_8 = true_y
        loss_1 = self.clal_loss_1(t_1,  p_1, mask)*1.5
        loss_2 = self.clal_loss_S(t_S,S,mask)*0.5
        loss_3 = self.clal_loss_2(t_2, p_2, mask) * 0.5
        loss_4 = self.clal_loss_3(t_3, p_3, mask) * 0.5
        loss_5 = self.clal_loss_4(t_4, p_4, mask) * 0.5
        loss_6 = self.clal_loss_5(t_5, p_5, mask) * 1
        loss_7 = self.clal_loss_6(t_6, p_6, mask) * 1
        loss_8 = self.clal_loss_7(t_7, p_7, mask) * 0.5
        loss_9 = self.clal_loss_8(t_8, p_8, mask) * 0.5
        loss_10 = self.clal_loss_tail(t_tail, p_tail, mask) * 0.5


        return loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7+loss_8+loss_9+loss_10
