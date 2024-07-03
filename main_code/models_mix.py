import torch
from torch import nn
import torch.nn.functional as F
from model_tools import *


class mix_soft_label(nn.Module):
    def __init__(self, text_encoder_path, feature_encoder_path, dropout=0.5):
        super(mix_soft_label, self).__init__()

        
        self.text_encoder = torch.load(text_encoder_path)
        self.feature_encoder = torch.load(feature_encoder_path)


    def forward(self, input_id, mask, feature_input):
        text_logit,textEncoding = self.text_encoder(input_id,mask)
        feature_logit,featureEncoding = self.feature_encoder(feature_input)
        
        # output1 = output1.cpu().numpy()
        # output2 = output2.cpu().numpy()
        # predictions = [output1, output2]
        # predictions = np.argmax(np.mean(predictions, axis=0), axis=1)
        
        return text_logit,feature_logit



class Mix_No_FeatureEncoding(nn.Module):
    def __init__(self, text_encoder_path, feature_encoder_path, dropout=0.5):
        super(Mix_No_FeatureEncoding, self).__init__()

        self.text_encoder = torch.load(text_encoder_path,map_location='cuda')
        self.feature_encoder = torch.load(feature_encoder_path,map_location='cuda')

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.feature_encoder.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(29 + 768, 768)
        self.fc2 = nn.Linear(768 + 29, 256)
        self.fc3 = nn.Linear(256, 5)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(1)

    def forward(self, input_id, mask, feature_input):

        text_logits,textEncoding = self.text_encoder(input_id,mask)
        textEncoding = self.dropout(textEncoding)
        

        concatenated_tensor = torch.cat((textEncoding, feature_input), dim=1)
        output = self.dropout((self.relu(self.fc1(concatenated_tensor))))
        concatenated_tensor = torch.cat((output, feature_input), dim=1)
        output = self.fc2(concatenated_tensor)
        output = self.fc3(output)
        return output,(textEncoding,),(text_logits,)

class Mix_liner(nn.Module):
    def __init__(self, text_encoder_path, feature_encoder_path, dropout=0.5):
        super(Mix_liner, self).__init__()

        self.text_encoder = torch.load(text_encoder_path,map_location='cuda')
        self.feature_encoder = torch.load(feature_encoder_path,map_location='cuda')

        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.feature_encoder.parameters():
        #     param.requires_grad = False

        self.mix_layer_1 = nn.Linear(768+64, 128)
        self.mix_layer_2 = nn.Linear(128, 5)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(1)

    def forward(self, input_id, mask, feature_input):

        text_logits,textEncoding = self.text_encoder(input_id,mask)
        textEncoding = self.dropout(textEncoding)


        feature_logits,featureEncoding = self.feature_encoder(feature_input)
        featureEncoding = self.dropout(featureEncoding)

        concatenated_tensor = torch.cat((textEncoding, featureEncoding), dim=1)
        
        output = (self.relu(self.mix_layer_1(concatenated_tensor)))
        output = self.mix_layer_2(output)
        return output,(textEncoding,featureEncoding),(text_logits,feature_logits)
    

class Mix_liner_2(nn.Module):
    def __init__(self, text_encoder_path, feature_encoder_path, dropout=0.5):
        super(Mix_liner_2, self).__init__()

        self.text_encoder = torch.load(text_encoder_path,map_location='cuda')
        self.feature_encoder = torch.load(feature_encoder_path,map_location='cuda')

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.feature_encoder.parameters():
            param.requires_grad = False

        self.text = nn.Linear(768, 256)
        self.feature = nn.Linear(64, 256)
        self.mix_layer_1 = nn.Linear(256+256, 5)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(0.7)

    def forward(self, input_id, mask, feature_input):

        text_logits,textEncoding = self.text_encoder(input_id,mask)
        textEncoding = self.dropout_2(textEncoding)
        textEncoding = self.relu(self.text(textEncoding))

        feature_logits,featureEncoding = self.feature_encoder(feature_input)
        featureEncoding = self.dropout(featureEncoding)
        featureEncoding = self.relu(self.feature(featureEncoding))

        concatenated_tensor = torch.cat((textEncoding, featureEncoding), dim=1)
        
        output = self.mix_layer_1(concatenated_tensor)
        return output,(textEncoding,featureEncoding),(text_logits,feature_logits)



class Mix_att(nn.Module):
    def __init__(self, text_encoder_path, feature_encoder_path, dropout=0.5):
        super(Mix_att, self).__init__()

        self.text_encoder = torch.load(text_encoder_path)
        self.feature_encoder = torch.load(feature_encoder_path)
    
        self.mix_layer = nn.Linear(512, 5)
        
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_id, mask, feature_input):

        _,textEncoding = self.text_encoder(input_id,mask)
        _,featureEncoding = self.feature_encoder(feature_input)
        
        concatenated_tensor = torch.cat((textEncoding, featureEncoding), dim=1)

        output = self.mix_layer(concatenated_tensor)

        return output


class Mix_att_1(nn.Module):
    # KQV   K QV
    def __init__(self, text_encoder_path, feature_encoder_path, dropout=0.5):
        super(Mix_att, self).__init__()

        
        self.text_encoder = torch.load(text_encoder_path)
        self.feature_encoder = torch.load(feature_encoder_path)

        
        self.text_layer_alignment = ffn_gelu(256,1024,256,dropout)
        self.feature_layer_alignment = ffn(256,1024,256,dropout)

        self.attention_text = TransformerEncoder(256,1024,256,dropout)
        self.attention_feature = TransformerEncoder(256,1024,256,dropout)
        self.mix_layer = nn.Linear(512, 256)
        self.mix_layer_2 = nn.Linear(256, 64)
        self.mix_class = nn.Linear(64, 5)
        
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_id, mask, feature_input):
        _,textEncoding = self.text_encoder(input_id,mask)
        _,featureEncoding = self.feature_encoder(feature_input)
        
        textEncoding = self.text_layer_alignment(textEncoding)
        featureEncoding = self.feature_layer_alignment(featureEncoding)

        attention_text_out = self.attention_text      (textEncoding, featureEncoding, featureEncoding)
        attention_feature_out = self.attention_feature(featureEncoding, textEncoding, textEncoding)
        
        concatenated_tensor = torch.cat((attention_text_out, attention_feature_out), dim=1)

        output = self.dropout(F.relu(self.mix_layer(concatenated_tensor)))
        output = self.dropout(F.relu(self.mix_layer_2(output)))
        output = self.mix_class(output)
        
        return output
    




class Mix_att_1(nn.Module):
    # KQV   KQ V
    def __init__(self, text_encoder_path, feature_encoder_path, dropout=0.5):
        super(Mix_att, self).__init__()

        
        self.text_encoder = torch.load(text_encoder_path)
        self.feature_encoder = torch.load(feature_encoder_path)

        
        self.text_layer_alignment = ffn_gelu(256,1024,256,dropout)
        self.feature_layer_alignment = ffn(256,1024,256,dropout)

        self.attention_text = TransformerEncoder(256,1024,256,dropout)
        self.attention_feature = TransformerEncoder(256,1024,256,dropout)
        self.mix_layer = nn.Linear(512, 256)
        self.mix_layer_2 = nn.Linear(256, 64)
        self.mix_class = nn.Linear(64, 5)
        
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_id, mask, feature_input):
        _,textEncoding = self.text_encoder(input_id,mask)
        _,featureEncoding = self.feature_encoder(feature_input)
        
        textEncoding = self.text_layer_alignment(textEncoding)
        featureEncoding = self.feature_layer_alignment(featureEncoding)

        attention_text_out = self.attention_text      (textEncoding, textEncoding, featureEncoding)
        attention_feature_out = self.attention_feature(featureEncoding, featureEncoding, textEncoding)
        
        concatenated_tensor = torch.cat((attention_text_out, attention_feature_out), dim=1)

        output = self.dropout(F.relu(self.mix_layer(concatenated_tensor)))
        output = self.dropout(F.relu(self.mix_layer_2(output)))
        output = self.mix_class(output)
        
        return output