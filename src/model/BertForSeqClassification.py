import torch.nn as nn
from transformers import BertModel

# https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForSequenceClassification

# Bert Sentente (Pair) Classification
# 单塔

# https://github.com/google-research/bert/issues/424
# https://github.com/allenai/allennlp/issues/3224



class BertForSeqClassifier(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.config = config        
        self.bert = BertModel.from_pretrained(config['name'], cache_dir = config['cache_dir'], return_dict = False)
        self.bert.resize_token_embeddings(config['vocab_size'])
        self.classifier = nn.Linear(config['hidden_size'], 1) # 2分类
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
    
    
    # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel.forward
    # token type 这里的处理逻辑和gpt不一样; 实现加入token type embedding的功能可能比较麻烦
    
    # sentences_ids: [bsz, seq_len]
    def forward(self, sentences_ids, attention_mask = None, token_type_ids = None):
        """
        :param sentences_ids: [bsz, seq_len]
        :return:
        """
        batch_size, seq_len = sentences_ids.shape
        bert_outputs = self.bert(input_ids = sentences_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict = False)
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        pooled_output = bert_outputs[1] # (batch_size, hidden_size)
        pooled_output = self.dropout(pooled_output) # [bsz, hidden_size]
        sentences_logits = self.classifier(pooled_output) # # [bsz, 1]     
        sentences_logits = sentences_logits.squeeze(-1) # [bsz,]     
        return sentences_logits


    # sentences_ids: [bsz, num_of_seq, seq_len]
    # def forward(self, sentences_ids, attention_mask = None, token_type_ids = None):
    #     """
    #     :param sentences_ids: [bsz, num_of_seq, seq_len]
    #     :return:
    #     """
    #     batch_size, num_of_seq, seq_len = sentences_ids.shape
        
    #     sentences_ids = sentences_ids.view(batch_size * num_of_seq, seq_len) 
    #     if attention_mask is not None:
    #         attention_mask = attention_mask.view(batch_size * num_of_seq, seq_len) 
    #     if token_type_ids is not None:
    #         token_type_ids = token_type_ids.view(batch_size * num_of_seq, seq_len) 


    #     bert_outputs = self.bert(input_ids = sentences_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict = False)
    #     # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    #     pooled_output = bert_outputs[1] # (new_batch_size, hidden_size)
    #     pooled_output = self.dropout(pooled_output)
    #     sentences_features = pooled_output.view(batch_size, num_of_seq, -1) # [bsz, num_of_seq, hidden_size]
    #     sentences_logits = self.classifier(sentences_features) # # [bsz, num_of_seq, 1]     
    #     sentences_logits = sentences_logits.squeeze(-1) # [bsz, num_of_seq]     
    #     return sentences_logits

        


