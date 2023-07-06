import torch
import transformers
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartModel, BartConfig

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        configuration = BartConfig(vocab_size = self.config.vocab_size,
                                  encoder_layers = self.config.num_encoder_layers,
                                  decoder_layers = self.config.num_decoder_layers,
                                  d_model = self.config.embedding_size,
                                  dropout = self.config.dropout,
                                  decoder_ffn_dim = self.config.hidden_dim,
                                  encoder_ffn_dim = self.config.hidden_dim,
                                  encoder_attention_head = self.config.nhead,
                                  decoder_attention_head = self.config.nhead)
        
        self.bart = BartForConditionalGeneration.from_pretrained(f'facebook/{self.config.model_name}')
        self.bart.config = configuration
    
    def forward(self, input_ids, decoder_input_ids):
        outputs = self.bart(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        return outputs

    @staticmethod
    def is_valid_model_name(model_name):
        if model_name in ['bart-base', 'bart-large']:
            return True
        else:
            return False

    @staticmethod
    def get_model_from_config(config):
        model_name = config.model_name
        
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))
            
        Net = Model(config)
        
        return Net
    
    
