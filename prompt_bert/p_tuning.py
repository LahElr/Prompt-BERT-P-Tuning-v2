r"""
#lahelr This file contains isolabel codes from P-tuning-v2
#lahelr Other inisolabel codes are also marked like this
"""

import torch


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2 * num_hidden_layers * hidden_size)
    '''
    def __init__(self, model_args, config):
        super().__init__()
        self.model_args = model_args
        self.config = config
        
        self.prefix_projection = model_args.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(model_args.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, model_args.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(model_args.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(model_args.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            try:
                past_key_values = self.embedding(prefix)
            except RuntimeError as e:
                print(self.embedding.device, prefix.device)
                raise e
        past_key_values = torch.nn.functional.dropout(past_key_values,p=self.model_args.prefix_encoder_dropout)
        return past_key_values
    
def get_prompt(self, batch_size):
    prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
    # print(f"BertPrefixForSequenceClassification:get_prompt:prefix_tokens:{prefix_tokens}, shape {prefix_tokens.shape}")
    past_key_values = self.prefix_encoder(prefix_tokens)
    # print(f"BertPrefixForSequenceClassification:get_prompt:past_key_values:{past_key_values}, shape {past_key_values.shape}")
    # bsz, seqlen, _ = past_key_values.shape

    embedding_size_per_head = self.config.hidden_size/self.config.num_attention_heads
    if not int(embedding_size_per_head) * int(self.config.num_attention_heads) == int(self.config.hidden_size):
        raise ValueError("`hidden_size` must be an integer multiple of `num_attention_heads`")
    embedding_size_per_head = int(embedding_size_per_head)

    past_key_values = past_key_values.view(
        batch_size,
        self.model_args.pre_seq_len,
        self.config.num_hidden_layers * 2, 
        self.config.num_attention_heads,
        embedding_size_per_head
    )
    r'''
    Tuple of tuple(torch.FloatTensor) of length config.n_layers, 
    with each tuple having 2 tensors of shape 
    (batch_size, num_heads, sequence_length, embed_size_per_head)
    '''
    # past_key_values = self.dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    # print(f"BertPrefixForSequenceClassification:past_key_values:{past_key_values}, shape {past_key_values.shape}") #lahelr
    return past_key_values

def get_prompt_(model_args, config,prefix_tokens, prefix_encoder,device, batch_size):
    r"""
    This function is used in evaluation only
    """
    prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
    # print(f"BertPrefixForSequenceClassification:get_prompt:prefix_tokens:{prefix_tokens}, shape {prefix_tokens.shape}")
    past_key_values = prefix_encoder(prefix_tokens)
    # print(f"BertPrefixForSequenceClassification:get_prompt:past_key_values:{past_key_values}, shape {past_key_values.shape}")
    # bsz, seqlen, _ = past_key_values.shape

    embedding_size_per_head = config.hidden_size/config.num_attention_heads
    if not int(embedding_size_per_head) * int(config.num_attention_heads) == int(config.hidden_size):
        raise ValueError("`hidden_size` must be an integer multiple of `num_attention_heads`")
    embedding_size_per_head = int(embedding_size_per_head)

    past_key_values = past_key_values.view(
        batch_size,
        model_args.pre_seq_len,
        config.num_hidden_layers * 2, 
        config.num_attention_heads,
        embedding_size_per_head
    )
    r'''
    Tuple of tuple(torch.FloatTensor) of length config.n_layers, 
    with each tuple having 2 tensors of shape 
    (batch_size, num_heads, sequence_length, embed_size_per_head)
    '''
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    # print(f"BertPrefixForSequenceClassification:past_key_values:{past_key_values}, shape {past_key_values.shape}") #lahelr
    return past_key_values