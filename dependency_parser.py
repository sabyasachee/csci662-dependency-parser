# author - Sabyasachee

import torch
from torch import nn

class DependencyParser(nn.Module):
    
    def __init__(self, word_vocab_size, pos_vocab_size, arc_vocab_size, label_size, use_cube_activation=False):
        '''
        word_vocab_size is the number of words in the word vocabulary
        pos_vocab_size is the number of pos tags in the pos tag vocabulary
        arc_vocab_size is the number of arc labels in the arc label vocabulary
        label_size is the number of labels
        '''
        super().__init__()
        self.word_embedding_layer = nn.Embedding(word_vocab_size, 50)
        self.pos_embedding_layer = nn.Embedding(pos_vocab_size, 50)
        self.arc_embedding_layer = nn.Embedding(arc_vocab_size, 50)
        self.dropout = nn.Dropout(p=0.5)
        self.embed_to_hidden = nn.Linear(2400, 200)
        self.hidden_to_out = nn.Linear(200, label_size)
        self.use_cube_activation = use_cube_activation
        
    def forward(self, batch):
        word, pos, arc = batch[:,:18], batch[:,18:36], batch[:,36:48]
        
        word_embedding = self.word_embedding_layer(word)
        pos_embedding = self.pos_embedding_layer(pos)
        arc_embedding = self.arc_embedding_layer(arc)
        
        embedding = torch.cat([word_embedding.reshape(-1,900), pos_embedding.reshape(-1,900), arc_embedding.reshape(-1,600)], dim=1)
        embedding = self.dropout(embedding)
        
        hidden = self.embed_to_hidden(embedding)

        if self.use_cube_activation:
            hidden = hidden * hidden * hidden
        else:
            hidden = torch.tanh(hidden)
        
        out = self.hidden_to_out(hidden)
        return out