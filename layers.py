"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

from help_functions import *
from copy import copy

import pdb

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class Embedding_Char(nn.Module):
    """Embedding layer used by BiDAF, with the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors. # (word_vocab_size, word_emb_size)
        char_vectors (torch.Tensor): Random character vectors. # (char_vocab_size, char_emb_size)
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    
    Remark:
        'char_vectors' is a randomly initialized matrix of character embeddings
        'vocab_size' is determined from the training set
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding_Char, self).__init__()
        self.drop_prob = drop_prob
        
        # word embedding
        self.word_emb_size = word_vectors.size(1)
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        
        # character embedding
        self.char_emb_size = char_vectors.size(1)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        
        # CNN layer
        n_filters = self.word_emb_size
        kernel_size = 5
        self.cnn = CNN(self.char_emb_size, n_filters, k=kernel_size)
        
        self.proj = nn.Linear(2*self.word_emb_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x_word, x_char):
        """
        Return the embedding for the words in a batch of sentences.
        Computed from the concatenation of a word-based lookup embedding and a character-based CNN embedding
      
        Args:
            'x_word' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len) where
                each integer is an index into the word vocabulary
            'x_char' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len, max_word_len) where
                each integer is an index into the character vocabulary

        Return:
            'emb' (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)containing the embeddings for each word of the sentences in the batch
        """
        
        # char embedding
        _, seq_len, max_word_len = x_char.size()
            # reshape to a batch of characters word-sequence
        x_char = x_char.view(-1, max_word_len)      # (b = batch_size*seq_len, max_word_len)
            # character-level embedding
        emb_char = self.char_embed(x_char)          # (b, max_word_len, char_emb_size)
            # transpose to match the CNN shape requirements
        emb_char = emb_char.transpose(1, 2)         # (b, n_channel_in = char_emb_size, max_word_len)
            # pass through cnn
        emb_char = self.cnn(emb_char)               # (b, n_channel_out = word_emb_size)
            # reshape to a batch of sentences of words embeddings
        emb_char = emb_char.view(-1, seq_len, self.word_emb_size)  # (batch_size, seq_len, word_emb_size)
    
        # word embedding
        emb_word = self.embed(x_word)               # (batch_size, seq_len, word_emb_size)
        
        # concatenate the char and word embeddings
        emb = torch.cat((emb_word, emb_char), 2)    # (batch_size, seq_len, 2*word_emb_size)
        
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)                        # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)                         # (batch_size, seq_len, hidden_size)

        return emb

class Embedding_Tag(nn.Module):
    """Embedding layer used by BiDAF, with the character-level embedding and the tag (POS and NER) features.

    Embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors. # (word_vocab_size, word_emb_size)
        char_vectors (torch.Tensor): Random character vectors. # (char_vocab_size, char_emb_size)
        pos_vectors (torch.Tensor): one-hot encoding POS vectors. # (n_pos_classes, n_pos_classes)
        ner_vectors (torch.Tensor): one-hot encoding NER vectors. # (n_ner_classes, n_ner_classes)
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    
    Remark:
        'char_vectors' is a randomly initialized matrix of character embeddings
        'vocab_size' is determined from the training set
    """
    def __init__(self, word_vectors, char_vectors, pos_vectors, ner_vectors, hidden_size, drop_prob, freeze_tag=True):
        super(Embedding_Tag, self).__init__()
        self.drop_prob = drop_prob
        
        # word embedding
        self.word_emb_size = word_vectors.size(1)
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        
        # character embedding
        self.char_emb_size = char_vectors.size(1)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        
        # tag features embedding
            # POS
        self.n_pos_classes = pos_vectors.size(1)
        self.pos_embed = nn.Embedding.from_pretrained(pos_vectors, freeze=freeze_tag)
            # NER
        self.n_ner_classes = ner_vectors.size(1)
        self.ner_embed = nn.Embedding.from_pretrained(ner_vectors, freeze=freeze_tag)
        
        # CNN layer
        n_filters = self.word_emb_size
        kernel_size = 5
        self.cnn = CNN(self.char_emb_size, n_filters, k=kernel_size)
        
        self.proj = nn.Linear(2*self.word_emb_size + self.n_pos_classes + self.n_ner_classes, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x_word, x_char, x_pos, x_ner):
        """
        Return the embedding for the words in a batch of sentences.
        Computed from the concatenation of a word-based lookup embedding and a character-based CNN embedding
      
        Args:
            'x_word' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len) where
                each integer is an index into the word vocabulary
            'x_char' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len, max_word_len) where
                each integer is an index into the character vocabulary
            'x_pos' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len) where
                each integer is an index into the POS vocabulary
            'x_ner' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len) where
                each integer is an index into the NER vocabulary
                
        Return:
            'emb' (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)containing the embeddings for each word of the sentences in the batch
        """
        
        # char embedding
        _, seq_len, max_word_len = x_char.size()
            # reshape to a batch of characters word-sequence
        x_char = x_char.view(-1, max_word_len)      # (b = batch_size*seq_len, max_word_len)
            # character-level embedding
        emb_char = self.char_embed(x_char)          # (b, max_word_len, char_emb_size)
            # transpose to match the CNN shape requirements
        emb_char = emb_char.transpose(1, 2)         # (b, n_channel_in = char_emb_size, max_word_len)
            # pass through cnn
        emb_char = self.cnn(emb_char)               # (b, n_channel_out = word_emb_size)
            # reshape to a batch of sentences of words embeddings
        emb_char = emb_char.view(-1, seq_len, self.word_emb_size)  # (batch_size, seq_len, word_emb_size)
    
        # word embedding
        emb_word = self.embed(x_word)               # (batch_size, seq_len, word_emb_size)
        
        # tag embedding
            # POS
        emb_pos = self.pos_embed(x_pos)                 # (batch_size, seq_len, n_pos_classes)
            # NER
        emb_ner = self.ner_embed(x_ner)                 # (batch_size, seq_len, n_ner_classes)
        
        # concatenate the char and word embeddings
        emb = torch.cat((emb_word, emb_char, emb_pos, emb_ner), 2)    # (batch_size, seq_len, 2*word_emb_size)
        
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)                        # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)                         # (batch_size, seq_len, hidden_size)

        return emb
        
class Embedding_Tag_Ext(nn.Module):
    """Embedding layer used by BiDAF, with the character-level embedding and the tag (POS, NER, EM, TF) features.

    Embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors. # (word_vocab_size, word_emb_size)
        char_vectors (torch.Tensor): Random character vectors. # (char_vocab_size, char_emb_size)
        pos_vectors (torch.Tensor): one-hot encoding POS vectors. # (n_pos_classes, n_pos_classes)
        ner_vectors (torch.Tensor): one-hot encoding NER vectors. # (n_ner_classes, n_ner_classes)
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    
    Remark:
        'char_vectors' is a randomly initialized matrix of character embeddings
        'vocab_size' is determined from the training set
    """
    def __init__(self, word_vectors, char_vectors, pos_vectors, ner_vectors, hidden_size, drop_prob, freeze_tag=True):
        super(Embedding_Tag_Ext, self).__init__()
        self.drop_prob = drop_prob
        
        # word embedding
        self.word_emb_size = word_vectors.size(1)
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        
        # character embedding
        self.char_emb_size = char_vectors.size(1)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        
        # tag features embedding
            # POS
        self.n_pos_classes = pos_vectors.size(1)
        self.pos_embed = nn.Embedding.from_pretrained(pos_vectors, freeze=freeze_tag)
            # NER
        self.n_ner_classes = ner_vectors.size(1)
        self.ner_embed = nn.Embedding.from_pretrained(ner_vectors, freeze=freeze_tag)
        
        # CNN layer
        n_filters = self.word_emb_size
        kernel_size = 5
        self.cnn = CNN(self.char_emb_size, n_filters, k=kernel_size)
        
        self.proj = nn.Linear(2*self.word_emb_size + self.n_pos_classes + self.n_ner_classes + 2, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x_word, x_char, x_pos, x_ner, x_em, x_tf):
        """
        Return the embedding for the words in a batch of sentences.
        Computed from the concatenation of a word-based lookup embedding and a character-based CNN embedding
      
        Args:
            'x_word' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len) where
                each integer is an index into the word vocabulary
            'x_char' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len, max_word_len) where
                each integer is an index into the character vocabulary
            'x_pos' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len) where
                each integer is an index into the POS vocabulary
            'x_ner' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len) where
                each integer is an index into the NER vocabulary
            'x_em' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len) where
                each integer is 1 if the word is present in both the context and the question 0 otherwise
            'x_tf' (torch.Tensor): Tensor of double of shape (batch_size, seq_len) where
                each double is the TF*IDF score of the word
                
        Return:
            'emb' (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)containing the embeddings for each word of the sentences in the batch
        """
                
        # char embedding
        _, seq_len, max_word_len = x_char.size()
            # reshape to a batch of characters word-sequence
        x_char = x_char.view(-1, max_word_len)      # (b = batch_size*seq_len, max_word_len)
            # character-level embedding
        emb_char = self.char_embed(x_char)          # (b, max_word_len, char_emb_size)
            # transpose to match the CNN shape requirements
        emb_char = emb_char.transpose(1, 2)         # (b, n_channel_in = char_emb_size, max_word_len)
            # pass through cnn
        emb_char = self.cnn(emb_char)               # (b, n_channel_out = word_emb_size)
            # reshape to a batch of sentences of words embeddings
        emb_char = emb_char.view(-1, seq_len, self.word_emb_size)  # (batch_size, seq_len, word_emb_size)
    
        # word embedding
        emb_word = self.embed(x_word)               # (batch_size, seq_len, word_emb_size)
        
        # tag embedding
            # POS
        emb_pos = self.pos_embed(x_pos)             # (batch_size, seq_len, n_pos_classes)
            # NER
        emb_ner = self.ner_embed(x_ner)             # (batch_size, seq_len, n_ner_classes)
            # EM
        emb_em = x_em.view(-1, seq_len, 1).float()  # (batch_size, seq_len, 1)
            # TF
        emb_tf = x_tf.view(-1, seq_len, 1).float()  # (batch_size, seq_len, 1)
        
        # concatenate the char and word embeddings
        emb = torch.cat((emb_word, emb_char, emb_pos, emb_ner, emb_em, emb_tf), 2)    # (batch_size, seq_len, 2*word_emb_size + self.n_pos_classes + self.n_ner_classes + 2)
        
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)                        # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)                         # (batch_size, seq_len, hidden_size)

        return emb
        
class CNN(nn.Module):
    """Convolutional layer
    1st stage of computing a word embedding from its char embeddings
    
    Remark: process each word in the batch independently
    """
    
    def __init__(self, char_emb_size, f, k=5):
        """Init CNN
        
        Args:
            'char_emb_size' (int): character Embedding Size (nb of input channels)
            'f' (int): number of filters (nb of output channels)
            'k' (int, default=5): kernel (window) size
        """
        super(CNN, self).__init__()
        self.conv1D = nn.Conv1d(char_emb_size, f, k, bias=True)
     
    def forward(self, X_reshaped):
        """Compute the first stage of the word embedding
        
        Args:
            'X_reshaped' (Tensor, shape=(b, char_emb_size, max_word_length)): char-embedded words
                b = batch of words size
        
        Returns:
            'X_conv_out' (Tensor, shape=(b, f)): output of the convolutional layer
        """
        
        X_conv = self.conv1D(X_reshaped) # (b, f, max_word_length - k +1)
        
        # pooling layer to collapse the last dimension
        X_conv_out, _ = torch.max(F.relu(X_conv), dim=2) # (b, f)
                
        return X_conv_out
        
class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

class Encoder(nn.Module):

    def __init__(self, hidden_dim, embedding_matrix, train_word_embeddings, dropout, number_of_layers):

        '''
                arguments passed:
                            1) hidden_dim : the words and chars would be concatenated and would be projected to a space of size=hidden_dim
                            2) embedding_matrix : number of pretrained words, size of embeddings.
                            3) train_word_embeddings : train the word embeddings or not(Boolean).
                            4) dropout : Fraction of neurons dropped
        '''

        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embedding_dim = embedding_matrix.shape[1]
        self.no_of_pretrained_words = embedding_matrix.shape[0]
        self.train_word_embeddings = train_word_embeddings  # boolean

        self.word_embeddings = nn.Embedding.from_pretrained(embedding_matrix)

        self.dropout_fraction = dropout
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.layers = number_of_layers

        self.word_lstm = nn.LSTM(self.word_embedding_dim, hidden_dim, num_layers=number_of_layers, dropout=dropout, batch_first=True, bidirectional=True)

        self.batch_norm = nn.BatchNorm1d(2 * hidden_dim)

        self.mlp = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        self.mlp1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

    def init_hidden(self, dimension, batch_size, device):
        #return torch.zeros(2 * self.layers, batch_size, dimension).cuda(), torch.zeros(2 * self.layers, batch_size, dimension).cuda()
        return torch.zeros(2 * self.layers, batch_size, dimension).to(device), torch.zeros(2 * self.layers, batch_size, dimension).to(device)

    def sort_sents(self, sentences, lengths):

        sorted_lengths, indexes = torch.sort(lengths, descending=True)
        sorted_sentences = sentences[indexes]

        return sorted_sentences, indexes, sorted_lengths

    def forward(self, max_text_length, batch_sentences, batch_lengths, batch_size, apply_batch_norm, istraining, isquestion, device):

        '''
                    batch_sentences : [	[1,3,34,4,23.......],
                                           [24,3,pad,pad,pad..],
                                         [1,2,2,pad,pad.....],
                                         [1,3,2,4,pad.......],]  -> torch tensor
                    batch_lengths : [5,2,3,4]
                    batch_size : number of examples received .
        '''

        word_idx_tensor, indexes, sequence_lengths = self.sort_sents(batch_sentences, batch_lengths)

        word_embed = (self.word_embeddings(word_idx_tensor)).view(batch_size, max_text_length, self.word_embedding_dim)

        if istraining:
            word_embed = self.dropout(word_embed)

        packed_embedds = torch.nn.utils.rnn.pack_padded_sequence(word_embed, sequence_lengths, batch_first=True)

        hidden_layer = word_embed.new_zeros((2*self.layers, batch_size, self.hidden_dim)), word_embed.new_zeros((2*self.layers, batch_size, self.hidden_dim))
        #hidden_layer = self.init_hidden(self.hidden_dim, batch_size, device)

        text_representation, hidden_layer = self.word_lstm(packed_embedds, hidden_layer)

        h, lengths = torch.nn.utils.rnn.pad_packed_sequence(text_representation, batch_first=True, total_length=max_text_length)

        text_representation = torch.zeros_like(h).scatter_(0, indexes.unsqueeze(1).unsqueeze(1).expand(-1, h.shape[1], h.shape[2]), h)

        if istraining and apply_batch_norm:
            text_representation = (self.batch_norm(text_representation.permute(0, 2, 1).contiguous())).permute(0, 2, 1)

        text_representation = self.mlp(text_representation)
        if istraining:
            text_representation = self.dropout(text_representation)

        if isquestion:
            text_representation = self.tanh(self.mlp1(text_representation))
            #zero_one = get_ones_and_zeros_mat(batch_size, sequence_lengths, self.hidden_dim, max_text_length)
            zero_one = text_representation.new_tensor(get_ones_and_zeros_mat(batch_size, sequence_lengths, self.hidden_dim, max_text_length))
            text_representation = text_representation * zero_one

        return text_representation

class Coattention_Encoder(nn.Module):

    def __init__(self, hidden_dim, dropout, number_of_layers):

        '''
            arguments passed:
                 1) hidden_dim : the words and chars would be concatenated and
                 would be projected to a space of size=hidden_dim
        '''

        super(Coattention_Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.layers = number_of_layers

        self.lstm_layer1 = nn.LSTM(2 * hidden_dim, hidden_dim, num_layers=number_of_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.lstm_layer2 = nn.LSTM(12 * hidden_dim, hidden_dim, num_layers=number_of_layers, dropout=dropout, bidirectional=True, batch_first=True)

        self.mlp1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.mlp2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        self.batch_norm = nn.BatchNorm1d(2 * hidden_dim)

    def init_hidden(self, dimension, batch_size, device):
        #return torch.zeros(2 * self.layers, batch_size, dimension).cuda(), torch.zeros(2 * self.layers, batch_size, dimension).cuda()
        return torch.zeros(2 * self.layers, batch_size, dimension).to(device), torch.zeros(2 * self.layers, batch_size, dimension).to(device)

    def sort_sents(self, S, lengths):

        sorted_lengths, indexes = torch.sort(lengths, descending=True)
        sorted_sentences = S[indexes]

        return sorted_sentences, sorted_lengths, indexes

    def forward(self, question, question_lens, passage, passage_lens, batch_size, apply_batch_norm, istraining, device):

        question = question.permute(0, 2, 1)
        L = torch.bmm(passage, question)

        AQ = copy(L)
        AD = copy(L.permute(0, 2, 1))


        #subtract_from_AQ_AD = get_one_zero_mat(batch_size, passage_lens, question_lens, AQ.size(1), AQ.size(2))
        #multiply_to_AD = get_ones_zeros_mat(batch_size, passage_lens, AD.size(1), AD.size(2))
        #multiply_to_AQ = get_ones_zeros_mat(batch_size, question_lens, AQ.size(1), AQ.size(2))

        subtract_from_AQ_AD = AQ.new_tensor(get_one_zero_mat(batch_size, passage_lens, question_lens, AQ.size(1), AQ.size(2)))
        multiply_to_AD = AD.new_tensor(get_ones_zeros_mat(batch_size, passage_lens, AD.size(1), AD.size(2)))
        multiply_to_AQ = AQ.new_tensor(get_ones_zeros_mat(batch_size, question_lens, AQ.size(1), AQ.size(2)))

        AQ = AQ - subtract_from_AQ_AD
        AQ = F.softmax(AQ, dim=1)
        AQ = AQ * multiply_to_AQ

        AD = AD - (subtract_from_AQ_AD.permute(0, 2, 1))
        AD = F.softmax(AD, dim=1)
        AD = AD * multiply_to_AD

        S1D = torch.bmm(question, AD)
        S1Q = torch.bmm(passage.permute(0, 2, 1), AQ)

        C1D = torch.bmm(S1Q, AD)

        S1D = S1D.permute(0, 2, 1)
        S1Q = S1Q.permute(0, 2, 1)

        S1D1, S1D_len, S1D_index = self.sort_sents(S1D, passage_lens)
        S1Q1, S1Q_len, S1Q_index = self.sort_sents(S1Q, question_lens)

        # ------#

        pack_S1D = torch.nn.utils.rnn.pack_padded_sequence(S1D1, S1D_len, batch_first=True)
        hidden_layer = S1D1.new_zeros((2*self.layers, batch_size, self.hidden_dim)), S1D1.new_zeros((2*self.layers, batch_size, self.hidden_dim))
        #hidden_layer = self.init_hidden(self.hidden_dim, batch_size, device)

        S1D1, hidden_layer = self.lstm_layer1(pack_S1D, hidden_layer)
        S1D1, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(S1D1, batch_first=True, total_length=S1D.size(1))

        # ------#

        pack_S1Q = torch.nn.utils.rnn.pack_padded_sequence(S1Q1, S1Q_len, batch_first=True)
        hidden_layer = S1Q1.new_zeros((2*self.layers, batch_size, self.hidden_dim)), S1Q1.new_zeros((2*self.layers, batch_size, self.hidden_dim))
        #hidden_layer = self.init_hidden(self.hidden_dim, batch_size, device)

        S1Q1, hidden_layer = self.lstm_layer1(pack_S1Q, hidden_layer)
        S1Q1, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(S1Q1, batch_first=True, total_length=S1Q.size(1))

        # ------#

        E2D = torch.zeros_like(S1D1).scatter_(0, S1D_index.unsqueeze(1).unsqueeze(1).expand(-1, S1D1.shape[1], S1D1.shape[2]), S1D1)
        E2Q = torch.zeros_like(S1Q1).scatter_(0, S1Q_index.unsqueeze(1).unsqueeze(1).expand(-1, S1Q1.shape[1], S1Q1.shape[2]), S1Q1)

        if istraining and apply_batch_norm:
            E2D = self.batch_norm(E2D.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
            E2Q = self.batch_norm(E2Q.permute(0, 2, 1).contiguous()).permute(0, 2, 1)

        E2D = self.mlp1(E2D)
        E2Q = self.mlp1(E2Q)

        # ------------------------------------------------------------#

        if istraining:
            E2D = self.dropout(E2D)
            E2Q = self.dropout(E2Q)

        E2Q = E2Q.permute(0, 2, 1)
        L = torch.bmm(E2D, E2Q)

        AQ = copy(L)
        AD = copy(L.permute(0, 2, 1))

        #subtract_from_AQ_AD = get_one_zero_mat(batch_size, passage_lens, question_lens, AQ.size(1), AQ.size(2))
        #multiply_to_AD = get_ones_zeros_mat(batch_size, passage_lens, AD.size(1), AD.size(2))
        #multiply_to_AQ = get_ones_zeros_mat(batch_size, question_lens, AQ.size(1), AQ.size(2))

        subtract_from_AQ_AD = AQ.new_tensor(get_one_zero_mat(batch_size, passage_lens, question_lens, AQ.size(1), AQ.size(2)))
        multiply_to_AD = AD.new_tensor(get_ones_zeros_mat(batch_size, passage_lens, AD.size(1), AD.size(2)))
        multiply_to_AQ = AQ.new_tensor(get_ones_zeros_mat(batch_size, question_lens, AQ.size(1), AQ.size(2)))
        
        AQ = AQ - subtract_from_AQ_AD
        AQ = F.softmax(AQ, dim=1)
        AQ = AQ * multiply_to_AQ

        AD = AD - (subtract_from_AQ_AD.permute(0, 2, 1))
        AD = F.softmax(AD, dim=1)
        AD = AD * multiply_to_AD

        S2D = torch.bmm(E2Q, AD)
        S2Q = torch.bmm(E2D.permute(0, 2, 1), AQ)

        C2D = torch.bmm(S2Q, AD)
        S2D = S2D.permute(0, 2, 1)

        C1D = C1D.permute(0, 2, 1)
        C2D = C2D.permute(0, 2, 1)

        # ---------------------------------------------------------#

        '''
            S1D -> batch*passage_lens*2hidden
            S2D -> batch*passage_lens*2hidden
            E1D(passage) ->  batch_size*passage_length*(2hidden)
            E2D -> batch*passage_lens*2hidden
            C1D -> batch*passage_lens*2hidden
            C2D -> batch*passage_lens*2hidden
        '''

        U = torch.cat((passage, E2D, S1D, S2D, C1D, C2D), dim=2)

        U, U_length, U_index = self.sort_sents(U, passage_lens)

        # ----#
        pack_U = torch.nn.utils.rnn.pack_padded_sequence(U, U_length, batch_first=True)
        hidden_layer = U.new_zeros((2*self.layers, batch_size, self.hidden_dim)), U.new_zeros((2*self.layers, batch_size, self.hidden_dim))
        #hidden_layer = self.init_hidden(self.hidden_dim, batch_size, device)

        pack_U, hidden_layer = self.lstm_layer2(pack_U, hidden_layer)
        pack_U, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack_U, batch_first=True, total_length=U.size(1))

        # ----#

        U = torch.zeros_like(pack_U).scatter_(0, U_index.unsqueeze(1).unsqueeze(1).expand(-1, pack_U.shape[1], pack_U.shape[2]), pack_U)

        if istraining and apply_batch_norm:
            U = self.batch_norm(U.permute(0, 2, 1).contiguous()).permute(0, 2, 1)

        U = self.mlp2(U)

        if istraining:
            U = self.dropout(U)

        return U.permute(0, 2, 1)


class Maxout(nn.Module):

    def __init__(self, dim_in, dim_out, pooling_size):
        super().__init__()

        self.d_in, self.d_out, self.pool_size = dim_in, dim_out, pooling_size
        self.lin = nn.Linear(dim_in, dim_out * pooling_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m


class Decoder(nn.Module):

    def __init__(self, hidden_dim, pooling_size, number_of_iters, dropout):

        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layer = nn.LSTMCell(4 * hidden_dim, hidden_dim)
        self.pooling_size = pooling_size
        self.number_of_iters = number_of_iters

        self.tanh = nn.Tanh()
        self.mlp = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.WD_start = nn.Linear(5 * hidden_dim, hidden_dim, bias=False)

        self.maxout1_start = Maxout(3 * hidden_dim, hidden_dim, pooling_size)
        self.maxout2_start = Maxout(hidden_dim, hidden_dim, pooling_size)
        self.maxout3_start = Maxout(2 * hidden_dim, 1, pooling_size)

        self.WD_end = nn.Linear(5 * hidden_dim, hidden_dim, bias=False)

        self.maxout1_end = Maxout(3 * hidden_dim, hidden_dim, pooling_size)
        self.maxout2_end = Maxout(hidden_dim, hidden_dim, pooling_size)
        self.maxout3_end = Maxout(2 * hidden_dim, 1, pooling_size)

    def init_hidden(self, dimension, batch_size, device):
        #return torch.zeros(batch_size, dimension).cuda(), torch.zeros(batch_size, dimension).cuda()
        return torch.zeros(batch_size, dimension).to(device), torch.zeros(batch_size, dimension).to(device)

    def forward(self, encoding_matrix, batch_size, passage_lens, apply_batch_norm, istraining, c_mask, device):

        '''
                    1) encoding_matrix: batch_size*(2*hidden_dim)*max_passage
                    2) batch_size: batch_size
                    3) passage_lens : list having all the proper lengths of the passages(unpaded)
        '''

        #start, end = torch.tensor([0] * batch_size).cuda(), torch.tensor([0] * batch_size).cuda()
        start, end = encoding_matrix.new_zeros(batch_size).long(), encoding_matrix.new_zeros(batch_size).long() 
        #start, end = torch.tensor([0] * batch_size).to(device), torch.tensor([0] * batch_size).to(device)
        #hx, cx = self.init_hidden(self.hidden_dim, batch_size, device)
        hx, cx = encoding_matrix.new_zeros((batch_size, self.hidden_dim)), encoding_matrix.new_zeros((batch_size, self.hidden_dim))

        #zero_one = get_zero_one_mat(batch_size, passage_lens, encoding_matrix.size(2))
        zero_one = encoding_matrix.new_tensor(get_zero_one_mat(batch_size, passage_lens, encoding_matrix.size(2)))

        s1 = start.view(-1, 1, 1).expand(encoding_matrix.size(0), encoding_matrix.size(1), 1)
        e1 = end.view(-1, 1, 1).expand(encoding_matrix.size(0), encoding_matrix.size(1), 1)

        encoding_start_state = encoding_matrix.gather(2, s1).view(batch_size, -1)
        encoding_end_state = encoding_matrix.gather(2, e1).view(batch_size, -1)

        entropies = []

        for i in range(self.number_of_iters):
            alphas, betas = zero_one.new().requires_grad_(), zero_one.new().requires_grad_()
            #alphas, betas = torch.tensor([], requires_grad=True).cuda(), torch.tensor([], requires_grad=True).cuda()
            #alphas, betas = torch.tensor([], requires_grad=True).to(device), torch.tensor([], requires_grad=True).to(device)

            h_s_e = torch.cat((hx, encoding_start_state, encoding_end_state), dim=1)
            r = self.tanh(self.WD_start(h_s_e))

            for j in range(encoding_matrix.shape[2]):
                jth_states = encoding_matrix[:, :, j]

                e_r = torch.cat((jth_states, r), dim=1)

                m1 = self.maxout1_start.forward(e_r)

                m2 = self.maxout2_start.forward(m1)

                hmn = self.maxout3_start.forward(torch.cat((m1, m2), dim=1))

                alphas = torch.cat((alphas, hmn), dim=1)

            alphas = alphas - zero_one
            start = torch.argmax(alphas, dim=1)

            s1 = start.view(-1, 1, 1).expand(encoding_matrix.size(0), encoding_matrix.size(1), 1)
            encoding_start_state = encoding_matrix.gather(2, s1).view(batch_size, -1)

            h_s_e = torch.cat((hx, encoding_start_state, encoding_end_state), dim=1)

            r = self.tanh(self.WD_end(h_s_e))
            for j in range(encoding_matrix.shape[2]):
                jth_states = encoding_matrix[:, :, j]

                e_r = torch.cat((jth_states, r), dim=1)

                m1 = self.maxout1_end.forward(e_r)

                m2 = self.maxout2_end.forward(m1)

                hmn = self.maxout3_end.forward(torch.cat((m1, m2), dim=1))

                betas = torch.cat((betas, hmn), dim=1)

            betas = betas - zero_one
            end = torch.argmax(betas, dim=1)

            e1 = end.view(-1, 1, 1).expand(encoding_matrix.size(0), encoding_matrix.size(1), 1)
            encoding_end_state = encoding_matrix.gather(2, e1).view(batch_size, -1)

            hx, cx = self.lstm_layer(torch.cat((encoding_start_state, encoding_end_state), dim=1), (hx, cx))

            if istraining and apply_batch_norm:
                hx = self.batch_norm(hx)

            hx = self.mlp(hx)

            if istraining:
                hx = self.dropout(hx)

            log_p1 = masked_softmax(alphas, c_mask, 1, log_softmax=True)
            log_p2 = masked_softmax(betas, c_mask, 1, log_softmax=True)
            
            entropies.append([alphas, betas])

        #return start, end, entropies
        return log_p1, log_p2 
        #return alphas, betas
