"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn

import pdb

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class BiDAF_char(nn.Module):
    """Baseline BiDAF model for SQuAD 2.0 with both the word-level and the character-level embedding.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices and character indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Random character vectors. 
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF_char, self).__init__()
        self.emb = layers.Embedding_Char(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)# (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)# (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class BiDAF_tag(nn.Module):
    """Baseline BiDAF model for SQuAD 2.0 with both the word-level and the character-level embedding, and the tag (POS, NER) features.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices and character indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors. # (word_vocab_size, word_emb_size)
        char_vectors (torch.Tensor): Random character vectors. # (char_vocab_size, char_emb_size)
        pos_vectors (torch.Tensor): one-hot encoding POS vectors. # (n_pos_classes, n_pos_classes)
        ner_vectors (torch.Tensor): one-hot encoding NER vectors. # (n_ner_classes, n_ner_classes)
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
        freeze_tag(bool): Whether to freeze the tag features embeddings
    """
    def __init__(self, word_vectors, char_vectors, pos_vectors, ner_vectors, hidden_size, drop_prob=0., freeze_tag=True):
        super(BiDAF_tag, self).__init__()
        self.emb = layers.Embedding_Tag(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    pos_vectors = pos_vectors,
                                    ner_vectors = ner_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    freeze_tag=freeze_tag)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, cpos_idxs, qpos_idxs, cner_idxs, qner_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs, cpos_idxs, cner_idxs)# (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs, qpos_idxs, qner_idxs)# (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
        
class BiDAF_tag_ext(nn.Module):
    """Baseline BiDAF model for SQuAD 2.0 with both the word-level and the character-level embedding, and the tag (POS, NER, EM, TF) features.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices and character indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors. # (word_vocab_size, word_emb_size)
        char_vectors (torch.Tensor): Random character vectors. # (char_vocab_size, char_emb_size)
        pos_vectors (torch.Tensor): one-hot encoding POS vectors. # (n_pos_classes, n_pos_classes)
        ner_vectors (torch.Tensor): one-hot encoding NER vectors. # (n_ner_classes, n_ner_classes)
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
        freeze_tag(bool): Whether to freeze the tag features embeddings
    """
    def __init__(self, word_vectors, char_vectors, pos_vectors, ner_vectors, hidden_size, drop_prob=0., freeze_tag=True):
        super(BiDAF_tag_ext, self).__init__()
        self.emb = layers.Embedding_Tag_Ext(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    pos_vectors = pos_vectors,
                                    ner_vectors = ner_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    freeze_tag=freeze_tag)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, cpos_idxs, qpos_idxs, cner_idxs, qner_idxs, cw_ems, qw_ems, cw_tfs, qw_tfs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs, cpos_idxs, cner_idxs, cw_ems, cw_tfs)# (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs, qpos_idxs, qner_idxs, qw_ems, qw_tfs)# (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
        
class CoattentionModel(nn.Module):

    def __init__(self, hidden_dim, embedding_matrix, train_word_embeddings, dropout, pooling_size, number_of_iters, number_of_layers):
        super(CoattentionModel, self).__init__()

        self.Encoder = layers.Encoder(hidden_dim, embedding_matrix, train_word_embeddings, dropout, number_of_layers)
        self.Coattention_Encoder = layers.Coattention_Encoder(hidden_dim, dropout, number_of_layers)
        self.Decoder = layers.Decoder(hidden_dim, pooling_size, number_of_iters, dropout)

    def forward(self, max_p_length, max_q_length, batch_passages, batch_questions, batch_p_lengths, batch_q_lengths, number_of_examples, apply_batch_norm, istraining):
       
        #max_p_length_batch = torch.max(batch_p_lengths).item()
        #max_q_length_batch = torch.max(batch_q_lengths).item()

        #if max_p_length_batch < max_p_length:
        #    batch_passages = batch_passages[:, :max_p_length_batch]
        #    max_p_length = max_p_length_batch

        #if max_q_length_batch < max_q_length:
        #    batch_questions = batch_questions[:, :max_q_length_batch]
        #    max_q_length = max_q_length_batch

        passage_representation = self.Encoder.forward(max_p_length, batch_passages, batch_p_lengths, number_of_examples, apply_batch_norm, istraining, isquestion=False)

        question_representation = self.Encoder.forward(max_q_length, batch_questions, batch_q_lengths, number_of_examples, apply_batch_norm, istraining, isquestion=True)

        '''
            In passage_length_index and question_length_index-> corresponding elements denote same passage question pair
        '''

        u_matrix = self.Coattention_Encoder.forward(question_representation,
                                                    batch_q_lengths.clone(),
                                                    passage_representation,
                                                    batch_p_lengths.clone(),
                                                    number_of_examples,
                                                    apply_batch_norm,
                                                    istraining)

        # size is batch*(2*hidden)*passage_lens

        alphas, betas = self.Decoder.forward(u_matrix,
                                     number_of_examples,
                                     batch_p_lengths.clone(),
                                     apply_batch_norm,
                                     istraining)

        return alphas, betas#start_outputs, end_outputs, entropies
