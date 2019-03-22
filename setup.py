"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile

# NOT USED: to determine token features for each word
#from stanfordcorenlp import StanfordCoreNLP
    # POS tagger: https://stanfordnlp.github.io/CoreNLP/pos.html 
    # NER tagger: https://stanfordnlp.github.io/CoreNLP/ner.html
#CORENLP_PATH = '../stanford-corenlp-full-2018-10-05'

# USED: Spacy tagger to stay consistent with the test tokenization
    # https://spacy.io/usage/linguistic-features#section-named-entities

def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print('Downloading {}...'.format(name))
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print('Unzipping {}...'.format(name))
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]
    
def word_tokenize_tag(sent):
    doc = nlp(sent)
    
    words = [token.text for token in doc]
    words_pos = [token.pos_ for token in doc]
    words_ner = [token.ent_type_ for token in doc]
    return words, words_pos, words_ner

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

    
# MODIFIED : to also tag (POS, NER, TF) each word of the context and its associated questions
def process_file(filename, data_type, word_counter, char_counter, pos_counter, ner_counter, example_words={}):
    print("Pre-processing {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    # number of examples
    total = 0
    
    # NOT USED : load the Stanford CoreNLP taggers
    #nlp_tagger = StanfordCoreNLP(CORENLP_PATH)
    
    with open(filename, "r") as fh:
        # load file
        source = json.load(fh)
        
        for article in tqdm(source["data"]):                
            for para in article["paragraphs"]:
                # preprocess the paragraph
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                # convert from paragraph String to list of words String
                #context_tokens_test = word_tokenize(context) # OLD
                context_tokens, context_pos, context_ner = word_tokenize_tag(context) # NEW
                
                # NOT USED : tag each word of the context
                    # POS tagger
                #context_pos = list(list(zip(*nlp_tagger.pos_tag(context)))[1])
                    # NER tagger
                #context_ner = list(list(zip(*nlp_tagger.ner(context)))[1])    
                
                # NEW : store the word context term frequency
                word_context_tf = {}
                # NEW : check what words we already encountered in an example
                is_example_word = {}
                
                # covert from list of words String to list of list of characters String
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                               
                # loop over the context words String
                for i, token in enumerate(context_tokens):
                    # count how many times the word 'token' appears in the context paragraph
                        # weighted by the number of associated questions
                    word_counter[token] += len(para["qas"])

                    # NEW : update the word context term frequency
                    word_context_tf[token] = word_context_tf.get(token, 0) +1
                    # only count each example once
                    if not is_example_word.get(token, False): # first time we encounter the word
                        example_words[token] = example_words.get(token, 0) + len(para["qas"])
                        # update the presence of the word
                        is_example_word[token] = True
                    
                    # NEW : count how many times the tags appear in the context paragraph
                        # weighted by the number of associated questions
                    # POS tagger
                    pos_counter[context_pos[i]] += len(para["qas"])
                    # NER tagger
                    ner_counter[context_ner[i]] += len(para["qas"])
                        
                    for char in token:
                        # count how many times the char 'char' appears in the context paragraph words
                            # weighted by the number of associated questions
                        char_counter[char] += len(para["qas"])
                
                
                # loop over the context associated questions
                for qa in para["qas"]:
                    total += 1
                    # preprocess the question
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                        
                    # convert from paragraph String to list of words String
                    #ques_tokens = word_tokenize(context) # OLD
                    ques_tokens, ques_pos, ques_ner = word_tokenize_tag(ques) # NEW
                    
                    # NEW : tag each word of the context
                        # POS tagger
                    #ques_pos = list(list(zip(*nlp_tagger.pos_tag(ques)))[1])
                        # NER tagger
                    #ques_ner = list(list(zip(*nlp_tagger.ner(ques)))[1])     

                    # NEW : store the word (context,question)-pair term frequency
                    word_question_tf = word_context_tf.copy()
                    
                    # covert from list of words String to list of list of characters String
                    ques_chars = [list(token) for token in ques_tokens]    
                    
                    # loop over the context words String
                    for i, token in enumerate(ques_tokens):
                        # count how many times the word 'token' appears in the question
                        word_counter[token] += 1
                        
                        # NEW : update the word (context,question)-pair term frequency
                        word_question_tf[token] = word_question_tf.get(token, 0) +1
                        
                        # NEW : count how many times the tags appear in the question
                            # POS tagger
                        pos_counter[ques_pos[i]] += 1
                            # NER tagger
                        ner_counter[ques_ner[i]] += 1
                        
                        for char in token:
                            # count how many times the char 'char' appears in the question words
                            char_counter[char] += 1
                    
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                        
                    def _is_word(word, list_words):
                        is_word = (word in list_words) or (word.lower() in list_words) or (word.capitalize() in list_words) or (word.upper() in list_words) 
                        return is_word
                    # NEW : exact match = whether a context word appears in the question (and vice-versa)
                    context_em = [1*_is_word(word, ques_tokens) for word in context_tokens]
                    ques_em = [1*_is_word(word, context_tokens) for word in ques_tokens]
                    
                    # NEW : word (context,question)-pair term frequency (normalized)
                    n_words = np.sum(list(word_question_tf.values()))
                    context_tf = [word_question_tf[word]/n_words for word in context_tokens]
                    ques_tf = [word_question_tf[word]/n_words for word in ques_tokens]
                    
                    # MODIFIED : build the example
                    example = {"context_tokens": context_tokens,
                               "context_chars": context_chars,
                               "context_pos": context_pos, # NEW
                               "context_ner": context_ner, # NEW
                               "context_em": context_em, # NEW
                               "context_tf": context_tf, # NEW
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars,
                               "ques_pos": ques_pos, # NEW
                               "ques_ner": ques_ner, # NEW
                               "ques_em": ques_em, # NEW      
                               "ques_tf": ques_tf, # NEW                               
                               "y1s": y1s,
                               "y2s": y2s,
                               "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {"context": context,
                                                 "question": ques,
                                                 "spans": spans,
                                                 "answers": answer_texts,
                                                 "uuid": qa["id"]}
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


# MODIFIED : to also build the tag embeddings (initialize to one_hot encoding)
def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None, tagger=False):
    print("Pre-processing {} vectors...".format(data_type))
    
    # dictionary: mapping token to embedding vector
    embedding_dict = {}
    # filter out of vocabulary tokens appearing less than limit
    filtered_elements = [k for k, v in counter.items() if v > limit]
    
    # load word embeddings
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    
    # initialize tag embeddings to one-hot encoding
    elif tagger:
        # number of tag classes after filtering
        vec_size = len(filtered_elements)
        for i, token in enumerate(filtered_elements):
            # one-hot encoding
            embedding_dict[token] = [(i == j)*1 for j in range(vec_size)]
        print("{} tokens have corresponding {} embedding vector".format(
            len(filtered_elements), data_type))
            
    # initialize char embeddings randomly
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding {} embedding vector".format(
            len(filtered_elements), data_type))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(args, data, word2idx_dict, char2idx_dict, is_test):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    char_limit = args.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0


# MODIFIED : to also use the words tags as features
def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, pos2idx_dict, ner2idx_dict, example_words, N_train, is_test=False):
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    # drop to long examples at train time
    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit or \
                   (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print("Converting {} examples to indices...".format(data_type))
    # number of examples after filtering
    total = 0
    # number of examples before filtering
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    
    # NEW : words tags features
    context_pos_idxs = []
    context_ner_idxs = []
    context_ems = []
    context_tfs = []
    ques_pos_idxs = []
    ques_ner_idxs = []
    ques_ems = []
    ques_tfs = []
    
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)
        
        # NEW : words tags features
        context_pos_idx = np.zeros([para_limit], dtype=np.int32)
        context_ner_idx = np.zeros([para_limit], dtype=np.int32)
        context_em = -np.ones([para_limit], dtype=np.int32) # NEW : padding token = -1
        context_tf = -np.ones([para_limit], dtype=np.float64) # NEW : padding token = -1.
        ques_pos_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_ner_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_em = -np.ones([ques_limit], dtype=np.int32) # NEW : padding token = -1
        ques_tf = -np.ones([ques_limit], dtype=np.float64) # NEW : padding token = -1.
        
        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
            
            # NEW : words tags
            context_pos_idx[i] = pos2idx_dict.get(example["context_pos"][i], 1)
            context_ner_idx[i] = ner2idx_dict.get(example["context_ner"][i], 1)
            context_em[i] = example["context_em"][i]
            # NEW : TF*IDF
            context_tf[i] = example["context_tf"][i]*np.log(N_train/example_words.get(token, 1))

        context_idxs.append(context_idx)
        
        # NEW : add the example words tags
        context_pos_idxs.append(context_pos_idx)
        context_ner_idxs.append(context_ner_idx)
        context_ems.append(context_em)
        context_tfs.append(context_tf)
        
        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
           
            # NEW : words tags
            ques_pos_idx[i] = pos2idx_dict.get(example["ques_pos"][i], 1)
            ques_ner_idx[i] = ner2idx_dict.get(example["ques_ner"][i], 1)
            ques_em[i] = example["ques_em"][i]
            # NEW : TF*IDF
            ques_tf[i] = example["ques_tf"][i]*np.log(N_train/example_words.get(token, 1))
            
        ques_idxs.append(ques_idx)
        
        # NEW : add the example words tags
        ques_pos_idxs.append(ques_pos_idx)
        ques_ner_idxs.append(ques_ner_idx)
        ques_ems.append(ques_em)
        ques_tfs.append(ques_tf)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(out_file,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             context_pos_idxs=np.array(context_pos_idxs), # NEW
             context_ner_idxs=np.array(context_ner_idxs), # NEW
             context_ems=np.array(context_ems), # NEW
             context_tfs=np.array(context_tfs), # NEW
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             ques_pos_idxs=np.array(ques_pos_idxs), # NEW
             ques_ner_idxs=np.array(ques_ner_idxs), # NEW 
             ques_ems=np.array(ques_ems), # NEW
             ques_tfs=np.array(ques_tfs), # NEW             
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

# MODIFIED : to also use the word tags
def pre_process(args):
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    
    # NEW : build taggers counters
    pos_counter, ner_counter = Counter(), Counter()
    
    # NEW : count in how many train examples each word appears
    example_words = {}
    # MODIFIED : process train file to also tag the context and question words
    #train_examples, train_eval = process_file(args.train_file, "train", word_counter, char_counter)
    train_examples, train_eval = process_file(args.train_file, "train", word_counter, char_counter, pos_counter, ner_counter, example_words=example_words)
    # size of train corpus = number of train (context,question)-pair examples
    N_train = len(train_examples)

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=None, vec_size=args.char_dim)
    # NEW : get the tag embedding matrices
    pos_emb_mat, pos2idx_dict = get_embedding(
        pos_counter, 'pos', emb_file=None, vec_size=None, tagger=True)
    ner_emb_mat, ner2idx_dict = get_embedding(
        ner_counter, 'ner', emb_file=None, vec_size=None, tagger=True)         

    # MODIFIED : process dev file to also tag the context and question words
    #dev_examples, dev_eval = process_file(args.dev_file, "dev", word_counter, char_counter)
    dev_examples, dev_eval = process_file(args.dev_file, "dev", word_counter, char_counter, pos_counter, ner_counter)
    
    # MODIFIED : to also use the words tags as features
    build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict, pos2idx_dict, ner2idx_dict, example_words, N_train)
    dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict, pos2idx_dict, ner2idx_dict, example_words, N_train)
    
    if args.include_test_examples:
        # MODIFIED : process test file to also tag the context and question words
        #test_examples, test_eval = process_file(args.test_file, "test", word_counter, char_counter)
        test_examples, test_eval = process_file(args.test_file, "test", word_counter, char_counter, pos_counter, ner_counter)
        
        save(args.test_eval_file, test_eval, message="test eval")
        # MODIFIED : to also use the words tags as features
        test_meta = build_features(args, test_examples, "test", args.test_record_file, word2idx_dict, char2idx_dict, pos2idx_dict, ner2idx_dict, example_words, N_train, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

    save(args.word_emb_file, word_emb_mat, message="word embedding")
    save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(args.pos_emb_file, pos_emb_mat, message="POS embedding") # NEW
    save(args.ner_emb_file, ner_emb_mat, message="NER embedding") # NEW   
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.word2idx_file, word2idx_dict, message="word dictionary")
    save(args.char2idx_file, char2idx_dict, message="char dictionary")
    save(args.pos2idx_file, pos2idx_dict, message="POS dictionary") # NEW
    save(args.ner2idx_file, ner2idx_dict, message="NER dictionary") # NEW    
    save(args.dev_meta_file, dev_meta, message="dev meta")


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    download(args_)

    # Import spacy language model
    #nlp = spacy.blank("en")
    # LM to do word tagging (POS, NER)
    nlp = spacy.load("en_core_web_sm")
    
    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    glove_ext = '.txt' if glove_dir.endswith('d') else '.{}d.txt'.format(args_.glove_dim)
    args_.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    pre_process(args_)
