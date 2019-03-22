"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, BiDAF_char, CoattentionModel, BiDAF_tag, BiDAF_tag_ext # NEW
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    char_vectors = util.torch_from_json(args.char_emb_file)

    # NEW : load the tag embeddings
    pos_vectors = util.torch_from_json(args.pos_emb_file)
    ner_vectors = util.torch_from_json(args.ner_emb_file)

    # Choose model
    log.info('Building model {}...'.format(args.name))

    if 'baseline' in args.name:
        model = BiDAF(word_vectors=word_vectors,
                      hidden_size=args.hidden_size)

    elif args.name == 'BiDAF_char':
        model = BiDAF_char(word_vectors=word_vectors,
                      char_vectors=char_vectors,
                      hidden_size=args.hidden_size)

    # NEW
    elif (args.name == 'BiDAF_tag') or (args.name == 'BiDAF_tag_unfrozen') or (args.name == 'BiDAF_tag_loss') or (args.name == 'BiDAF_tag_unfrozen_loss'):
        model = BiDAF_tag(word_vectors=word_vectors,
                      char_vectors=char_vectors,
                      pos_vectors=pos_vectors,
                      ner_vectors=ner_vectors,
                      hidden_size=args.hidden_size)

    elif (args.name == 'BiDAF_tag_ext') or (args.name == 'BiDAF_tag_ext_unfrozen'):
        model = BiDAF_tag_ext(word_vectors=word_vectors,
                      char_vectors=char_vectors,
                      pos_vectors=pos_vectors,
                      ner_vectors=ner_vectors,
                      hidden_size=args.hidden_size)

    elif args.name == 'coattn':
        model = CoattentionModel(hidden_dim=args.hidden_size,
                                embedding_matrix=word_vectors,
                                train_word_embeddings=False,
                                dropout=0.35,
                                pooling_size=16,
                                number_of_iters=4,
                                number_of_layers=2,
                                device=device)

    else:
        raise NameError('No model named ' + args.name)

    model = nn.DataParallel(model, gpu_ids)
    log.info('Loading checkpoint from {}...'.format(args.load_path))
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)['{}_record_file'.format(args.split)]
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info('Evaluating on {} split...'.format(args.split))
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)['{}_eval_file'.format(args.split)]
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, cpos_idxs, cner_idxs, cw_ems, cw_tfs, qw_idxs, qc_idxs, qpos_idxs, qner_idxs, qw_ems, qw_tfs, y1, y2, ids in data_loader: # NEW
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            if 'baseline' in args.name:
                log_p1, log_p2 = model(cw_idxs, qw_idxs)

            elif args.name == 'BiDAF_char':
                # Additional setup for forward
                cc_idxs = cc_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)

            elif args.name == 'coattn':
                max_c_len = cw_idxs.size(1)
                max_q_len = qw_idxs.size(1)

                c_len = []
                q_len = []

                for i in range(cw_idxs.size(0)):
                    if len((cw_idxs[i] == 0).nonzero()) != 0:
                        c_len_i = (cw_idxs[i] == 0).nonzero()[0].item()
                    else:
                        c_len_i = cw_idxs.size(1)

                    if len((qw_idxs[i] == 0).nonzero()) != 0:
                        q_len_i = (qw_idxs[i] == 0).nonzero()[0].item()
                    else:
                        q_len_i = qw_idxs.size(1)

                    c_len.append(int(c_len_i))
                    q_len.append(int(q_len_i))

                c_len = torch.Tensor(c_len).int()
                q_len = torch.Tensor(q_len).int()

                num_examples = int(cw_idxs.size(0) / len(gpu_ids))

                log_p1, log_p2 = model(max_c_len, max_q_len, cw_idxs, qw_idxs, c_len, q_len, num_examples, True, False)

            # NEW
            elif (args.name == 'BiDAF_tag') or (args.name == 'BiDAF_tag_unfrozen') or (args.name == 'BiDAF_tag_loss') or (args.name == 'BiDAF_tag_unfrozen_loss'):
                # Additional setup for forward
                cc_idxs = cc_idxs.to(device)
                cpos_idxs = cpos_idxs.to(device)
                cner_idxs = cner_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                qpos_idxs = qpos_idxs.to(device)
                qner_idxs = qner_idxs.to(device)

                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs, cpos_idxs, qpos_idxs, cner_idxs, qner_idxs)

            elif (args.name == 'BiDAF_tag_ext') or (args.name == 'BiDAF_tag_ext_unfrozen'):
                # Additional setup for forward
                cc_idxs = cc_idxs.to(device)
                cpos_idxs = cpos_idxs.to(device)
                cner_idxs = cner_idxs.to(device)
                cw_ems = cw_ems.to(device)
                cw_tfs = cw_tfs.to(device)
                qc_idxs = qc_idxs.to(device)
                qpos_idxs = qpos_idxs.to(device)
                qner_idxs = qner_idxs.to(device)
                qw_ems = qw_ems.to(device)
                qw_tfs = qw_tfs.to(device)

                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs, cpos_idxs, qpos_idxs, cner_idxs, qner_idxs, cw_ems, qw_ems, cw_tfs, qw_tfs)

            else:
                raise NameError('No model named ' + args.name)

            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                for k, v in results.items())
        log.info('{} {}'.format(args.split.title(), results_str))
        
        # More detailed error analysis
        results = util.analysis_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                for k, v in results.items())
        log.info('{} {}'.format(args.split.title(), results_str))

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info('Writing submission file to {}...'.format(sub_path))
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
