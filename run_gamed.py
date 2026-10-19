"""GAMED training and validation for Fakeddit and Yang."""
import argparse
import datetime
import os
from pathlib import Path
import sys
from output_paths import create_run_directory, create_epoch_directory

PROJECT_ROOT = Path(__file__).resolve().parent
GT_size = 224
word_token_length = 197
image_token_length = 197


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('-network_arch', required=True, choices=('gamed', 'gamed_without_veto'))
    parser.add_argument('-train_dataset', required=True, choices=('fakeddit', 'yang'))
    parser.add_argument('-test_dataset', required=True, choices=('fakeddit', 'yang'))
    parser.add_argument('-output_file', default='checkpoint/', help='Root directory for checkpoints, features and predictions')
    parser.add_argument('-checkpoint', default='')
    parser.add_argument('-finetune', type=int, default=0)
    parser.add_argument('-val', type=int, default=0)
    parser.add_argument('-is_filter', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('-duplicate_fake_times', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('-is_sample_positive', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-not_on_12', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('-device', type=int, default=0, help='CUDA device index')
    return parser


def validate_arguments(arguments):
    build_parser().parse_args(arguments)
    return list(arguments)


def initialise_runtime():
    """Load runtime dependencies and the English tokenizer before research seeding.

    Importing this entry, requesting help or rejecting invalid CLI values does
    not load frameworks, tokenizers, data or weights.
    """
    global _read_path, np, torch, nn, Variable, DataLoader, warmup
    global AutoTokenizer, ImageFile, token_uncased_fakeddit, Progbar
    from resource_paths import resolve_read_path as _read_path, resolve_english_encoder_path
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    # Preserve MAE/timm/torchvision import initialisation before research seeding.
    # The entry does not otherwise read this binding; the model uses the cache.
    import models_mae
    import pytorch_warmup as warmup
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    token_uncased_fakeddit = AutoTokenizer.from_pretrained(resolve_english_encoder_path())
    from utils import Progbar


def resolve_model_class(name):
    if name == 'gamed':
        from models.gamed import GAMED
        return GAMED
    if name == 'gamed_without_veto':
        from models.gamed_without_veto import GAMEDWithoutVeto
        return GAMEDWithoutVeto
    raise ValueError('Unsupported GAMED variant: ' + str(name))


def cli(arguments=None):
    args = build_parser().parse_args(arguments)
    os.chdir(PROJECT_ROOT)
    initialise_runtime()
    torch.cuda.set_device(args.device)
    print('Using GPU: {} '.format(args.device))
    main(args)


def to_var(x):
    
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def collate_fn_english_fakeddit(data):
    item = data[0]
    sents = [i[0][0] for i in data]
    image = [i[1][0] for i in data]
    image_aug = [torch.tensor(i[1][1]) for i in data]
    labels = [i[1][2] for i in data]
    category = [0 for i in data]
    GT_path = [i[2] for i in data]
    token_data = token_uncased_fakeddit(sents,return_tensors='pt',truncation=True,return_length=True,padding='max_length',max_length=word_token_length)
    
    input_ids = token_data['input_ids']
    attention_mask = token_data['attention_mask']
    token_type_ids = token_data['token_type_ids']
    image = torch.stack(image)
    image_aug = torch.stack(image_aug)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)

    if len(item) <= 2:
        return (input_ids, attention_mask, token_type_ids), (image, image_aug, labels, category, sents), GT_path
    else:
        sents1 = [i[0][0] for i in data]
        image1 = [i[1][0] for i in data]
        labels1 = [i[1][2] for i in data]
        token_data1 = token_uncased_fakeddit(sents1,return_tensors='pt',return_length=True,padding='max_length',max_length=word_token_length,truncation=True,)

        input_ids1 = token_data1['input_ids']
        attention_mask1 = token_data1['attention_mask']
        token_type_ids1 = token_data1['token_type_ids']
        image1 = torch.stack(image1)
        labels1 = torch.LongTensor(labels1)

        return (input_ids, attention_mask, token_type_ids), (image, image_aug, labels, category, sents), GT_path, \
               (input_ids1, attention_mask1, token_type_ids1), (image1, labels1, sents1)


stateful_metrics = ['L-RealTime','lr','APEXGT','empty','exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
                                'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main(args):
    print(args)


    seed = 25
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print("Initialising CUDA AMP gradient scaler")
    scaler = torch.cuda.amp.GradScaler()

    print('loading data')
    setting = {}
    setting['checkpoint_path'] = args.checkpoint
    print('loading checkpoint from {}'.format(setting['checkpoint_path']))
    setting['train_dataname'] = args.train_dataset
    setting['val_dataname'] = args.test_dataset
    setting['val'] = args.val
    setting['network_arch'] = args.network_arch
    setting['is_filter'] = args.is_filter>0
    setting['duplicate_fake_times'] = args.duplicate_fake_times
    setting['is_use_unimodal'] = True
    setting['with_ambiguity'] = False
    setting['data_augment'] = False
    setting['not_on_12'] = args.not_on_12
    setting['is_use_bce'] = True
    setting['use_soft_label'] = False
    setting['is_sample_positive'] = args.is_sample_positive
    setting['thresh'] = 0.5

    custom_batch_size = args.batch_size
    custom_num_epochs = args.epochs
    print("Compatibility filter flag (unused by supported loaders): {}".format(setting['is_filter']))
    train_dataset, validate_dataset, train_loader, validate_loader = None,None,None,None
    train_sampler = None


    if setting['train_dataname']=="fakeddit":
        from data.Fakeddit_dataset import Fakeddit_dataset
        NUM_WORKER = 56
        dataset_dir = 'dataset'
        train_dataset = Fakeddit_dataset(root_path=os.path.join(os.getcwd(), dataset_dir), mode = 'train', image_size=GT_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=custom_batch_size,
            num_workers=NUM_WORKER,
            shuffle=True,collate_fn = collate_fn_english_fakeddit,)
        print(f"thresh:{setting['thresh']}")

    elif setting['train_dataname']=="yang":
        from data.Yang_dataset import Yang_dataset
        NUM_WORKER = 56
        dataset_dir = 'dataset'
        train_dataset = Yang_dataset(root_path=os.path.join(os.getcwd(), dataset_dir), mode = 'train', image_size=GT_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=custom_batch_size,
            num_workers=NUM_WORKER,
            shuffle=True,collate_fn = collate_fn_english_fakeddit,)
        print(f"thresh:{setting['thresh']}")
    else:
        raise ValueError("Unsupported dataset")

    if setting['val_dataname']=='fakeddit':
        print("Preparing Fakeddit validation data")
        from data.Fakeddit_dataset import Fakeddit_dataset
        dataset_dir = 'dataset'
        validate_dataset = Fakeddit_dataset(root_path=os.path.join(os.getcwd(), dataset_dir), mode = 'validate',
            is_train=False, image_size=GT_size,
        )
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                     collate_fn = collate_fn_english_fakeddit,
                                     num_workers=32, sampler=None, drop_last=False,
                                     pin_memory=True)
    elif setting['val_dataname']=='yang':
        print("Preparing Yang validation data")
        from data.Yang_dataset import Yang_dataset
        dataset_dir = 'dataset'
        validate_dataset = Yang_dataset(root_path=os.path.join(os.getcwd(), dataset_dir), mode = 'validate',
            is_train=False, image_size=GT_size,
        )
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                     collate_fn = collate_fn_english_fakeddit,
                                     num_workers=32, sampler=None, drop_last=False,
                                     pin_memory=True)        
    else:
        raise ValueError("Unsupported dataset")

    print('building model')
    model_type = resolve_model_class(setting['network_arch'])
    print(f"Network {setting['network_arch']}")
    model = model_type(dataset=setting['train_dataname'],
                      text_token_len=word_token_length,
                      image_token_len=image_token_length,
                      is_use_bce=setting['is_use_bce'],
                      batch_size=custom_batch_size,
                      thresh=setting['thresh'],
                      )

    if len(setting['checkpoint_path'])!=0:
        print("loading checkpoint: {}".format(setting['checkpoint_path']))
        load_model(model, setting['checkpoint_path'])
    model = model.cuda()
    model.train()

    print(get_parameter_number(model))
    loss_cross_entropy = nn.CrossEntropyLoss().cuda()
    loss_bce = nn.BCEWithLogitsLoss().cuda()
    criterion = loss_bce
    l1_loss = nn.L1Loss().cuda()
    optim_params_normal, optim_params_fast, optim_params_extremefast = [], [], []
    name_params_normal, name_params_fast, name_params_extremefast = [], [], []
    for k, v in model.named_parameters():
        if v.requires_grad:
            if "image_encoder" in k or "text_encoder" in k:
                name_params_normal.append(k)
                optim_params_normal.append(v)
            elif "ip_backbone" in k or "irrelevant_tensor" in k:
                name_params_extremefast.append(k)
                optim_params_extremefast.append(v)
            else:
                name_params_fast.append(k)
                optim_params_fast.append(v)
    fine_tuning = args.finetune>0
    print(f"Fine-tuning mode: {fine_tuning}")
    optimizer = torch.optim.AdamW(optim_params_normal,
                                 lr=1e-5,betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_fast = torch.optim.AdamW(optim_params_fast,
                                  lr=5e-5 if not fine_tuning>0 else 1e-5, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_extremefast = torch.optim.AdamW(optim_params_extremefast,
                                       lr=1e-4 if not fine_tuning else 1e-5, betas=(0.9, 0.999), weight_decay=0.01)
    num_steps = int(len(train_loader) * custom_num_epochs * 1.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    scheduler_fast = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fast, T_max=num_steps)
    warmup_scheduler_fast = warmup.UntunedLinearWarmup(optimizer_fast)
    scheduler_extremefast = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_extremefast, T_max=num_steps)
    warmup_scheduler_extremefast = warmup.UntunedLinearWarmup(optimizer_extremefast)
    print("Using CosineAnnealingLR+UntunedLinearWarmup")

    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_acc_so_far = 0.000
    best_epoch_record = 0
    global_step = 0
    print('Starting training or validation')


    if setting['val']!=0:
        custom_num_epochs = 1

    run_directory = create_run_directory(args.output_file, args.train_dataset, args.test_dataset, args.network_arch, validation=args.val != 0)

    for epoch in range(custom_num_epochs):
        cost_vector = []
        acc_vector = []
        if setting['val']==0:
            total = len(train_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            for i, items in enumerate(train_loader):

                with torch.enable_grad():
                    logs = []
                    model.train()

                    if setting['with_ambiguity']:
                        """
                        (input_ids, attention_mask, token_type_ids), (image, labels, category, sents), GT_path, 
               (input_ids1, attention_mask1, token_type_ids1), (image1, labels1, sents1)
                        """
                        texts, others, GT_path, texts1, others1 = items
                        input_ids, attention_mask, token_type_ids = texts
                        input_ids1, attention_mask1, token_type_ids1 = texts1
                        image, image_aug, labels, category, sents = others
                        image1, labels1, sents1 = others1
                        input_ids, attention_mask, token_type_ids, image, image_aug, labels, category = \
                            to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), \
                            to_var(image), to_var(image_aug), to_var(labels), to_var(category)
                        input_ids1, attention_mask1, token_type_ids1, image1, labels1 = \
                            to_var(input_ids1), to_var(attention_mask1), to_var(token_type_ids1), \
                            to_var(image1), to_var(labels1)
                    else:
                        """
                        (input_ids, attention_mask, token_type_ids), (image, labels, category, sents)
                        """
                        texts, others, GT_path, _, _ = items
                        input_ids, attention_mask, token_type_ids = texts
                        image, image_aug, labels, category, sents = others
                        input_ids, attention_mask, token_type_ids, image, image_aug, labels, category = \
                             to_var(input_ids),to_var(attention_mask), to_var(token_type_ids), \
                            to_var(image), to_var(image_aug), to_var(labels), to_var(category)

                    loss_ambiguity = 0
                    if setting['with_ambiguity']:
                        mm_aux_output, *_ = model(input_ids=input_ids1, 
                                          attention_mask=attention_mask1,
                                          token_type_ids=token_type_ids1,
                                          image=image1,
                                         no_ambiguity=False,
                                          category=torch.zeros_like(category),
                                         calc_ambiguity=True,
                                         )
                        loss_ambiguity += criterion(mm_aux_output,labels1.float().unsqueeze(1))
                        logs.append(('MM_ambiguity_loss', loss_ambiguity.item()))


                    mix_output, is_output, t_output, ip_output, mm_aux_output, irrelevant_mean = model(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          image=image,
                                          image_aug=image_aug,
                                          no_ambiguity=not setting['with_ambiguity'],
                                          category=category,
                                          calc_ambiguity=False,
                                      )

                    if setting['is_use_bce']:
                        labels = labels.float().unsqueeze(1)

                    mix_loss = criterion(mix_output, labels)
                    is_loss = criterion(is_output, labels)
                    t_loss = criterion(t_output, labels)
                    ip_loss = criterion(ip_output, labels)
                    loss_single_modal = (ip_loss+t_loss+is_loss)/3
                    loss = mix_loss+2.0*loss_ambiguity+1.0*loss_single_modal

                    global_step += 1

                    optimizer.zero_grad()
                    optimizer_fast.zero_grad()
                    optimizer_extremefast.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    cvt = 6
                    if epoch>=cvt:
                        optimizer.step()
                    optimizer_fast.step()
                    optimizer_extremefast.step()

                    logs.append(('Mix_loss',mix_loss.item()))
                    logs.append(('IS_loss', is_loss.item()))
                    logs.append(('T_loss', t_loss.item()))
                    logs.append(('IP_loss', ip_loss.item()))
                    logs.append(('MM_mean_prob', torch.mean(torch.sigmoid(mm_aux_output)).item()))
                    logs.append(('Irr_mean', irrelevant_mean.item()))
                    if not setting['is_use_bce']:
                        _, argmax = torch.max(mix_output, 1)
                        accuracy = (labels == argmax.squeeze()).float().mean()
                    else:
                        accuracy = (torch.sigmoid(mix_output).round_() == labels.round_()).float().mean()

                    cost_vector.append(loss.item())
                    acc_vector.append(accuracy.item())
                    mean_cost, mean_acc = np.mean(cost_vector), np.mean(acc_vector)
                    logs.append(('Train_acc', mean_acc))
                    if model.mm_score is not None:
                        mean_mm_score = torch.mean(model.mm_score).item()
                        logs.append(('mm_score', mean_mm_score))
                        mean_text_score = torch.mean(model.text_score).item()
                        logs.append(('text_score', mean_text_score))
                        mean_image_score = torch.mean(model.image_score).item()
                        logs.append(('image_score', mean_image_score))
                    progbar.add(len(image), values=logs)
                    if epoch>=cvt:
                        with warmup_scheduler.dampening():
                            scheduler.step()
                    with warmup_scheduler_fast.dampening():
                        scheduler_fast.step()
                    with warmup_scheduler_extremefast.dampening():
                        scheduler_extremefast.step()

            print('Epoch [%d/%d],  Loss: %.4f, Train_Acc: %.4f,  '
                  % (
                      epoch + 1, custom_num_epochs, np.mean(cost_vector), np.mean(acc_vector)))
            print("Training epoch completed")

        with torch.no_grad():
            total = len(validate_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            model.eval()
            print("Starting validation")
            validate_acc_list, validate_real_items, validate_fake_items, val_loss, single_items = evaluate(validate_loader, model, criterion, progbar=progbar, setting=setting, output_directory=create_epoch_directory(run_directory, epoch + 1))

            validate_acc = max(validate_acc_list)
            print("validate_acc_list:",validate_acc_list)
            val_thresh = validate_acc_list.index(validate_acc)

            validate_real_precision, validate_real_recall, validate_real_accuracy, validate_real_F1 = validate_real_items
            validate_fake_precision, validate_fake_recall, validate_fake_accuracy, validate_fake_F1 = validate_fake_items
            is_correct, t_correct, ip_correct, mm_correct = single_items
            is_acc, t_acc, ip_acc, mm_acc = is_correct[val_thresh], t_correct[val_thresh], ip_correct[val_thresh], mm_correct[val_thresh]
            if validate_acc > best_acc_so_far:
                best_acc_so_far = validate_acc
                best_epoch_record = epoch+1

            print('Epoch [%d/%d],  validation accuracy: %.4f at threshold index %d (best %.4f at epoch %d)'
                  % (
                      epoch + 1, custom_num_epochs, validate_acc, val_thresh, best_acc_so_far, best_epoch_record,
                  ))
            print(f'Branch accuracies: IS {is_acc}, T {t_acc}, IP {ip_acc}, MM {mm_acc}')
            print("------Real News -----------")
            print("Precision: {}".format(validate_real_precision))
            print("Recall: {}".format(validate_real_recall))
            print("Accuracy: {}".format(validate_real_accuracy))
            print("F1: {}".format(validate_real_F1))
            print("------Fake News -----------")
            print("Precision: {}".format(validate_fake_precision))
            print("Recall: {}".format(validate_fake_recall))
            print("Accuracy: {}".format(validate_fake_accuracy))
            print("F1: {}".format(validate_fake_F1))
            print("---------------------------")
            print("Validation completed")
            if True:
                best_validate_acc = validate_acc
                os.makedirs(run_directory / 'checkpoints', exist_ok=True)
                best_validate_dir = run_directory / 'checkpoints' / f'epoch_{epoch + 1:03d}.pkl'
                torch.save(model.state_dict(), best_validate_dir)
                print("Model saved at {}".format(best_validate_dir))

from collections import OrderedDict
def load_model(model, load_path, strict=True):
    load_net = torch.load(_read_path(load_path))
    return model.load_state_dict(load_net, strict=strict)


def evaluate(validate_loader, model, criterion, progbar=None, setting={}, *, output_directory):
    model.eval()
    validate_acc_vector_temp, validate_precision_vector_temp, validate_recall_vector_temp, validate_F1_vector_temp  = [], [], [], []
    val_loss = 0
    threshold = setting['thresh']
    THRESH = [threshold+i/1000 for i in range(-50,50)]

    print(f"thresh: {THRESH}")
    realnews_TP, realnews_TN, realnews_FP, realnews_FN = [0]*len(THRESH), [0]*len(THRESH), [0]*len(THRESH), [0]*len(THRESH)
    fakenews_TP, fakenews_TN, fakenews_FP, fakenews_FN = [0]*len(THRESH), [0]*len(THRESH), [0]*len(THRESH), [0]*len(THRESH)
    realnews_sum, fakenews_sum = [0]*len(THRESH), [0]*len(THRESH)
    is_correct, mm_correct, t_correct, ip_correct = [0]*len(THRESH), [0]*len(THRESH), [0]*len(THRESH),  [0]*len(THRESH)
    y_pred_full, y_GT_full = None, None
    y_pred_fake_full, y_GT_fake_full, y_pred_real_full, y_GT_real_full = None, None, None, None
    image_no,results = 0,[]
    dataset_name = setting['val_dataname']

    export_mix_features = torch.zeros(1,64).cuda()
    export_is_features = torch.zeros(1,64).cuda()
    export_t_features = torch.zeros(1,64).cuda()
    export_ip_features = torch.zeros(1,64).cuda()
    export_mm_features = torch.zeros(1,64).cuda()
    all_labels = torch.zeros(1,1).cuda()
    for i, items in enumerate(validate_loader):
        
        
        texts, others, GT_path, _, _ = items
        input_ids, attention_mask, token_type_ids = texts
        image, image_aug, labels, category, sents = others
        input_ids, attention_mask, token_type_ids, image, image_aug, labels, category = \
                to_var(input_ids),to_var(attention_mask), to_var(token_type_ids), \
            to_var(image), to_var(image_aug), to_var(labels), to_var(category)
        mix_output, is_output, t_output, ip_output, mm_aux_output, _ , features = model(input_ids=input_ids,
                                                                        attention_mask=attention_mask,
                                                                        token_type_ids=token_type_ids,
                                                                        image=image,
                                                                        image_aug=image_aug,
                                                                        no_ambiguity=True,
                                                                        category=category,
                                                                       return_features=True)
        mix_feature, is_feature, t_feature, ip_feature, mm_aux_feature = features
        export_mix_features = torch.cat([export_mix_features,mix_feature],0)
        export_is_features = torch.cat([export_is_features,is_feature],0)
        export_t_features = torch.cat([export_t_features,t_feature],0)
        export_ip_features = torch.cat([export_ip_features,ip_feature],0)
        export_mm_features = torch.cat([export_mm_features,mm_aux_feature],0)
        if setting['is_use_bce']:
            labels = labels.float().unsqueeze(1)
        all_labels = torch.cat([all_labels,labels],0)
        val_loss = criterion(mix_output, labels)
        val_is_loss = criterion(is_output, labels)
        val_t_loss = criterion(t_output, labels)
        val_ip_loss = criterion(ip_output, labels)
        if progbar is not None:
            logs = []
            logs.append(('Mix_loss', val_loss.item()))
            logs.append(('IS_loss', val_is_loss.item()))
            logs.append(('T_loss', val_t_loss.item()))
            logs.append(('IP_loss', val_ip_loss.item()))
            progbar.add(len(image), values=logs) 

        mix_output, is_output, t_output, ip_output, mm_aux_output = torch.sigmoid(mix_output), torch.sigmoid(
            is_output), torch.sigmoid(t_output), torch.sigmoid(ip_output), torch.sigmoid(mm_aux_output)
        for thresh_idx, thresh in enumerate(THRESH):
            validate_argmax = torch.where(mix_output<thresh,0,1)
            validate_mm_argmax = torch.where(mm_aux_output < 0.5, 0, 1)
            validate_is_argmax = torch.where(is_output < thresh, 0, 1)
            validate_t_argmax = torch.where(t_output < thresh, 0, 1)
            validate_ip_argmax = torch.where(ip_output < thresh, 0, 1)
            y_pred = validate_argmax.squeeze().cpu().numpy()
            y_pred_is = validate_is_argmax.squeeze().cpu().numpy()
            y_pred_mm = validate_mm_argmax.squeeze().cpu().numpy()
            y_pred_t = validate_t_argmax.squeeze().cpu().numpy()
            y_pred_ip = validate_ip_argmax.squeeze().cpu().numpy()
            y_GT = labels.int().cpu().numpy()
            try:
                if y_pred.shape[0] == 0:
                    continue
            except (IndexError):
                continue
            for idx, _ in enumerate(y_pred):
                if thresh_idx==0:
                    record = {}
                    record['final_feature'] = mix_feature[idx].cpu().numpy().tolist()
                    record['is_feature'] = is_feature[idx].cpu().numpy().tolist()
                    record['t_feature'] = t_feature[idx].cpu().numpy().tolist()
                    record['t_feature'] = t_feature[idx].cpu().numpy().tolist()
                    record['ip_feature'] = ip_feature[idx].cpu().numpy().tolist()
                    record['mm_feature'] = mm_aux_feature[idx].cpu().numpy().tolist()

                    record['image_no'], record['text'] = image_no, sents[idx]
                    record['y_GT'], record['y_pred'] = y_GT[idx], mix_output[idx].item()
                    record['y_pred_mm'], record['y_pred_is'], record['y_pred_t'], record['y_pred_ip'] = mm_aux_output[idx].item(), \
                                                                                                             is_output[idx].item(), \
                                                                                                             t_output[idx].item(), \
                                                                                                             ip_output[idx].item()
                    results.append(record)


                if y_pred_is[idx]==y_GT[idx]: is_correct[thresh_idx] += 1
                if y_pred_mm[idx] == y_GT[idx]: mm_correct[thresh_idx] += 1
                if y_pred_t[idx] == y_GT[idx]: t_correct[thresh_idx] += 1
                if y_pred_ip[idx] == y_GT[idx]: ip_correct[thresh_idx] += 1

                if y_GT[idx]==1:
                    fakenews_sum[thresh_idx] +=1
                    if y_pred[idx]==0:
                        fakenews_FN[thresh_idx] += 1
                        realnews_FP[thresh_idx] += 1
                    else:
                        fakenews_TP[thresh_idx] += 1
                        realnews_TN[thresh_idx] += 1
                else:
                    realnews_sum[thresh_idx] +=1
                    if y_pred[idx]==1:
                        realnews_FN[thresh_idx] +=1
                        fakenews_FP[thresh_idx] +=1
                    else:
                        realnews_TP[thresh_idx] += 1
                        fakenews_TN[thresh_idx] += 1
    print(realnews_sum,fakenews_sum,fakenews_FP,realnews_FN,realnews_TP,fakenews_TN)
    export_mix_features = export_mix_features[1:,:]
    export_is_features = export_is_features[1:,:]
    export_t_features = export_t_features[1:,:]
    export_ip_features = export_ip_features[1:,:]
    export_mm_features = export_mm_features[1:,:]
    all_labels = all_labels[1:,:]

    export_mix_features = torch.cat([all_labels ,export_mix_features],1)
    export_is_features = torch.cat([all_labels ,export_is_features],1)
    export_t_features = torch.cat([all_labels ,export_t_features],1)
    export_ip_features = torch.cat([all_labels ,export_ip_features],1)
    export_mm_features = torch.cat([all_labels ,export_mm_features],1)

    export_mix_features = export_mix_features.cpu()
    export_is_features = export_is_features.cpu()
    export_t_features = export_t_features.cpu()
    export_ip_features = export_ip_features.cpu()
    export_mm_features = export_mm_features.cpu()
    mix_feature_array = np.array(export_mix_features)
    is_feature_array = np.array(export_is_features)
    t_feature_array = np.array(export_t_features)
    ip_feature_array = np.array(export_ip_features)
    mm_feature_array = np.array(export_mm_features)
    np.savetxt(Path(output_directory) / 'mix_features.txt',mix_feature_array)
    np.savetxt(Path(output_directory) / 'is_features.txt',is_feature_array)
    np.savetxt(Path(output_directory) / 't_features.txt',t_feature_array)
    np.savetxt(Path(output_directory) / 'ip_features.txt',ip_feature_array)
    np.savetxt(Path(output_directory) / 'mm_aux_features.txt',mm_feature_array)


    import pandas as pd
    df = pd.DataFrame(results)
    pandas_file = Path(output_directory) / 'predictions.xlsx'
    df.to_excel(pandas_file)
    print(f"Excel saved to {pandas_file}")

    val_accuracy, real_accuracy, fake_accuracy, real_precision, fake_precision = [0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH)
    real_recall, fake_recall, real_F1, fake_F1 = [0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH)
    
    for thresh_idx, _ in enumerate(THRESH):
        try:
            mm_correct[thresh_idx] = mm_correct[thresh_idx] / (realnews_sum[thresh_idx] + fakenews_sum[thresh_idx])
            is_correct[thresh_idx] = is_correct[thresh_idx]/(realnews_sum[thresh_idx]+fakenews_sum[thresh_idx])
            t_correct[thresh_idx] = t_correct[thresh_idx] / (realnews_sum[thresh_idx] + fakenews_sum[thresh_idx])
            ip_correct[thresh_idx] = ip_correct[thresh_idx] / (realnews_sum[thresh_idx] + fakenews_sum[thresh_idx])

            val_accuracy[thresh_idx] = (realnews_TP[thresh_idx]+realnews_TN[thresh_idx])/(realnews_TP[thresh_idx]+realnews_TN[thresh_idx]+realnews_FP[thresh_idx]+realnews_FN[thresh_idx])
            real_accuracy[thresh_idx] = (realnews_TP[thresh_idx])/realnews_sum[thresh_idx]
            fake_accuracy[thresh_idx] = (fakenews_TP[thresh_idx])/fakenews_sum[thresh_idx]
            real_precision[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FP[thresh_idx]))
            fake_precision[thresh_idx] = fakenews_TP[thresh_idx] / max(1,(fakenews_TP[thresh_idx] + fakenews_FP[thresh_idx]))
            real_recall[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FN[thresh_idx]))
            fake_recall[thresh_idx] = fakenews_TP[thresh_idx] / max(1,(fakenews_TP[thresh_idx] + fakenews_FN[thresh_idx]))
            real_F1[thresh_idx] = 2*(real_recall[thresh_idx]*real_precision[thresh_idx])/max(1,(real_recall[thresh_idx]+real_precision[thresh_idx]))
            fake_F1[thresh_idx] = 2 * (fake_recall[thresh_idx] * fake_precision[thresh_idx]) / max(1,(fake_recall[thresh_idx] + fake_precision[thresh_idx]))
        except ZeroDivisionError:
            continue
    return val_accuracy, (real_precision, real_recall, real_accuracy, real_F1),\
           (fake_precision, fake_recall, fake_accuracy, fake_F1), \
           val_loss,\
           (is_correct,t_correct,ip_correct,mm_correct)




if __name__ == '__main__':
    cli()


