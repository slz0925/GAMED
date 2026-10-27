"""GAMED training and validation for Fakeddit and Yang."""
import argparse
import json
import os
import re
from pathlib import Path
from output_paths import create_run_directory, create_epoch_directory

PROJECT_ROOT = Path(__file__).resolve().parent
GT_size = 224
word_token_length = 197


def _non_negative_int(value):
    try:
        number = int(value)
        if number < 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('must be a non-negative integer') from None
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('-network_arch', required=True, choices=('gamed', 'gamed_without_veto'))
    parser.add_argument('-train_dataset', required=True, choices=('fakeddit', 'yang'))
    parser.add_argument('-test_dataset', required=True, choices=('fakeddit', 'yang'))
    parser.add_argument('-output_file', default='checkpoint/', help='Root directory for checkpoints, features and predictions')
    parser.add_argument('-checkpoint', default='')
    parser.add_argument('-finetune', type=int, default=0)
    parser.add_argument('-val', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-train_workers', type=_non_negative_int, default=56, help='Training DataLoader workers')
    parser.add_argument('-val_workers', type=_non_negative_int, default=32, help='Validation DataLoader workers')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-device', type=int, default=0, help='CUDA device index')
    return parser


def validate_arguments(arguments):
    build_parser().parse_args(arguments)
    return list(arguments)


def initialise_runtime():
    """Load runtime dependencies and the English tokenizer before setting the seed."""
    global _read_path, np, torch, nn, DataLoader
    global AutoTokenizer, ImageFile, token_uncased_fakeddit, Progbar
    from resource_paths import resolve_read_path as _read_path, resolve_english_encoder_path
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    # Load backbone dependencies before setting the experiment seed.
    import models_mae
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


def to_var(x, device=None):
    
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else x.device
    x = x.to(device)
    return x


def collate_news_batch(data):
    """Tokenise each news batch once and preserve its sample order."""
    sentences, images, labels, image_paths = zip(*data)
    tokens = token_uncased_fakeddit(
        list(sentences), return_tensors='pt', truncation=True,
        padding='max_length', max_length=word_token_length)
    text = tuple(tokens[name] for name in ('input_ids', 'attention_mask', 'token_type_ids'))
    return text, (torch.stack(images), torch.tensor(labels, dtype=torch.long), list(sentences)), list(image_paths)


def load_dataset(name, mode):
    if name == 'fakeddit':
        from data.Fakeddit_dataset import Fakeddit_dataset as Dataset
    elif name == 'yang':
        from data.Yang_dataset import Yang_dataset as Dataset
    else:
        raise ValueError('Unsupported dataset: ' + str(name))
    return Dataset(root_path=PROJECT_ROOT / 'dataset', mode=mode, image_size=GT_size)


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

    training = args.val == 0
    if training:
        train_dataset = load_dataset(args.train_dataset, 'train')
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.train_workers,
            shuffle=True, collate_fn=collate_news_batch)
    validate_dataset = load_dataset(args.test_dataset, 'validate')
    validate_loader = DataLoader(
        validate_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_news_batch, num_workers=args.val_workers, drop_last=False,
        pin_memory=True)

    model_type = resolve_model_class(args.network_arch)
    model = model_type()
    if args.checkpoint:
        print('Loading checkpoint: ' + args.checkpoint)
        load_model(model, args.checkpoint)
    device = torch.device('cuda', args.device)
    model = model.to(device)
    print(get_parameter_number(model))
    criterion = nn.BCEWithLogitsLoss().to(device)

    if training:
        import pytorch_warmup as warmup
        optim_params_normal, optim_params_fast, optim_params_extremefast = [], [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if 'image_encoder' in name or 'text_encoder' in name:
                    optim_params_normal.append(parameter)
                elif 'ip_backbone' in name or 'irrelevant_tensor' in name:
                    optim_params_extremefast.append(parameter)
                else:
                    optim_params_fast.append(parameter)
        fine_tuning = args.finetune > 0
        optimizer = torch.optim.AdamW(optim_params_normal, lr=1e-5,
                                     betas=(0.9, 0.999), weight_decay=0.01)
        optimizer_fast = torch.optim.AdamW(optim_params_fast,
                                          lr=1e-5 if fine_tuning else 5e-5,
                                          betas=(0.9, 0.999), weight_decay=0.01)
        optimizer_extremefast = torch.optim.AdamW(optim_params_extremefast,
                                                 lr=1e-5 if fine_tuning else 1e-4,
                                                 betas=(0.9, 0.999), weight_decay=0.01)
        num_steps = int(len(train_loader) * args.epochs * 1.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        scheduler_fast = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fast, T_max=num_steps)
        warmup_scheduler_fast = warmup.UntunedLinearWarmup(optimizer_fast)
        scheduler_extremefast = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_extremefast, T_max=num_steps)
        warmup_scheduler_extremefast = warmup.UntunedLinearWarmup(optimizer_extremefast)

    run_directory = create_run_directory(
        args.output_file, args.train_dataset, args.test_dataset,
        args.network_arch, validation=not training)
    best_acc_so_far, best_epoch_record = 0.0, 0
    validation_setting = {'thresh': 0.5}
    for epoch in range(args.epochs if training else 1):
        if training:
            loss_total, correct_total, sample_total = 0.0, 0, 0
            progbar = Progbar(len(train_dataset), width=10)
            model.train()
            for texts, (image, labels, _), _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                optimizer_fast.zero_grad(set_to_none=True)
                optimizer_extremefast.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    input_ids, attention_mask, token_type_ids = (
                        to_var(value, device=device) for value in texts)
                    image, labels = to_var(image, device=device), to_var(labels, device=device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids, image=image)
                    mix_output, is_output, t_output, ip_output, mm_aux_output, irrelevant_mean = outputs
                    labels = labels.float().unsqueeze(1)
                    mix_loss = criterion(mix_output, labels)
                    is_loss = criterion(is_output, labels)
                    t_loss = criterion(t_output, labels)
                    ip_loss = criterion(ip_output, labels)
                    loss_single_modal = (ip_loss + t_loss + is_loss) / 3
                    loss = mix_loss + loss_single_modal

                    loss.backward()
                    parameters_to_clip = (
                        model.parameters() if epoch >= 6
                        else optim_params_fast + optim_params_extremefast
                    )
                    nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=1)
                    if epoch >= 6:
                        optimizer.step()
                    optimizer_fast.step()
                    optimizer_extremefast.step()

                    batch_size = len(image)
                    correct = ((torch.sigmoid(mix_output) >= 0.5) == labels).sum().item()
                    loss_total += loss.item() * batch_size
                    correct_total += correct
                    sample_total += batch_size
                    logs = [('Mix_loss', mix_loss.item()), ('IS_loss', is_loss.item()),
                            ('T_loss', t_loss.item()), ('IP_loss', ip_loss.item()),
                            ('MM_mean_prob', torch.sigmoid(mm_aux_output).mean().item()),
                            ('Irr_mean', irrelevant_mean.item()), ('Train_acc', correct / batch_size)]
                    progbar.add(batch_size, values=logs)
                    if epoch >= 6:
                        with warmup_scheduler.dampening():
                            scheduler.step()
                    with warmup_scheduler_fast.dampening():
                        scheduler_fast.step()
                    with warmup_scheduler_extremefast.dampening():
                        scheduler_extremefast.step()
            print('Epoch [%d/%d], Loss: %.4f, Train_Acc: %.4f' %
                  (epoch + 1, args.epochs, loss_total / sample_total, correct_total / sample_total))

        with torch.no_grad():
            model.eval()
            progbar = Progbar(len(validate_dataset), width=10)
            metrics = evaluate(validate_loader, model, criterion, progbar=progbar,
                               setting=validation_setting,
                               output_directory=create_epoch_directory(run_directory, epoch + 1))
            validate_acc_list, real_items, fake_items, _, single_items = metrics
            validate_acc = max(validate_acc_list)
            val_thresh = validate_acc_list.index(validate_acc)
            val_threshold = _validation_thresholds(validation_setting['thresh'])[val_thresh]
            if validate_acc > best_acc_so_far:
                best_acc_so_far, best_epoch_record = validate_acc, epoch + 1
            print('Epoch [%d/%d], validation accuracy: %.4f at threshold %.3f (best %.4f at epoch %d)' %
                  (epoch + 1, args.epochs if training else 1, validate_acc, val_threshold,
                   best_acc_so_far, best_epoch_record))
            print('Branch accuracies: IS {}, T {}, IP {}, MM {}'.format(
                *(values[val_thresh] for values in single_items)))
            for name, items in (('Real News', real_items), ('Fake News', fake_items)):
                print('------' + name + '-----------')
                for field, values in zip(('Precision', 'Recall', 'Accuracy', 'F1'), items):
                    print('{}: {}'.format(field, values))
            checkpoint_directory = run_directory / 'checkpoints'
            checkpoint_directory.mkdir(exist_ok=True)
            checkpoint = checkpoint_directory / f'epoch_{epoch + 1:03d}.pkl'
            torch.save(model.state_dict(), checkpoint)
            print('Model saved at {}'.format(checkpoint))


def load_model(model, load_path, strict=True):
    load_net = torch.load(_read_path(load_path), map_location='cpu')
    return model.load_state_dict(load_net, strict=strict)


def _excel_text_preview(text):
    """Make an Excel-safe preview while the JSONL export retains the original."""
    preview = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]',
                     lambda match: '\\u{:04x}'.format(ord(match.group())), text)
    encoded = preview.encode('utf-16-le')
    truncated = len(encoded) > 32767 * 2
    if truncated:
        suffix = '… [truncated; full text in predictions.jsonl]'
        preview = encoded[:(32767 - len(suffix)) * 2].decode('utf-16-le', errors='ignore') + suffix
    return preview, truncated


def _validation_thresholds(threshold):
    return [threshold + i / 1000 for i in range(-50, 50)]


def evaluate(validate_loader, model, criterion, progbar=None, setting=None, *, output_directory):
    """Return binary threshold metrics, sample-mean loss and aligned exports."""
    model.eval()
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    device = reference.device if reference is not None else torch.device('cpu')
    reduction = getattr(criterion, 'reduction', 'mean')
    if reduction not in ('mean', 'sum'):
        raise ValueError('Evaluation requires a scalar mean- or sum-reduced loss')
    threshold = (setting or {}).get('thresh', 0.5)
    thresholds = _validation_thresholds(threshold)
    true_positive = np.zeros((2, len(thresholds)), dtype=np.int64)
    false_positive = np.zeros_like(true_positive)
    false_negative = np.zeros_like(true_positive)
    branch_correct = np.zeros((4, len(thresholds)), dtype=np.int64)
    class_counts = np.zeros(2, dtype=np.int64)
    feature_batches = [[] for _ in range(5)]
    label_batches, results = [], []
    loss_total, sample_count = 0, 0

    for texts, (image, labels, sentences), _ in validate_loader:
        input_ids, attention_mask, token_type_ids = (
            to_var(value, device=device) for value in texts)
        image, labels = to_var(image, device=device), to_var(labels, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, image=image, return_features=True)
            logits, features = outputs[:5], outputs[-1]
            batch_size = image.shape[0]
            if any(tuple(output.shape) != (batch_size, 1) for output in logits):
                raise ValueError('Evaluation expects each binary logit output to have shape [batch, 1]')
            if len(features) != 5 or any(tuple(feature.shape) != (batch_size, 64) for feature in features):
                raise ValueError('Evaluation expects five feature tensors of shape [batch, 64]')
            labels = labels.float().unsqueeze(1)
            loss = criterion(logits[0], labels)
            loss_total = loss_total + loss.detach() * (batch_size if reduction == 'mean' else 1)
            sample_count += batch_size
            if progbar is not None:
                logs = [('Mix_loss', loss.item())] + [
                    (name, criterion(output, labels).item()) for name, output in
                    zip(('IS_loss', 'T_loss', 'IP_loss'), logits[1:4])]
                progbar.add(batch_size, values=logs)
            # Move each batch once; retain neither full GPU exports nor computation graphs.
            probabilities = [torch.sigmoid(output).cpu() for output in logits]
            cpu_features = [feature.detach().to(
                device='cpu', dtype=torch.promote_types(feature.dtype, torch.float32), copy=True).numpy()
                for feature in features]
            cpu_labels = labels.cpu().numpy()
        targets = cpu_labels.astype(np.int32).reshape(-1)
        label_batches.append(cpu_labels)
        for batches, values in zip(feature_batches, cpu_features):
            batches.append(values)

        decisions = []
        for index, probability in enumerate(probabilities):
            limits = probability.new_tensor([0.5] if index == 4 else thresholds)
            # Equality is classified as positive; thresholds use the probability dtype.
            decisions.append((~(probability < limits)).numpy())
        prediction = decisions[0]
        for target in (0, 1):
            actual = targets[:, None] == target
            selected = prediction == bool(target)
            true_positive[target] += (selected & actual).sum(axis=0)
            false_positive[target] += (selected & ~actual).sum(axis=0)
            false_negative[target] += (~selected & actual).sum(axis=0)
            class_counts[target] += int(actual.sum())
        for branch, decision in enumerate(decisions[1:]):
            branch_correct[branch] += (decision == targets[:, None]).sum(axis=0)

        probability_values = [probability[:, 0].tolist() for probability in probabilities]
        for index, sentence in enumerate(sentences):
            record = {name: values[index].tolist() for name, values in zip(
                ('final_feature', 'is_feature', 't_feature', 'ip_feature', 'mm_feature'), cpu_features)}
            record.update(image_no=len(results), text=sentence, y_GT=int(targets[index]),
                          y_pred=probability_values[0][index], y_pred_is=probability_values[1][index],
                          y_pred_t=probability_values[2][index], y_pred_ip=probability_values[3][index],
                          y_pred_mm=probability_values[4][index])
            results.append(record)

    if sample_count == 0:
        raise ValueError('Cannot evaluate an empty validation loader')
    labels = np.concatenate(label_batches, axis=0)
    for name, batches in zip(('mix_features.txt', 'is_features.txt', 't_features.txt',
                              'ip_features.txt', 'mm_aux_features.txt'), feature_batches):
        values = np.concatenate((labels, np.concatenate(batches, axis=0)), axis=1)
        np.savetxt(Path(output_directory) / name, values)
    with (Path(output_directory) / 'predictions.jsonl').open('w', encoding='utf-8') as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    import pandas as pd
    columns = ['final_feature', 'is_feature', 't_feature', 'ip_feature', 'mm_feature',
               'image_no', 'text', 'y_GT', 'y_pred', 'y_pred_mm', 'y_pred_is', 'y_pred_t', 'y_pred_ip']
    frame = pd.DataFrame(results, columns=columns)
    previews, truncated = zip(*(_excel_text_preview(record['text']) for record in results))
    frame['text'] = previews
    frame.rename(columns={'text': 'text_preview'}, inplace=True)
    frame['text_truncated'] = truncated
    with pd.ExcelWriter(Path(output_directory) / 'predictions.xlsx', engine='openpyxl') as writer:
        frame.to_excel(writer)
        # News text is literal content, including strings beginning with '='.
        column = frame.columns.get_loc('text_preview') + 2  # Include the export index.
        for row in writer.sheets['Sheet1'].iter_rows(min_row=2, min_col=column, max_col=column):
            row[0].data_type = 's'
    metrics = []
    for target in (0, 1):
        tp, fp, fn = true_positive[target], false_positive[target], false_negative[target]
        precision = tp / np.maximum(1, tp + fp)
        recall = tp / max(1, class_counts[target])
        f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
        metrics.append((precision.tolist(), recall.tolist(), recall.tolist(), f1.tolist()))
    accuracy = (true_positive.sum(axis=0) / sample_count).tolist()
    return (accuracy, metrics[0], metrics[1], loss_total / sample_count,
            tuple(values.tolist() for values in branch_correct / sample_count))


if __name__ == '__main__':
    cli()


