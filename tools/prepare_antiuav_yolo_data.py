"""
Generate YOLOv5 image-list .txt files for AntiUAV600 training.

Reads antiuav_train_val_split.json (train/tune keys) and writes:
  {out_dir}/train_images.txt  — one absolute image path per line (train split)
  {out_dir}/val_images.txt    — one absolute image path per line (tune split)

YOLOv5 resolves label paths by replacing the .jpg extension with .txt in the
same directory, which matches the AntiUAV600 layout (images and labels co-located).

Usage:
    python tools/prepare_antiuav_yolo_data.py \
        --dataset_root /gpfs/work5/0/prjs1970/data/AntiUAV600/train \
        --split_json   /gpfs/work5/0/prjs1970/code/EDTC/antiuav_train_val_split.json \
        --out_dir      /gpfs/work5/0/prjs1970/data/AntiUAV600/yolo_lists
"""
import os
import json
import argparse
import glob


def collect_images(dataset_root, sequences):
    paths = []
    missing = []
    for seq in sequences:
        seq_dir = os.path.join(dataset_root, seq)
        if not os.path.isdir(seq_dir):
            missing.append(seq)
            continue
        imgs = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
        paths.extend(imgs)
    return paths, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True,
                        help='Path to AntiUAV600 train split directory')
    parser.add_argument('--split_json', required=True,
                        help='Path to antiuav_train_val_split.json')
    parser.add_argument('--out_dir', required=True,
                        help='Directory to write train_images.txt and val_images.txt')
    args = parser.parse_args()

    with open(args.split_json) as f:
        split = json.load(f)

    train_seqs = split['train']
    val_seqs   = split['tune']

    print(f'Train sequences: {len(train_seqs)}')
    print(f'Val sequences:   {len(val_seqs)}')

    train_imgs, train_missing = collect_images(args.dataset_root, train_seqs)
    val_imgs,   val_missing   = collect_images(args.dataset_root, val_seqs)

    if train_missing:
        print(f'WARNING: {len(train_missing)} train sequences not found: {train_missing[:5]}...')
    if val_missing:
        print(f'WARNING: {len(val_missing)} val sequences not found: {val_missing[:5]}...')

    os.makedirs(args.out_dir, exist_ok=True)

    train_out = os.path.join(args.out_dir, 'train_images.txt')
    val_out   = os.path.join(args.out_dir, 'val_images.txt')

    with open(train_out, 'w') as f:
        f.write('\n'.join(train_imgs) + '\n')
    with open(val_out, 'w') as f:
        f.write('\n'.join(val_imgs) + '\n')

    print(f'Train images: {len(train_imgs):,} → {train_out}')
    print(f'Val   images: {len(val_imgs):,} → {val_out}')


if __name__ == '__main__':
    main()
