# split_yolo_dataset.py
import os
import argparse
from sklearn.model_selection import train_test_split
import shutil

def main(args):
    print("--- Splitting dataset into training and validation sets ---")

    os.makedirs(os.path.join(args.output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'val', 'labels'), exist_ok=True)

    all_filenames = [os.path.splitext(f)[0] for f in os.listdir(args.source_img_dir) if f.endswith(('.png', '.jpg'))]
    
    train_filenames, val_filenames = train_test_split(all_filenames, test_size=args.val_split, random_state=42)

    print(f"Total files: {len(all_filenames)}")
    print(f"Training set size: {len(train_filenames)}")
    print(f"Validation set size: {len(val_filenames)}")

    def move_files(filenames, set_type):
        for name in filenames:
            shutil.copy(os.path.join(args.source_img_dir, name + '.png'), os.path.join(args.output_dir, set_type, 'images'))
            shutil.copy(os.path.join(args.source_label_dir, name + '.txt'), os.path.join(args.output_dir, set_type, 'labels'))

    print("Copying training files...")
    move_files(train_filenames, 'train')
    print("Copying validation files...")
    move_files(val_filenames, 'val')

    print(f"\n✅ Dataset splitting complete! Final dataset is in '{args.output_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val sets.")
    parser.add_argument('--source-img-dir', type=str, required=True, help="Path to source images.")
    parser.add_argument('--source-label-dir', type=str, required=True, help="Path to source labels.")
    parser.add_argument('--output-dir', type=str, default='multiclass_yolo_data', help="Path to output split data.")
    parser.add_argument('--val-split', type=float, default=0.2, help="Validation set split ratio.")
    args = parser.parse_args()
    main(args)