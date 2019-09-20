from PIL import Image
import argparse
import glob
import os


def move_character_toward_middle(input_paths, output_dir):
    for p in input_paths:
        image = Image.open(p)

        cropped_image = image.crop((0, 0, 256-((256-150)/2), 256-((256-150)/2)))
        new_image = Image.new('L', (256, 256), color=0)
        new_image.paste(cropped_image, ((256-150)//2-20, (256-150)//2-20))
        print(os.path.basename(p))
        new_image.save(os.path.join(output_dir, os.path.basename(p)))



parser = argparse.ArgumentParser(description='Move characters toward middle of images')
parser.add_argument('--dir', dest='dir', required=True, help='path of examples')
parser.add_argument('--save_dir', dest='save_dir', required=True, help='path to save pickled files')

args = parser.parse_args()

if __name__ == "__main__":
    move_character_toward_middle(sorted(glob.glob(os.path.join(args.dir, "*.png"))), args.save_dir)
