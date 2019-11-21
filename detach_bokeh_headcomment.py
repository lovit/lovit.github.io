import argparse
from glob import glob


def passline(line):
    if not line.strip():
        return True
    return '<!DOCTYPE html>' in line

def detach(path):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    lines = [line for line in lines if not passline(line)]
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('{}\n'.format(line))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str, default='_posts/figures/bokeh_tutorial/', help='HTML bokeh image directory')

    args = parser.parse_args()
    dirname = args.dirname
    paths = sorted(glob('{}/*.html'.format(dirname)))
    for path in paths:
        detach(path)
        print(f'done with {path}')

if __name__ == '__main__':
    main()
