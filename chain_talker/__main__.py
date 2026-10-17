import argparse, sys

def main():
    parser=argparse.ArgumentParser(description='Chain-Talker')
    parser.add_argument('command',choices=['prepare','features','train','infer','make-job','check'])
    args=parser.parse_args(sys.argv[1:2]);rest=sys.argv[2:]
    if args.command=='prepare':from .prepare import main as run
    elif args.command=='features':from .features import main as run
    elif args.command=='train':from .train import main as run
    elif args.command=='infer':from .infer import main as run
    elif args.command=='make-job':from .infer import make_job as run
    else:from .checks import main as run
    run(rest)

if __name__=='__main__':main()
