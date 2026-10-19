"""Launch GAMED from its project root; --dry-run only validates a plan."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from resource_paths import settings

ENTRIES=('run_gamed.py',)

def build_plan(entry, forwarded):
    if entry not in ENTRIES:
        raise ValueError('Unsupported entry: '+entry)
    from run_gamed import validate_arguments
    forwarded=validate_arguments(forwarded)
    return {'entry':entry,'working_directory':str(ROOT),'argv':[entry,*forwarded],
            'implementation_entry':'run_gamed.py','settings':settings()}

def execute_plan(plan):
    os.chdir(ROOT)
    entry=ROOT/plan['implementation_entry']
    sys.path.insert(0,str(ROOT))
    sys.argv=[str(entry),*plan['argv'][1:]]
    runpy.run_path(str(entry),run_name='__main__')

def launch(entry, forwarded):
    execute_plan(build_plan(entry,forwarded))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',help='JSON file, relative to caller CWD')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('entry',choices=ENTRIES)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.config:os.environ['GAMED_CONFIG']=str(Path(args.config).resolve())
    forwarded=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    plan=build_plan(args.entry,forwarded)
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=True,indent=2))
        return
    execute_plan(plan)

if __name__=='__main__':main()
