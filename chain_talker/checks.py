"""Validate WAV/LAB data boundaries, source files, and tokenizer controls."""
import argparse, json
from .runtime import ROOT, project_path, controls, read_jsonl, tokenizer, write_json, validate_prepared_data

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=project_path,default=ROOT/'data/dailytalk')
    p.add_argument('--output',type=project_path)
    args=p.parse_args(argv)
    args.output=args.output or args.data/'data_check.json'
    tok=tokenizer();sps,spe=controls(tok)
    assert tok.encoding.n_vocab<=60515
    prepare=validate_prepared_data(args.data,verify_sources=True)
    records={r['id']:r for r in read_jsonl(args.data/'utterances.jsonl')}
    splits=json.loads((args.data/'dialogue_splits.json').read_text(encoding='utf8'))
    sets=[set(x) for x in splits.values()]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    counts={}
    for split in ['train','valid','test']:
        windows=read_jsonl(args.data/(split+'.jsonl'));counts[split]=len(windows)
        for w in windows:
            assert len(w['history'])==2
            target=records[w['target']];assert target['dialogue_id'] in splits[split]
            turns=[records[x] for x in w['history']]+[target]
            assert all(t['dialogue_id']==target['dialogue_id'] for t in turns)
            assert all(b['turn_id']==a['turn_id']+1 for a,b in zip(turns,turns[1:]))
            if w['reference']:
                reference=records[w['reference']]
                assert reference['speaker_key']==target['speaker_key'] and reference['turn_id']<target['turn_id']
                assert w['reference'] in w['history']
    report={'status':'passed','source_manifest_sha256':prepare['source_manifest_sha256'],'data_rows':len(records),
            'windows':counts,'dialogue_splits_disjoint':True,'history_contiguous':True,
            'no_future_reference':True,'tokenizer_vocab':tok.encoding.n_vocab,'model_embedding_rows':60515,'sps_id':sps,'spe_id':spe,
            'source_files_verified':True}
    write_json(args.output,report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
