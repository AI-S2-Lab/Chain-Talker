import argparse
import hashlib
import json
import re
import shutil
import tempfile
import wave
from collections import Counter
from pathlib import Path

from .runtime import ROOT, project_path, sha256, write_json, write_jsonl

SPLITS = ('train', 'valid', 'test')
NAME = re.compile(r'(?P<turn>\d+)_(?P<speaker>\d+)_d(?P<dialogue>\d+)\.wav')


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}. Reuse it or choose a new --output.')
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Input and output directories must be separate and must not contain each other.')
    records, sources = [], {}
    split_ids = {split: [] for split in SPLITS}
    windows = {split: [] for split in SPLITS}
    seen_dialogues = set()
    for split in SPLITS:
        directory = source/split
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing split directory: {directory}')
        folders = [path for path in directory.iterdir() if path.is_dir()]
        if not folders or any(not path.name.isdecimal() for path in folders):
            raise ValueError(f'Expected numeric dialogue directories under {directory}')
        for folder in sorted(folders, key=lambda path: int(path.name)):
            dialogue = folder.name
            if str(int(dialogue)) != dialogue or dialogue in seen_dialogues:
                raise ValueError(f'Duplicate or noncanonical dialogue ID: {dialogue}')
            seen_dialogues.add(dialogue)
            split_ids[split].append(dialogue)
            turns, expected_labs = [], set()
            wavs = sorted(folder.glob('*.wav'))
            if not wavs:
                raise ValueError(f'No WAV files in {folder}')
            for wav in wavs:
                match = NAME.fullmatch(wav.name)
                if not match or match['dialogue'] != dialogue:
                    raise ValueError(f'Expected <turn>_<speaker>_d{dialogue}.wav: {wav}')
                text_path = wav.with_suffix('.lab')
                caption_path = wav.with_name(wav.stem+'.caption.lab')
                expected_labs.update((text_path, caption_path))
                text = text_path.read_text(encoding='utf-8-sig').strip()
                caption = caption_path.read_text(encoding='utf-8-sig').strip()
                if not text or not caption:
                    raise ValueError(f'Empty transcript or caption for {wav}')
                with wave.open(str(wav), 'rb') as audio:
                    if audio.getcomptype() != 'NONE' or audio.getsampwidth() not in (1, 2, 3, 4):
                        raise ValueError(f'Expected PCM WAV: {wav}')
                    frames, rate, channels = audio.getnframes(), audio.getframerate(), audio.getnchannels()
                    duration = frames/rate
                    decoded_bytes = 0
                    while block := audio.readframes(65536):
                        decoded_bytes += len(block)
                    if decoded_bytes != frames*channels*audio.getsampwidth():
                        raise ValueError(f'Truncated WAV: {wav}')
                if not .1 <= duration <= 30:
                    raise ValueError(f'Audio duration outside 0.1-30 seconds: {wav}')
                hashes = {path: sha256(path) for path in (wav, text_path, caption_path)}
                sources.update(hashes)
                speaker = 'Spk-'+match['speaker']
                record = {'id': 'dailyTalk_'+wav.stem, 'dialogue_id': dialogue,
                          'turn_id': int(match['turn']), 'speaker': speaker,
                          'speaker_key': f'dailytalk:{dialogue}:{speaker}',
                          'audio': wav.relative_to(source).as_posix(), 'text': text, 'caption': caption,
                          'emotion': '', 'attributes': {key: '' for key in ('gender', 'pitch', 'tempo', 'energy')},
                          'duration': duration, 'sample_rate': rate, 'channels': channels,
                          'audio_bytes': wav.stat().st_size, 'audio_sha256': hashes[wav],
                          'text_lab': text_path.relative_to(source).as_posix(),
                          'caption_lab': caption_path.relative_to(source).as_posix(),
                          'text_lab_sha256': hashes[text_path], 'caption_lab_sha256': hashes[caption_path],
                          'split': split}
                turns.append(record)
            if set(folder.glob('*.lab')) != expected_labs:
                raise ValueError(f'LAB files do not match WAV files in {folder}')
            turns.sort(key=lambda row: row['turn_id'])
            ids = [row['turn_id'] for row in turns]
            if ids != list(range(ids[0], ids[0]+len(ids))):
                raise ValueError(f'Duplicate or missing turn numbers in {folder}: {ids}')
            records.extend(turns)
            for index in range(2, len(turns)):
                target, history = turns[index], turns[index-2:index]
                refs = [row for row in history if row['speaker_key'] == target['speaker_key']]
                windows[split].append({'target': target['id'], 'history': [row['id'] for row in history],
                                       'reference': refs[-1]['id'] if refs else None})
        if not windows[split]:
            raise ValueError(f'No three-turn windows in split {split}')
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('Duplicate utterance IDs.')
    source_manifest = {'schema': 1, 'source_directory': '.',
                       'files': [{'path': path.relative_to(source).as_posix(), 'sha256': digest, 'bytes': path.stat().st_size}
                                 for path, digest in sorted(sources.items())]}
    source_bytes = (json.dumps(source_manifest, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    report = {'schema': 2, 'source_manifest': 'source_manifest.json', 'source_manifest_sha256': source_hash,
              'source_directory': '.', 'source_layout': 'wav_lab_caption',
              'paths_relative_to': 'data_directory', 'source_files_copied': True, 'split_policy': 'preserve input train/valid/test directories',
              'seed': None, 'history_turns': 2, 'total_turns_including_target': 3, 'window_stride': 1,
              'input_rows': len(records), 'usable_rows': len(records), 'excluded_rows': 0,
              'dialogues': len(seen_dialogues), 'total_windows': sum(map(len, windows.values())),
              'targets_without_three_turn_window': len(records)-sum(map(len, windows.values())),
              'hours': sum(row['duration'] for row in records)/3600,
              'channels': dict(Counter(row['channels'] for row in records)),
              'splits': {split: {'dialogues': len(split_ids[split]),
                                 'source_utterances': sum(row['split'] == split for row in records),
                                 'windows': len(windows[split]),
                                 'inference_eligible': sum(row['reference'] is not None for row in windows[split])}
                         for split in SPLITS},
              'policies': ['No dialogue overlaps between splits.',
                           'Two history turns plus target; stride one; no bridging gaps.',
                           'Speaker IDs are dialogue-local; references must occur in visible history.',
                           'WAV and LAB files are copied unchanged into the output train/valid/test directories.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing.', dir=output.parent)).resolve()
    try:
        for path, expected in sorted(sources.items()):
            destination = temporary/path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if sha256(destination) != expected:
                raise RuntimeError(f'Copied data does not match source: {path}')
        (temporary/'source_manifest.json').write_bytes(source_bytes)
        write_jsonl(temporary/'utterances.jsonl', records)
        write_json(temporary/'dialogue_splits.json', split_ids)
        for split in SPLITS:
            write_jsonl(temporary/(split+'.jsonl'), windows[split])
        report['manifest_sha256'] = sha256(temporary/'utterances.jsonl')
        report['dialogue_splits_sha256'] = sha256(temporary/'dialogue_splits.json')
        report['windows_sha256'] = {split: sha256(temporary/(split+'.jsonl')) for split in SPLITS}
        write_json(temporary/'prepare_report.json', report)
        for path, expected in sources.items():
            if sha256(path) != expected:
                raise RuntimeError(f'Source changed during preparation: {path}')
        if output.exists():
            raise FileExistsError(f'Output appeared during preparation: {output}')
        temporary.rename(output)
    finally:
        if temporary.exists():
            if (temporary.parent != output.parent.resolve() or temporary.is_symlink()
                    or not temporary.name.startswith('.'+output.name+'.preparing.')):
                raise RuntimeError(f'Unexpected temporary path: {temporary}')
            shutil.rmtree(temporary)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Copy DailyTalk WAV/LAB files and prepare self-contained train/valid/test data.')
    parser.add_argument('--input', type=project_path, required=True, help='Directory containing train/valid/test WAV and LAB files.')
    parser.add_argument('--output', type=project_path, default=ROOT/'data/dailytalk')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.input, args.output), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
