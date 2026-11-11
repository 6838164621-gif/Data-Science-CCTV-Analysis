#!/usr/bin/env python3
"""Local CCTV quality screening and reproducible cleaning. Python 3.9+."""
import argparse
import base64
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png'}
ACTIONS = {'keep', 'adjust', 'drop', 'review'}


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=True, indent=2))
    temporary.replace(path)


def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe image path in input')
    return p


def output_relative(name, legacy_zip=False):
    # Some ZIPs store UTF-8 Thai names without the UTF-8 flag. Keep the source
    # identifier untouched, but recover readable names for filesystem output.
    if legacy_zip:
        try:
            name = name.encode('cp437').decode('utf-8')
        except UnicodeError:
            pass
    p = safe_relative(name)
    if any(len(part.encode('utf-8')) > 240 for part in p.parts):
        raise ValueError('Output filename is too long; normalize source filenames first')
    return p


class Source:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.zip:
            self.names = [i.filename for i in self.zip.infolist() if not i.is_dir()
                          and PurePosixPath(i.filename).suffix.lower() in EXTENSIONS
                          and '__MACOSX' not in PurePosixPath(i.filename).parts
                          and not PurePosixPath(i.filename).name.startswith('.')]
        else:
            if not self.path.is_dir():
                raise ValueError('Source must be a ZIP or image folder')
            self.names = [p.relative_to(self.path).as_posix() for p in self.path.rglob('*')
                          if p.is_file() and p.suffix.lower() in EXTENSIONS
                          and not any(x.startswith('.') or x == '__MACOSX'
                                      for x in p.relative_to(self.path).parts)]
        if len(self.names) != len(set(self.names)):
            raise ValueError('Duplicate archive paths must be resolved before cleaning')
        for name in self.names:
            safe_relative(name)
        self.names.sort()

    def read(self, name):
        safe_relative(name)
        if self.zip:
            return self.zip.read(name)
        p = (self.path / name).resolve()
        if not p.is_relative_to(self.path):
            raise ValueError('Image symlink points outside source')
        return p.read_bytes()

    def close(self):
        if self.zip:
            self.zip.close()


def image_key(name):
    p = PurePosixPath(name)
    match = re.search(r'_(\d{8})_(\d{6})\.[^.]+$', p.name)
    return p.parent.name + '/' + '_'.join(match.groups()) if match else None


def clahe_luminance(image, clip=2.0, grid=8):
    """Same NumPy/Pillow preview algorithm accepted in examples 2 and 3."""
    ycc = np.array(image.convert('YCbCr'))
    y = ycc[:, :, 0]
    h, w = y.shape
    p = np.pad(y, ((0, (-h) % grid), (0, (-w) % grid)), mode='reflect')
    th, tw = p.shape[0] // grid, p.shape[1] // grid
    maps = np.empty((grid, grid, 256), dtype=np.float64)
    for j in range(grid):
        for i in range(grid):
            tile = p[j*th:(j+1)*th, i*tw:(i+1)*tw]
            hist = np.bincount(tile.ravel(), minlength=256)
            limit = max(1, int(clip * tile.size / 256))
            excess = int(np.maximum(hist - limit, 0).sum())
            hist = np.minimum(hist, limit) + excess // 256
            if excess % 256:
                hist[np.linspace(0, 255, excess % 256, dtype=int)] += 1
            maps[j, i] = np.cumsum(hist) * 255 / tile.size
    xf, yf = np.arange(w)/tw - .5, np.arange(h)/th - .5
    x0, y0 = np.floor(xf).astype(int), np.floor(yf).astype(int)
    wx, wy = xf-x0, yf-y0
    x1, y1 = np.clip(x0+1, 0, grid-1), np.clip(y0+1, 0, grid-1)
    x0, y0 = np.clip(x0, 0, grid-1), np.clip(y0, 0, grid-1)
    a, b = maps[y0[:, None], x0[None, :], y], maps[y0[:, None], x1[None, :], y]
    c, d = maps[y1[:, None], x0[None, :], y], maps[y1[:, None], x1[None, :], y]
    result = (a*(1-wx)+b*wx)*(1-wy[:, None]) + (c*(1-wx)+d*wx)*wy[:, None]
    ycc[:, :, 0] = np.clip(np.rint(result), 0, 255).astype('uint8')
    return Image.fromarray(ycc, 'YCbCr').convert('RGB')


def measure(image):
    # Normalize measurement size so sharpness scores are more comparable.
    small = image.convert('L').resize((320, 240))
    y = np.asarray(small, dtype=np.float32)
    lap = y[:-2, 1:-1]+y[2:, 1:-1]+y[1:-1, :-2]+y[1:-1, 2:]-4*y[1:-1, 1:-1]
    tiles = y.reshape(6, 40, 8, 40).transpose(0, 2, 1, 3)
    return {
        'mean_brightness': round(float(y.mean()), 3),
        'contrast_std': round(float(y.std()), 3),
        'contrast_range': round(float(np.percentile(y, 95)-np.percentile(y, 5)), 3),
        'dark_fraction': round(float((y < 20).mean()), 4),
        'bright_fraction': round(float((y > 240).mean()), 4),
        'laplacian_variance': round(float(lap.var()), 3),
        'dark_tile_fraction': round(float(((tiles < 20).mean((2, 3)) > .8).mean()), 4),
        'bright_tile_fraction': round(float(((tiles > 240).mean((2, 3)) > .8).mean()), 4),
    }


def flags_for(m):
    # Provisional review triggers, NOT validated keep/drop thresholds.
    flags = []
    if m['dark_fraction'] > .35: flags.append('dark_regions')
    if m['bright_fraction'] > .12: flags.append('bright_regions_or_glare')
    if m['contrast_range'] < 65: flags.append('low_contrast')
    if m['laplacian_variance'] < 60: flags.append('soft_detail')
    return flags


def scan(args):
    source = Source(args.source)
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new empty report directory')
    if not source.zip and out.is_relative_to(source.path):
        raise ValueError('Report directory must be outside the image source')
    out.mkdir(parents=True, exist_ok=True)
    seeds = json.loads((HERE/'reference_decisions.json').read_text())
    rows = []
    names = source.names[:args.limit] if args.limit else source.names
    for i, name in enumerate(names, 1):
        raw = source.read(name)
        row = {'path': name, 'camera': PurePosixPath(name).parent.name,
               'sha256': hashlib.sha256(raw).hexdigest(), 'flags': [],
               'decision': 'keep', 'decision_origin': 'provisional_default',
               'vehicle_usability': 'unassessed', 'pedestrian_usability': 'unassessed'}
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                row['width'], row['height'] = im.size
                row['metrics'] = measure(im)
                if min(im.size) < 16:
                    row['flags'].append('tiny_dimensions')
                # Flag uniform frames, including uniform colors other than white/black.
                if np.asarray(im.convert('RGB')).reshape(-1, 3).std(axis=0).max() < .5:
                    row['flags'].append('uniform_frame')
                row['flags'] += flags_for(row['metrics'])
        except (OSError, ValueError, SyntaxError) as exc:
            row['flags'] = ['decode_error']
            row['error'] = str(exc)
        if row['flags']:
            row['decision'], row['decision_origin'] = 'review', 'quality_flag'
        seed = seeds.get(image_key(name))
        if seed:
            row.update(decision=seed['decision'], decision_origin='user_reference',
                       example=seed['example'], note=seed.get('note', ''))
        rows.append(row)
        if i % 2000 == 0:
            print(f'Screened {i:,}/{len(names):,}', flush=True)
    # Relative camera-specific checks handle cameras whose baseline sharpness differs.
    groups = defaultdict(list)
    for row in rows:
        if 'metrics' in row: groups[row['camera']].append(row)
    for group in groups.values():
        if len(group) < 20: continue
        baseline = float(np.median([r['metrics']['laplacian_variance'] for r in group]))
        for row in group:
            if row['metrics']['laplacian_variance'] < baseline * .3:
                row['flags'].append('unusually_soft_for_camera')
                if row['decision_origin'] == 'provisional_default':
                    row.update(decision='review', decision_origin='quality_flag')
    manifest = {'schema_version': 1, 'source': str(source.path),
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'partial_scan': len(names) != len(source.names),
                'enhancement': {'method': 'numpy_luminance_clahe', 'clip': 2.0, 'grid': 8},
                'images': rows}
    save_json(out/'manifest.json', manifest)
    summary = {'images': len(rows), 'cameras': len(groups),
               'decisions': dict(Counter(r['decision'] for r in rows)),
               'flags': dict(Counter(f for r in rows for f in r['flags'])),
               'note': 'Flags are provisional. No input files were changed.'}
    save_json(out/'summary.json', summary)
    make_review(source, manifest, out, args.review_limit, args.review_offset)
    source.close()
    print(json.dumps(summary, indent=2))
    print(f'Open {out / "review.html"}')


def preview_data(image):
    image = image.copy()
    image.thumbnail((480, 360))
    b = io.BytesIO()
    image.save(b, format='JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()


def make_review(source, manifest, out, limit, offset):
    rows = manifest['images']
    references = [r for r in rows if r['decision_origin'] == 'user_reference']
    # Sample distinct issue types instead of ranking by flag count: blank frames
    # trigger many flags and would otherwise crowd out every useful scene.
    grouped = defaultdict(list)
    reference_paths = {r['path'] for r in references}
    for row in rows:
        if row['path'] in reference_paths:
            continue
        category = next((f for f in ['decode_error', 'uniform_frame', 'dark_regions',
                        'bright_regions_or_glare', 'low_contrast',
                        'unusually_soft_for_camera', 'soft_detail'] if f in row['flags']), 'no_flags')
        grouped[category].append(row)
    queues = {}
    for category, group in grouped.items():
        cameras = defaultdict(list)
        for row in group:
            cameras[row['camera']].append(row)
        # Stable ordering, with camera diversity within each issue type.
        per_camera = [sorted(g, key=lambda r: r['sha256']) for _, g in sorted(cameras.items())]
        queues[category] = [q[j] for j in range(max(map(len, per_camera)))
                            for q in per_camera if j < len(q)]
    blanks = queues.pop('uniform_frame', [])
    failures = queues.pop('decode_error', [])
    ordered = blanks[:3] + failures[:3]
    for j in range(max((len(q) for q in queues.values()), default=0)):
        ordered.extend(q[j] for _, q in sorted(queues.items()) if j < len(q))
    ordered += blanks[3:] + failures[3:]
    selected = references + ordered[offset:offset+limit]
    cards = []
    for row in selected:
        try:
            original = Image.open(io.BytesIO(source.read(row['path']))).convert('RGB')
            before = preview_data(original)
            after = preview_data(clahe_luminance(original))
        except (OSError, ValueError, SyntaxError):
            placeholder = Image.new('RGB', (352, 288), '#dddddd')
            ImageDraw.Draw(placeholder).text((20, 120), 'Image cannot be decoded', fill='black')
            before = after = preview_data(placeholder)
        cards.append({**{k: row[k] for k in ('path', 'camera', 'decision', 'decision_origin',
                                          'vehicle_usability', 'pedestrian_usability', 'flags')},
                      'key': image_key(row['path']), 'before': before, 'after': after,
                      'example': row.get('example')})
    template = (HERE/'review_template.html').read_text()
    payload = json.dumps({'source': manifest['source'], 'cards': cards,
                          'total': len(rows), 'offset': offset}, ensure_ascii=True)
    # Escape script terminators even if filenames contain HTML.
    (out/'review.html').write_text(template.replace('/*DATA*/null', payload.replace('<', '\\u003c')))


def review(args):
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Review requires a version-1 manifest')
    source = Source(manifest['source'])
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new empty review directory')
    if not source.zip and out.is_relative_to(source.path):
        raise ValueError('Review directory must be outside the source')
    out.mkdir(parents=True, exist_ok=True)
    make_review(source, manifest, out, args.review_limit, args.review_offset)
    source.close()
    print(f'Open {out / "review.html"}')


def apply(args):
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get('schema_version') != 1 or manifest.get('partial_scan'):
        raise ValueError('Apply requires a full version-1 scan')
    rows = manifest['images']
    paths = {r['path'] for r in rows}
    if len(paths) != len(rows) or not rows:
        raise ValueError('Manifest must contain distinct image paths and at least one image')
    overrides = {}
    for file in args.decisions:
        edits = json.loads(Path(file).read_text())
        if edits.get('source') != manifest['source']:
            raise ValueError('Decisions belong to a different source')
        for edit in edits['decisions']:
            if edit['path'] not in paths or edit['decision'] not in ACTIONS:
                raise ValueError('Invalid review decision')
            for key in ('vehicle_usability', 'pedestrian_usability'):
                if edit.get(key, 'unassessed') not in {'unassessed', 'usable', 'unusable'}:
                    raise ValueError('Invalid task usability value')
            overrides[edit['path']] = edit
    source = Source(manifest['source'])
    if set(source.names) != paths:
        raise ValueError('Source image list changed; rescan before applying')
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Cleaned output must be a new empty directory')
    if not source.zip and (out.is_relative_to(source.path) or source.path.is_relative_to(out)):
        raise ValueError('Cleaned output and source must be separate folders')
    # Reject output-name collisions before touching files (including adjustments).
    destinations = {}
    for row in rows:
        relative = output_relative(row['path'], bool(source.zip))
        choice = overrides.get(row['path'], {}).get('decision', row['decision'])
        reserved = [relative]
        if choice == 'adjust':
            reserved.append(relative.with_name(relative.name + '.clahe.png'))
        # Case-folding also catches collisions on common Mac filesystems.
        for candidate in reserved:
            identity = str(candidate).casefold()
            if identity in destinations and destinations[identity] != row['path']:
                raise ValueError('Output filenames collide; normalize source filenames first')
            destinations[identity] = row['path']
    # Validate every source fingerprint before any output mutation.
    print('Checking source fingerprints…', flush=True)
    for row in rows:
        if hashlib.sha256(source.read(row['path'])).hexdigest() != row['sha256']:
            raise ValueError('Source image changed; rescan before applying')
    out.mkdir(parents=True, exist_ok=True)
    image_root = out/'images'
    audit = []
    for i, row in enumerate(rows, 1):
        decision = dict(row)
        if row['path'] in overrides:
            decision.update({k: v for k, v in overrides[row['path']].items()
                             if k in {'decision', 'vehicle_usability', 'pedestrian_usability'}})
            decision['decision_origin'] = 'manual_review'
        raw = source.read(row['path'])
        dest = image_root / str(output_relative(row['path'], bool(source.zip)))
        original_dest = dest
        # Materialize the working copy, then remove confirmed rejected files.
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        record = {k: decision[k] for k in ('path', 'decision', 'decision_origin',
                                          'vehicle_usability', 'pedestrian_usability')}
        record['flags'] = row['flags']
        record['reason'] = decision.get('reason', decision['decision_origin'])
        if 'darkness_metrics' in decision:
            record['darkness_metrics'] = decision['darkness_metrics']
        if decision['decision'] == 'drop':
            dest.unlink()
            record['result'] = 'deleted_from_working_dataset'
        elif decision['decision'] == 'adjust':
            try:
                with Image.open(io.BytesIO(raw)) as original:
                    adjusted = clahe_luminance(original.convert('RGB'))
                # Keep the exact accepted pixel result without another lossy JPEG encoding.
                adjusted_dest = dest.with_name(dest.name + '.clahe.png')
                adjusted.save(adjusted_dest)
                dest.unlink()
                dest = adjusted_dest
                record.update(result='adjusted', output_path=str(dest.relative_to(out)))
            except (OSError, ValueError, SyntaxError) as exc:
                # Retain bytes and unresolved status if an adjustment cannot be made.
                original_dest.write_bytes(raw)
                record.update(result='adjustment_failed_retained', error=str(exc), decision='review')
        else:
            record.update(result='retained_pending_review' if decision['decision'] == 'review'
                          else 'retained', output_path=str(dest.relative_to(out)))
        audit.append(record)
        if i % 2000 == 0:
            save_json(out/'audit.json', {'source': manifest['source'], 'complete': False, 'images': audit})
            print(f'Applied {i:,}/{len(rows):,}', flush=True)
    save_json(out/'audit.json', {'source': manifest['source'], 'complete': True, 'images': audit})
    summary = dict(Counter(r['result'] for r in audit))
    save_json(out/'summary.json', summary)
    source.close()
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('scan', help='Measure images and make an offline review page')
    s.add_argument('--source', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--review-limit', type=int, default=100)
    s.add_argument('--review-offset', type=int, default=0)
    s.add_argument('--limit', type=int, default=0, help='Smoke test only; partial scans cannot be applied')
    s.set_defaults(func=scan)
    r = sub.add_parser('review', help='Make another review batch without rescanning')
    r.add_argument('--manifest', required=True)
    r.add_argument('--output', required=True)
    r.add_argument('--review-limit', type=int, default=100)
    r.add_argument('--review-offset', type=int, default=0)
    r.set_defaults(func=review)
    a = sub.add_parser('apply', help='Create a cleaned dataset, deleting confirmed rejects from that copy')
    a.add_argument('--manifest', required=True)
    a.add_argument('--decisions', nargs='*', default=[])
    a.add_argument('--output', required=True)
    a.set_defaults(func=apply)
    args = parser.parse_args()
    for key in ('review_limit', 'review_offset', 'limit'):
        if hasattr(args, key) and getattr(args, key) < 0:
            parser.error(f'{key} must be nonnegative')
    try:
        args.func(args)
    except (ValueError, OSError, zipfile.BadZipFile, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
