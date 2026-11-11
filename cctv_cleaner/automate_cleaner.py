#!/usr/bin/env python3
"""Run an automatic, configurable quality model. No manual review is required."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFilter

import clean_cctv as cleaner


def extra_metrics(image, rules=None):
    # Nearest-neighbor preserves pixel values for corruption/noise measurements.
    rgb = np.asarray(image.convert('RGB').resize((160, 120), Image.Resampling.NEAREST))
    y = np.asarray(Image.fromarray(rgb).convert('L'), dtype=np.float32)
    neutral = rgb.max(2).astype(int)-rgb.min(2).astype(int) < 6
    hist = np.bincount(y[neutral].astype('uint8').ravel(), minlength=256)
    peak = int(np.argmax(hist[32:224])+32)
    fill = float((neutral & (np.abs(y-peak) < 3)).mean())
    mixed = y[:-1, :-1]-y[1:, :-1]-y[:-1, 1:]+y[1:, 1:]
    sigma = float(np.median(np.abs(mixed))*.7413)
    smooth = sum(y[j:j+118, i:i+158] for j in range(3) for i in range(3))/9
    gradient = float(np.abs(np.diff(y, axis=0)).mean()+np.abs(np.diff(y, axis=1)).mean())
    smooth_gradient = float(np.abs(np.diff(smooth, axis=0)).mean()+np.abs(np.diff(smooth, axis=1)).mean())
    structure = smooth_gradient/max(gradient, .001)
    rough = float((np.abs(mixed) > 20).mean())
    # A broadcast test pattern has several saturated vertical color strips.
    band = rgb[25:78].astype(float)
    colored = (band.max(2)-band.min(2)>120) & (band.max(2)>170)
    vertical_stability = float((band.std(axis=0).mean(axis=1) < 8).mean())
    colors = ((band.mean(axis=0)>128)*[4, 2, 1]).sum(axis=1).astype(int)
    distinct = sum(n >= 5 for n in np.bincount(colors, minlength=8))
    test_pattern = float(colored.mean()) > .55 and vertical_stability > .70 and distinct >= 5
    result = {'gray_peak_fraction': round(fill, 5), 'gray_peak': peak,
            'noise_sigma_proxy': round(sigma, 4), 'structure_ratio': round(structure, 4),
            'rough_fraction': round(rough, 4), 'test_pattern': bool(test_pattern)}
    rules = rules or {'noise_sigma_min': 15, 'noise_max_structure_ratio': .30,
                      'noise_min_rough_fraction': .65}
    if (sigma >= rules['noise_sigma_min'] and structure < rules['noise_max_structure_ratio']
        and rough > rules['noise_min_rough_fraction']):
        native = np.asarray(image.convert('L'), dtype=np.float32)
        mixed_native = native[:-1, :-1]-native[1:, :-1]-native[:-1, 1:]+native[1:, 1:]
        result['noise_native_rough_fraction'] = round(float((np.abs(mixed_native) > 20).mean()), 5)
        result['noise_native_sigma'] = round(float(np.median(np.abs(mixed_native))*.7413), 4)
    return result


def decide(metrics, extra, rules):
    if extra['test_pattern']:
        return 'drop', 'broadcast_color_test_pattern'
    if (96 <= extra['gray_peak'] <= 160 and
        extra['gray_peak_fraction'] > rules['gray_fill_min_fraction'] and
        metrics['contrast_range'] < rules['gray_fill_max_contrast_range']):
        return 'drop', 'large_gray_corruption_regions'
    # Noise is recorded diagnostically; it is never a deletion reason.
    if metrics['dark_fraction'] > rules['black_min_fraction'] and metrics['contrast_std'] < 5:
        return 'drop', 'almost_black_no_usable_detail'
    if metrics['bright_fraction'] > rules['white_min_fraction'] and metrics['contrast_std'] < 5:
        return 'drop', 'almost_white_no_usable_detail'
    if (metrics['laplacian_variance'] < rules['featureless_max_sharpness'] and
        metrics['contrast_range'] < rules['featureless_max_contrast_range']):
        return 'drop', 'extremely_low_detail_and_contrast'
    if (extra['noise_sigma_proxy'] < rules['adjust_max_noise_sigma'] and
        (metrics['contrast_range'] < rules['adjust_max_contrast_range'] or
         metrics['dark_fraction'] > rules['adjust_min_dark_fraction'])):
        return 'adjust', 'selective_local_contrast'
    return 'keep', 'usable_under_initial_quality_rules'


def darkness_candidate(metrics, rules):
    return (metrics['mean_brightness'] < rules.get('darkness_candidate_mean', 90) or
            metrics['dark_fraction'] > rules.get('darkness_candidate_fraction', .05))


def regional_darkness(image, rules):
    """Spatial visibility proxy, not semantic road/sidewalk segmentation."""
    width, height = image.size
    # Exclude a narrow top strip and bottom timestamp area, keeping side areas.
    crop = (0, int(height*.12), width, max(int(height*.95), int(height*.12)+1))
    original = image.convert('L').crop(crop).resize((240, 180), Image.Resampling.BILINEAR)
    adjusted = cleaner.clahe_luminance(image).convert('L').crop(crop).resize((240, 180), Image.Resampling.BILINEAR)
    def tiles(array):
        return array.reshape(6, 30, 8, 30).transpose(0, 2, 1, 3).reshape(48, -1)
    before = tiles(np.asarray(original, dtype=float))
    after = tiles(np.asarray(adjusted, dtype=float))
    original_dark = ((before < rules.get('dark_tile_pixel_level', 50)).mean(1) >
                     rules.get('dark_tile_min_dark_pixels', .70))
    def visible_structure(gray):
        smooth = np.asarray(gray.filter(ImageFilter.GaussianBlur(1.2)), dtype=float)
        dx = np.diff(smooth, axis=1, append=smooth[:, -1:])
        dy = np.diff(smooth, axis=0, append=smooth[-1:, :])
        edges = np.hypot(dx, dy) > rules.get('dark_tile_gradient_level', 4)
        return tiles(edges).mean(1)
    edges_after = visible_structure(adjusted)
    edges_before = visible_structure(original)
    contrast_after = np.percentile(after, 90, axis=1)-np.percentile(after, 10, axis=1)
    contrast_before = np.percentile(before, 90, axis=1)-np.percentile(before, 10, axis=1)
    def weak(contrast, edges, medians):
        return ((contrast < rules.get('dark_tile_min_contrast', 35)) |
                (edges < rules.get('dark_tile_min_edge_fraction', .08)) |
                (medians < rules.get('dark_tile_min_adjusted_median', 25)))
    unreadable_before = original_dark & weak(contrast_before, edges_before, np.median(before, axis=1))
    unreadable_after = original_dark & weak(contrast_after, edges_after, np.median(after, axis=1))
    # Bright signs/headlights only help their own tiles, not the whole frame.
    return {'method': '48_tile_central_lower_scene_proxy',
            'original_dark_area_fraction': round(float(original_dark.mean()), 5),
            'unreadable_area_before': round(float(unreadable_before.mean()), 5),
            'unreadable_area_after_clahe': round(float(unreadable_after.mean()), 5),
            'unreadable_tile_indices_after': np.flatnonzero(unreadable_after).tolist(),
            'roi_fraction': [0, .12, 1, .95]}


def darkness_decision(measurements, rules):
    if measurements['unreadable_area_after_clahe'] > rules.get('darkness_drop_area_fraction', .55):
        return 'drop', 'dark_scene_unreadable_after_adjustment'
    if (measurements['unreadable_area_before']-measurements['unreadable_area_after_clahe'] >=
        rules.get('darkness_adjust_min_area_gain', .10)):
        return 'adjust', 'dark_scene_visibility_improved'
    return None


def plan(args):
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get('partial_scan') or manifest.get('schema_version') != 1:
        raise ValueError('Use a complete version-1 scan')
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh empty output directory')
    rules = json.loads(Path(args.rules).read_text())
    if rules.get('version') not in (1, 2) or rules.get('kind') != 'rule_based_quality_model':
        raise ValueError('Unsupported quality rules')
    paths = {r['path'] for r in manifest['images']}
    overrides = {}
    for file in args.decisions:
        d = json.loads(Path(file).read_text())
        if d['source'] != manifest['source']:
            raise ValueError('Decision file has a different source')
        for row in d['decisions']:
            if row['path'] not in paths or row['decision'] not in {'keep', 'adjust', 'drop', 'review'}:
                raise ValueError('Invalid decision')
            if row['decision'] != 'review': overrides[row['path']] = row['decision']
    source = cleaner.Source(manifest['source'])
    if set(source.names) != paths:
        raise ValueError('Source image list changed; rescan')
    if not source.zip and out.is_relative_to(source.path):
        raise ValueError('Output must be separate from source')
    out.mkdir(parents=True, exist_ok=True)
    counts, reasons = Counter(), Counter()
    by_camera = defaultdict(Counter)
    by_hour = defaultdict(Counter)
    for i, row in enumerate(manifest['images'], 1):
        raw = source.read(row['path'])
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Source image changed; rescan before automation')
        try:
            if 'decode_error' in row['flags']:
                action, reason = 'drop', 'unreadable_image'
            elif 'uniform_frame' in row['flags']:
                action, reason = 'drop', 'uniform_frame_no_scene'
            else:
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    extra = extra_metrics(image, rules)
                    action, reason = decide(row['metrics'], extra, rules)
                    if action != 'drop' and darkness_candidate(row['metrics'], rules):
                        darkness = regional_darkness(image, rules)
                        row['darkness_metrics'] = darkness
                        update = darkness_decision(darkness, rules)
                        if update is not None:
                            action, reason = update
                row['extra_quality_metrics'] = extra
        except (OSError, ValueError, SyntaxError):
            action, reason = 'drop', 'unreadable_image'
        # User decisions have priority; these remain explicitly human-supplied.
        if row['path'] in overrides:
            action, reason = overrides[row['path']], 'downloaded_user_decision'
        elif row['decision_origin'] == 'user_reference':
            action, reason = row['decision'], 'earlier_user_decision'
        row['decision'], row['reason'] = action, reason
        row['decision_origin'] = 'user_reference' if reason.endswith('user_decision') else f'automated_rules_v{rules["version"]}'
        counts[action] += 1
        reasons[reason] += 1
        by_camera[row['camera']][action] += 1
        key = cleaner.image_key(row['path'])
        hour = key.rsplit('_', 1)[-1][:2] if key else 'unknown'
        by_hour[hour][action] += 1
        if i % 5000 == 0: print(f'Quality model evaluated {i:,}/{len(paths):,}', flush=True)
    manifest['automation'] = {'kind': 'rule_based_quality_model', 'rules': rules,
                              'automatic_actions_enabled': True,
                              'note': 'No required manual review. Previous weak learned classifier is not used.'}
    cleaner.save_json(out/'automated_manifest.json', manifest)
    summary = {'images': len(paths), 'decisions': dict(counts), 'reasons': dict(reasons),
               'by_camera': {k: dict(v) for k, v in by_camera.items()},
               'by_filename_hour': {k: dict(v) for k, v in sorted(by_hour.items())},
               'model_kind': 'rule_based_quality_model', 'validation_accuracy': None,
               'note': 'Complete first-pass decisions, not validated object-counting accuracy. ZIP preserved. No duplicate removal or removal solely for night/rain.'}
    cleaner.save_json(out/'automation_summary.json', summary)
    source.close()
    print(json.dumps({'decisions': dict(counts), 'reasons': dict(reasons)}, indent=2))
    if args.apply:
        cleaner.apply(SimpleNamespace(manifest=str(out/'automated_manifest.json'),
                                      decisions=[], output=str(out/'cleaned_dataset')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--decisions', nargs='*', default=[])
    p.add_argument('--rules', default=str(Path(__file__).with_name('quality_rules.json')))
    p.add_argument('--output', required=True)
    p.add_argument('--apply', action='store_true', help='Also create the cleaned working dataset')
    args = p.parse_args()
    try: plan(args)
    except (ValueError, OSError, KeyError) as exc: p.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
