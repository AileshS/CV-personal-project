"""Review package trust boundaries and durable human-only links."""
import copy
import json

import pytest
from PIL import Image

from playlab.dataset import canonical_hash, file_hash
from playlab.media import source_fingerprint
from playlab.review import (
    confirm_possession,
    frame_review,
    frame_selection,
    load_batch,
    register_batch,
    request_missing_media,
    save_navigation,
    save_review,
)
from playlab.store import Store


@pytest.fixture
def bundle(tmp_path):
    store = Store(tmp_path / 'data')
    video = tmp_path / 'original.mp4'
    video.write_bytes(b'original source bytes')
    game = store.save('game', {'id': 'g', 'source_path': str(video),
        'source_fingerprint': source_fingerprint(video), 'duration': 100, 'width': 20, 'height': 10,
        'split_mode': 'game', 'split': 'train', 'split_blocks': []})
    parent = store.save('possession', {'id': 'parent', 'game_id': 'g', 'start': 0, 'end': 100,
                                     'source': 'automatic', 'reviewed': False, 'split': 'train'})
    root = tmp_path / 'batch'
    (root / 'images').mkdir(parents=True)
    Image.new('RGB', (20, 10)).save(root / 'images/f.png')
    Image.new('RGB', (10, 5)).save(root / 'images/f-preview.jpg')
    manifest = {'schema': 1, 'kind': 'unreviewed_label_batch', 'reviewed': False,
        'frame_count': 1, 'review_item_count': 1,
        'sources': {'g': {k: game[k] for k in ('source_fingerprint', 'duration', 'width', 'height', 'split_mode', 'split', 'split_blocks')}},
        'review_items': [{'id': 'i', 'game_id': 'g', 'possession_id': parent['id'],
                          'start': 20, 'end': 40, 'split': 'train', 'reviewed': False}],
        'frames': [{'id': 'f', 'game_id': 'g', 'possession_id': 'parent', 'review_item_id': 'i',
                    'time': 30, 'split': 'train', 'image': 'images/f.png', 'width': 20, 'height': 10,
                    'reviewed': False, 'negative': False, 'proposed_boxes': []}],
        'files': {name: file_hash(root / name) for name in ('images/f.png', 'images/f-preview.jpg')}}
    manifest['sources']['g']['source_size_bytes'] = video.stat().st_size
    write_manifest(root, manifest)
    return store, root, manifest


def write_manifest(root, manifest):
    manifest.pop('batch_hash', None)
    manifest['batch_hash'] = canonical_hash(manifest)
    (root / 'manifest.json').write_text(json.dumps(manifest))


def confirm(store, batch, **kwargs):
    return confirm_possession(store, batch, 'i', start=20, end=40,
                              expected_revision=kwargs.pop('expected_revision', 0), **kwargs)


def test_preview_browsing_creates_no_game_or_labels(bundle):
    store, root, _ = bundle
    (root / 'images/f.png').unlink()
    empty = Store(root.parent / 'empty')
    record = register_batch(empty, root)
    batch = load_batch(root)
    assert batch.media('f')['native_path'] is None
    assert batch.media('f')['preview_path'].exists()
    assert not empty.list('game') and not empty.list('frame')
    with pytest.raises(ValueError):
        frame_selection(store, batch, 'f')
    with pytest.raises(ValueError, match='Native'):
        confirm(store, batch)
    action = request_missing_media(empty, batch, 'f')
    assert action['notification']['reason'] == 'Delivery disabled'
    saved = save_navigation(empty, batch, 'f', expected_revision=record['revision'])
    assert Store(empty.root).get('review_batch', batch.id) == saved


def test_manual_link_durable_cas_and_stale_target(bundle):
    store, root, _ = bundle
    batch = load_batch(root)
    with pytest.raises(ValueError):
        confirm(store, batch, expected_revision=99)
    assert len(store.list('possession')) == 1  # No orphan corrected record on CAS failure.
    review = confirm(store, batch, notes='offensive rebound retained', offensive_rebound=True)
    selected = frame_selection(store, batch, 'f')
    assert selected['time'] == 30
    assert store.get('possession', 'parent')['source'] == 'automatic'
    assert len(store.list('possession')) == 2
    assert not any(store.list(k) for k in ('frame', 'box', 'shot', 'event'))
    assert frame_review(Store(store.root), batch, 'f') == review
    with pytest.raises(ValueError, match='Already linked'):
        confirm(store, batch, expected_revision=review['revision'])
    target = store.get('possession', selected['possession_id'])
    store.save('possession', {**target, 'reviewed': False})
    with pytest.raises(ValueError, match='current'):
        frame_selection(store, batch, 'f')
    with pytest.raises(ValueError, match='changed'):
        confirm(store, batch, expected_revision=review['revision'], target_id=target['id'], target_revision=1)


def test_notes_cas_and_unreviewed_suggestions_remain_separate(bundle):
    store, root, manifest = bundle
    manifest['frames'][0]['proposed_boxes'] = [{'label': 'ball', 'box': [.1, .1, .2, .2],
                                             'confidence': .9, 'reviewed': False}]
    write_manifest(root, manifest)
    batch = load_batch(root)
    note = save_review(store, batch, 'i', notes='uncertain', uncertain=True,
                       offensive_rebound=False, expected_revision=0)
    with pytest.raises(ValueError):
        save_review(store, batch, 'i', notes='stale', uncertain=False,
                    offensive_rebound=False, expected_revision=0)
    assert note['uncertain'] and not store.list('box')


@pytest.mark.parametrize('mutation', [
    lambda m: m.update(frame_count=2),
    lambda m: m['frames'].append(copy.deepcopy(m['frames'][0])),
    lambda m: m['frames'][0].update(game_id='wrong'),
    lambda m: m['frames'][0].update(possession_id='wrong'),
    lambda m: m['frames'][0].update(time=90),
    lambda m: m['frames'][0].update(split='test'),
    lambda m: m['review_items'][0].update(split='test'),
    lambda m: m['files'].update({'../outside.png': 'a' * 64}),
    lambda m: m['files'].update({'/outside.png': 'a' * 64}),
    lambda m: m['files'].update({'images\\outside.png': 'a' * 64}),
    lambda m: m['files'].update({'images/f.png': 'bad'}),
    lambda m: m['frames'][0].update(proposed_boxes=[{'box': [0, 0, 2, 1], 'label': 'ball',
                                                   'confidence': .5, 'reviewed': False}]),
])
def test_invalid_manifest(bundle, mutation):
    _, root, manifest = bundle
    mutation(manifest)
    write_manifest(root, manifest)
    with pytest.raises(ValueError):
        load_batch(root)


def test_hash_nonfinite_and_symlink_boundaries(bundle, tmp_path):
    _, root, manifest = bundle
    original = (root / 'manifest.json').read_text()
    (root / 'manifest.json').write_text(original.replace('"frame_count": 1', '"frame_count": 2'))
    with pytest.raises(ValueError, match='hash'):
        load_batch(root)
    (root / 'manifest.json').write_text(original.replace('"time": 30', '"time": NaN'))
    with pytest.raises(ValueError):
        load_batch(root)
    (root / 'manifest.json').write_text(original)
    native = root / 'images/f.png'
    native.unlink()
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'not read')
    native.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        load_batch(root)
    native.unlink()
    native.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='hash'):
        load_batch(root).media('f')


def test_policy_guard_source_and_target_validation(bundle):
    store, root, manifest = bundle
    blocks = [{'start': 0, 'end': 50, 'split': 'train'}, {'start': 50, 'end': 100, 'split': 'test'}]
    manifest['sources']['g'].update(split_mode='contiguous', split='unassigned', split_blocks=blocks)
    write_manifest(root, manifest)
    batch = load_batch(root)  # Locked metadata is allowed; locked rows are not.
    with pytest.raises(ValueError, match='policy'):
        confirm(store, batch)
    game = store.get('game', 'g')
    store.save('game', {**game, 'split_mode': 'contiguous', 'split': 'unassigned', 'split_blocks': blocks})
    with pytest.raises(ValueError, match='guards'):
        confirm_possession(store, batch, 'i', start=20, end=45, expected_revision=0)
    with pytest.raises(ValueError, match='selected frame'):
        confirm_possession(store, batch, 'i', start=10, end=20, expected_revision=0)
    target = store.save('possession', {'game_id': 'g', 'start': 20, 'end': 40, 'split': 'train',
                                      'source': 'manual', 'reviewed': True})
    with pytest.raises(ValueError, match='overlapping'):
        confirm(store, batch)
    with pytest.raises(ValueError):
        confirm(store, batch, target_id=target['id'], target_revision=99)
    confirm(store, batch, target_id=target['id'], target_revision=target['revision'])
    (root.parent / 'original.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError, match='original'):
        frame_selection(store, batch, 'f')


def test_atomic_collection_guards_and_deletes(tmp_path):
    store = Store(tmp_path)
    box = store.save('box', {'possession_id': 'p', 'time': 1})
    snapshot = {box['id']: box['revision']}
    other = store.save('box', {'possession_id': 'p', 'time': 1})
    with pytest.raises(ValueError, match='changed'):
        store.save_many([], deletes=[('box', box['id'], 1)],
                        collections=[('box', {'possession_id': 'p'}, snapshot)])
    assert store.get('box', box['id'])
    store.save_many([], deletes=[('box', box['id'], 1), ('box', other['id'], 1)])
    assert not store.list('box')


def test_window_can_link_frames_to_distinct_whole_possessions(bundle):
    store, root, manifest = bundle
    frame = copy.deepcopy(manifest['frames'][0])
    frame.update(id='f2', time=35, image='images/f2.png')
    Image.new('RGB', (20, 10)).save(root / frame['image'])
    manifest['files'][frame['image']] = file_hash(root / frame['image'])
    manifest['frames'].append(frame)
    manifest['frame_count'] = 2
    write_manifest(root, manifest)
    batch = load_batch(root)
    first = confirm_possession(store, batch, 'i', frame_id='f', start=25, end=32, expected_revision=0)
    second = confirm_possession(store, batch, 'i', frame_id='f2', start=33, end=38, expected_revision=0)
    assert first['possession_id'] != second['possession_id']
    assert frame_selection(store, batch, 'f')['possession_id'] == first['possession_id']
    assert frame_selection(store, batch, 'f2')['possession_id'] == second['possession_id']
    target = store.get('possession', first['possession_id'])
    changed = store.save('possession', {**target, 'end': 31})
    with pytest.raises(ValueError):
        frame_selection(store, batch, 'f')
    relinked = confirm_possession(store, batch, 'i', frame_id='f', start=25, end=31,
                                 expected_revision=first['revision'], target_id=target['id'],
                                 target_revision=changed['revision'])
    assert relinked['possession_revision'] == changed['revision']
    assert frame_selection(store, batch, 'f')['possession_revision'] == changed['revision']


def test_uncertainty_and_native_dimension_substitution(bundle):
    store, root, manifest = bundle
    batch = load_batch(root)
    with pytest.raises(ValueError, match='uncertainty'):
        confirm(store, batch, uncertain=True)
    assert len(store.list('possession')) == 1
    Image.new('RGB', (10, 5)).save(root / 'images/f.png')
    manifest['files']['images/f.png'] = file_hash(root / 'images/f.png')
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match='dimensions'):
        load_batch(root).media('f')
    manifest['sources']['g']['width'] = 1920
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match='native source'):
        load_batch(root)


def test_frame_draft_cas_and_uncertainty_requires_explicit_reconfirmation(bundle):
    from playlab.review import save_frame_review

    store, root, _ = bundle
    batch = load_batch(root)
    review = confirm(store, batch, notes='confirmed bounds')
    target = store.get('possession', review['possession_id'])
    draft = save_frame_review(store, batch, 'f', notes='uncertain rebound boundary',
                             uncertain=True, offensive_rebound=True,
                             expected_revision=review['revision'])
    assert draft['notes'] == 'uncertain rebound boundary'
    assert draft['possession_id'] == target['id']
    with pytest.raises(ValueError):
        frame_selection(store, batch, 'f')
    with pytest.raises(ValueError, match='changed'):
        save_frame_review(store, batch, 'f', notes='stale', uncertain=False,
                          offensive_rebound=False, expected_revision=review['revision'])
    cleared = save_frame_review(store, batch, 'f', notes='ready to reconfirm', uncertain=False,
                               offensive_rebound=True, expected_revision=draft['revision'])
    with pytest.raises(ValueError):
        frame_selection(store, batch, 'f')
    reconfirmed = confirm(store, batch, expected_revision=cleared['revision'],
                         target_id=target['id'], target_revision=target['revision'],
                         notes=cleared['notes'], offensive_rebound=True)
    assert reconfirmed['notes'] == 'ready to reconfirm'
    assert not reconfirmed['requires_boundary_confirmation']
    assert frame_selection(store, batch, 'f')['possession_id'] == target['id']
    assert not store.list('frame') and not store.list('box')


def test_source_native_dimensions_mandatory_and_match_known_game(bundle):
    store, root, manifest = bundle
    del manifest['sources']['g']['width']
    write_manifest(root, manifest)
    with pytest.raises(ValueError):
        load_batch(root)
    manifest['sources']['g']['width'] = 20
    write_manifest(root, manifest)
    game = store.get('game', 'g')
    store.save('game', {**game, 'width': 80, 'height': 60})
    with pytest.raises(ValueError, match='native dimensions'):
        confirm(store, load_batch(root))


def test_uncertain_boundary_draft_survives_restart_without_gold(bundle):
    from playlab.review import save_frame_review

    store, root, _ = bundle
    batch = load_batch(root)
    draft = save_frame_review(store, batch, 'f', notes='need original boundaries', uncertain=True,
                             offensive_rebound=True, expected_revision=0,
                             start=15, end=45, split='val', target_id='candidate', target_revision=3)
    reopened = frame_review(Store(store.root), batch, 'f')
    assert reopened == draft
    assert (reopened['draft_start'], reopened['draft_end'], reopened['draft_split']) == (15, 45, 'val')
    assert reopened['draft_target_id'] == 'candidate'
    assert 'possession_id' not in reopened
    assert len(store.list('possession')) == 1
    with pytest.raises(ValueError):
        frame_selection(store, batch, 'f')
    with pytest.raises(ValueError, match='Draft boundaries'):
        save_frame_review(store, batch, 'f', notes='', uncertain=True, offensive_rebound=False,
                          expected_revision=draft['revision'], start=-1, end=40)
    assert frame_review(store, batch, 'f') == draft


def test_changed_boundary_draft_invalidates_link_but_notes_only_does_not(bundle):
    from playlab.review import save_frame_review

    store, root, _ = bundle
    batch = load_batch(root)
    confirmed = confirm(store, batch)
    target = store.get('possession', confirmed['possession_id'])
    note = save_frame_review(store, batch, 'f', notes='just a note', uncertain=False,
                            offensive_rebound=False, expected_revision=confirmed['revision'])
    assert frame_selection(store, batch, 'f')['possession_id'] == target['id']
    draft = save_frame_review(store, batch, 'f', notes='wider possession', uncertain=False,
                             offensive_rebound=False, expected_revision=note['revision'],
                             start=15, end=40, split='train', target_id=target['id'], target_revision=1)
    assert draft['requires_boundary_confirmation']
    with pytest.raises(ValueError):
        frame_selection(store, batch, 'f')
    # Explicitly reverting to the reviewed target is a new human confirmation.
    confirm(store, batch, expected_revision=draft['revision'], target_id=target['id'], target_revision=1)
    assert frame_selection(store, batch, 'f')['possession_id'] == target['id']
