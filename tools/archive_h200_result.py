#!/usr/bin/env python3
"""Read a completed run and export small, path-anonymized audit evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    training = run / 'training'
    summary = json.loads((run / 'training_summary.json').read_text())
    complete = json.loads((training / 'training_complete.json').read_text())
    state = json.loads((training / 'checkpoint-2115/trainer_state.json').read_text())
    assert summary['exit_code'] == 0 and summary['succeeded']
    assert complete['global_step'] == state['global_step'] == 2115
    assert not summary['monitor_errors']
    logs = [row for row in state['log_history'] if 'loss' in row]
    assert len(logs) == 2115
    assert [row['step'] for row in logs] == list(range(1, 2116))
    assert all(math.isfinite(value) for row in logs for value in row.values()
               if isinstance(value, (int, float)))
    index = json.loads((training / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    assert len(shards) == 4
    files = {}
    for name in shards:
        path = training / name
        assert path.stat().st_size > 0
        print(f'Hashing final model: {name}', flush=True)
        files[name] = {'sha256': digest(path), 'size_bytes': path.stat().st_size}
    checkpoint = training / 'checkpoint-2115'
    for rank in range(4):
        for relative in [f'rng_state_{rank}.pth',
                         f'global_step2115/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt',
                         f'global_step2115/zero_pp_rank_{rank}_mp_rank_00_model_states.pt']:
            assert (checkpoint / relative).stat().st_size > 0, relative
    assert (checkpoint / 'scheduler.pt').stat().st_size > 0
    out.mkdir(parents=True, exist_ok=True)

    def sanitize(value):
        if isinstance(value, str):
            return value.replace('/volume/yzhao04', '<CLUSTER_2_ROOT>')
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    def write(name, value):
        (out / name).write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2) + '\n')

    write('resource_summary.json', summary)
    write('final_model_manifest.json', {'run': run.name, 'files': files})
    write('validation.json', {
        'run': run.name, 'exit_code': 0, 'global_step': 2115,
        'loss_records': len(logs), 'steps_contiguous': True,
        'logged_numeric_metrics_finite': True,
        'four_rank_checkpoint_files_present': True,
        'checkpoint_restore_tested': False,
        'final_checkpoint_log': logs[-1],
        'final_model_index_tensor_count': len(index['weight_map']),
    })
    selected = ['launch_config.txt', 'source_provenance.txt', 'training_summary.json',
                'runtime_gate.json', 'data_gate.json', 'smoke_gate.json', 'training.log',
                'training/training_complete.json', 'training/modality_sampler_plan.jsonl',
                'training/checkpoint-2115/trainer_state.json']
    write('evidence_manifest.json', {name: {'sha256': digest(run / name),
                                          'size_bytes': (run / name).stat().st_size}
                                     for name in selected})
    for name in ['launch_config.txt', 'source_provenance.txt']:
        (out / name).write_text(sanitize((run / name).read_text()))
    print('Archive validation PASS; checkpoint presence is not a restore test.', flush=True)


if __name__ == '__main__':
    main()
