import itertools
import subprocess
import os
import yaml
import json
from pathlib import Path
import sys


def load_param_grid(grid_config_path):
    with open(grid_config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_experiments(param_grid):
    total_params = param_grid['total_loss_params']
    freq_params = param_grid['freq_loss_params']

    keys_total = list(total_params.keys())
    values_total = list(total_params.values())
    keys_freq = list(freq_params.keys())
    values_freq = list(freq_params.values())

    all_combinations = []
    for combo_total in itertools.product(*values_total):
        params_dict = dict(zip(keys_total, combo_total))
        for combo_freq in itertools.product(*values_freq):
            freq_dict = {f'freq_{k}': v for k, v in zip(keys_freq, combo_freq)}
            all_combinations.append({**params_dict, **freq_dict})
    return all_combinations


def run_single_experiment(exp_id, params, base_exp_dir, config_path='../configs/training.yml'):
    exp_name = f"exp_{exp_id:03d}_char{params['char_weight']}_edge{params['edge_weight']}_freq{params['freq_weight']}"
    exp_save_dir = Path(base_exp_dir) / exp_name
    exp_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Starting Experiment {exp_id}: {exp_name}")
    print(f"{'=' * 60}")

    param_file = exp_save_dir / 'hyperparameters.json'
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=4)

    cmd = [
        sys.executable, 'train.py',
        '--char_weight', str(params['char_weight']),
        '--edge_weight', str(params['edge_weight']),
        '--freq_weight', str(params['freq_weight']),
        '--freq_amp_weight', str(params['freq_amp_weight']),
        '--freq_phase_weight', str(params['freq_phase_weight']),
        '--freq_consistency_weight', str(params['freq_consistency_weight']),
        '--save_dir', str(exp_save_dir)
    ]

    log_file_path = exp_save_dir / 'training_log.txt'
    try:
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
            process.wait()
            return_code = process.returncode
    except Exception as e:
        print(f"Failed to run experiment {exp_id}: {e}")
        return {'exp_id': exp_id, 'exp_name': exp_name, 'params': params, 'best_psnr': None, 'status': 'ERROR',
                'error': str(e)}

    best_psnr = None
    if log_file_path.exists() and return_code == 0:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'Best Val:' in line:
                    try:
                        import re
                        match = re.search(r'Best Val:\s*([\d.]+)', line)
                        if match:
                            best_psnr = float(match.group(1))
                            break
                    except ValueError:
                        continue

    result = {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'params': params,
        'best_psnr': best_psnr,
        'status': 'SUCCESS' if return_code == 0 else 'FAILED',
        'log_file': str(log_file_path),
        'model_dir': str(exp_save_dir)
    }
    print(f"xperiment {exp_id} finished. Best PSNR: {best_psnr}")
    return result


def main():
    param_grid_path = '../configs/param_grid.yaml'
    base_experiment_dir = '../experiments'
    Path(base_experiment_dir).mkdir(parents=True, exist_ok=True)

    param_grid = load_param_grid(param_grid_path)
    all_param_sets = generate_experiments(param_grid)

    print(f"Loaded parameter grid.")
    print(f"Total number of experiments to run: {len(all_param_sets)}")

    all_results = []
    for idx, param_set in enumerate(all_param_sets):
        result = run_single_experiment(idx + 1, param_set, base_experiment_dir)
        all_results.append(result)

        progress_path = Path(base_experiment_dir) / 'experiment_progress.json'
        with open(progress_path, 'w') as pf:
            json.dump(all_results, pf, indent=4)

    final_results_path = Path(base_experiment_dir) / 'all_experiment_results.json'
    with open(final_results_path, 'w') as rf:
        json.dump(all_results, rf, indent=4)

    successful_results = [r for r in all_results if r['status'] == 'SUCCESS' and r['best_psnr'] is not None]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['best_psnr'])
        print(f"\n HYPERPARAMETER TUNING COMPLETED ")
        print(f"Best Experiment: {best_result['exp_name']}")
        print(f"Best PSNR: {best_result['best_psnr']:.4f}")
        print(f"Parameters: {best_result['params']}")
        print(f"Model saved at: {best_result['model_dir']}")
    else:
        print("\nNo successful experiments with valid PSNR found.")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()