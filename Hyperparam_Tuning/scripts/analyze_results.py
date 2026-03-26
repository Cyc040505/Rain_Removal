import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np  # Moved to top-level import


def load_and_analyze_results(results_file='../experiments/all_experiment_results.json'):
    with open(results_file, 'r') as f:
        results = json.load(f)

    df_data = []
    for r in results:
        if r['status'] == 'SUCCESS' and r['best_psnr'] is not None:
            row = r['params'].copy()
            row['exp_id'] = r['exp_id']
            row['exp_name'] = r['exp_name']
            row['best_psnr'] = r['best_psnr']
            df_data.append(row)

    if not df_data:
        print("No valid results to analyze.")
        return

    df = pd.DataFrame(df_data)

    # 1. Top experiments by PSNR
    print("=== Top 10 Experiments by PSNR ===")
    top10 = df.nlargest(10, 'best_psnr')[['exp_id', 'exp_name', 'best_psnr',
                                          'char_weight', 'edge_weight', 'freq_weight',
                                          'freq_amp_weight', 'freq_phase_weight', 'freq_consistency_weight']]
    print(top10.to_string(index=False))
    print("\n")

    # 2. Parameter-PSNR Correlation Analysis
    # Focus on the three specified weights for scatter plots and overall correlation
    focus_params = ['freq_weight', 'freq_phase_weight', 'freq_consistency_weight']
    all_param_cols = ['char_weight', 'edge_weight', 'freq_weight',
                      'freq_amp_weight', 'freq_phase_weight', 'freq_consistency_weight']

    print("=== Correlation of Parameters with PSNR ===")
    correlations = df[all_param_cols + ['best_psnr']].corr()['best_psnr'].drop('best_psnr')
    print(correlations.sort_values(ascending=False))
    print("\n")

    # 3. Generate Scatter Plots (PNG) - Only for the three specified parameters
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    for idx, param in enumerate(focus_params):
        ax = axes[idx]
        ax.scatter(df[param], df['best_psnr'], alpha=0.6, s=50)
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel('Best PSNR (dB)', fontsize=12)
        ax.set_title(f'{param} vs. PSNR', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add a linear trendline
        z = np.polyfit(df[param], df['best_psnr'], 1)
        p = np.poly1d(z)
        ax.plot(df[param].sort_values(), p(df[param].sort_values()), "r--", linewidth=2, alpha=0.8, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        ax.legend(fontsize=10)

    plt.tight_layout()
    scatter_plot_path = '../experiments/focused_param_vs_psnr_scatter.png'
    plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Scatter plots saved to '{scatter_plot_path}'")
    plt.close(fig)  # Close the figure to free memory

    # 4. Generate Correlation Heatmap (Supplementary Visualization - PNG)
    # This heatmap shows correlations between all parameters and PSNR.
    heatmap_params = ['freq_weight', 'freq_phase_weight', 'freq_consistency_weight']

    corr_matrix = df[heatmap_params + ['best_psnr']].corr()
    plt.figure(figsize=(8, 6))

    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.3f', linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=1, bottom=0.1, top=1)
    heatmap_path = '../experiments/param_psnr_correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Correlation heatmap saved to '{heatmap_path}'")
    plt.close()

    # 5. Generate Parallel Coordinates Plot
    try:
        import plotly.express as px
        import plotly.io as pio

        # For the parallel plot, we can use all parameters or a subset. Using all for context.
        fig = px.parallel_coordinates(df,
                                      dimensions=all_param_cols + ['best_psnr'],
                                      color='best_psnr',
                                      color_continuous_scale='Plasma',
                                      range_color=[df['best_psnr'].min(), df['best_psnr'].max()])

        parallel_plot_path_png = '../experiments/parallel_coordinates.png'

        # Save the figure as a PNG file
        fig.write_image(parallel_plot_path_png, width=1200, height=600, scale=2)
        print(f"[INFO] Parallel coordinates plot (PNG) saved to '{parallel_plot_path_png}'")
    except ImportError:
        print("[WARNING] Install `plotly` to generate the parallel coordinates plot: pip install plotly")

    print("\n=== Evaluation Complete ===")
    print(f"Generated charts: 1) Focused Scatter Plots, 2) Correlation Heatmap, 3) Parallel Coordinates Plot.")
    return df


if __name__ == '__main__':
    results_path = Path('../experiments/all_experiment_results.json')
    if results_path.exists():
        df_results = load_and_analyze_results(str(results_path))
    else:
        print(f"[ERROR] Results file not found at {results_path}. Please run experiments first.")