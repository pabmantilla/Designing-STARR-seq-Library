#!/usr/bin/env python3
"""
Plot Ensemble vs Control Performance

This script loads the existing performance CSV from the ensemble training and creates
comprehensive plots comparing all 10 fine-tuned models against the control model,
including the ensemble average.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_train_summary(csv_path):
    """Load performance data from existing CSV file."""
    print("Loading performance data from CSV...")
    
    if not os.path.exists(csv_path):
        print(f"✗ Performance CSV not found at: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded performance data: {len(df)} entries")
    print(f"Model types: {df['model_type'].unique()}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    
    return df

def process_performance_data(df):
    """Process the CSV data into structured format for plotting."""
    print("Processing performance data...")
    
    # Separate data by model type
    control_data = df[df['model_type'] == 'control']
    finetuned_data = df[df['model_type'] == 'finetuned']
    student_data = df[df['model_type'] == 'uncertainty_student']
    
    # Extract control metrics
    if len(control_data) > 0:
        control_pearson = control_data['pearson_mean'].iloc[0]
        control_spearman = control_data['spearman_mean'].iloc[0]
    else:
        print("✗ No control model data found")
        return None
    
    # Extract individual ensemble model metrics
    ensemble_results = []
    ensemble_seeds = []
    
    for _, row in finetuned_data.iterrows():
        ensemble_results.append({
            'seed': row['seed'],
            'pearson_mean': row['pearson_mean'],
            'spearman_mean': row['spearman_mean'],
            'pearson_std': row['pearson_std'],
            'spearman_std': row['spearman_std']
        })
        ensemble_seeds.append(row['seed'])
    
    # Calculate ensemble average
    ensemble_pearson_means = [r['pearson_mean'] for r in ensemble_results]
    ensemble_spearman_means = [r['spearman_mean'] for r in ensemble_results]
    
    ensemble_avg_pearson = np.mean(ensemble_pearson_means)
    ensemble_avg_spearman = np.mean(ensemble_spearman_means)
    
    # Extract student model metrics
    student_performance = None
    if len(student_data) > 0:
        student_performance = {
            'pearson_mean': student_data['pearson_mean'].iloc[0],
            'pearson_std': student_data['pearson_std'].iloc[0],
            'spearman_mean': student_data['spearman_mean'].iloc[0] if not pd.isna(student_data['spearman_mean'].iloc[0]) else None,
            'spearman_std': student_data['spearman_std'].iloc[0] if not pd.isna(student_data['spearman_std'].iloc[0]) else None
        }
        print(f"✓ Found student model performance:")
        print(f"  Pearson: {student_performance['pearson_mean']:.4f}")
        if student_performance['spearman_mean'] is not None:
            print(f"  Spearman: {student_performance['spearman_mean']:.4f}")
        else:
            print(f"  Spearman: Not available")
    else:
        print("ℹ No student model data found in CSV")
    
    return {
        'control': {
            'pearson_mean': control_pearson,
            'spearman_mean': control_spearman
        },
        'ensemble_individual': ensemble_results,
        'ensemble_average': {
            'pearson_mean': ensemble_avg_pearson,
            'spearman_mean': ensemble_avg_spearman
        },
        'ensemble_seeds': sorted(ensemble_seeds),
        'student': student_performance
    }

def plot_performance(results, plots_dir):
    """Create comprehensive performance comparison plots."""
    print("Creating performance plots...")
    
    # Extract data for plotting
    control_pearson = results['control']['pearson_mean']
    control_spearman = results['control']['spearman_mean']
    
    ensemble_seeds = results['ensemble_seeds']
    ensemble_pearson_means = [r['pearson_mean'] for r in results['ensemble_individual']]
    ensemble_spearman_means = [r['spearman_mean'] for r in results['ensemble_individual']]
    ensemble_pearson_stds = [r['pearson_std'] for r in results['ensemble_individual']]
    ensemble_spearman_stds = [r['spearman_std'] for r in results['ensemble_individual']]
    
    ensemble_avg_pearson = results['ensemble_average']['pearson_mean']
    ensemble_avg_spearman = results['ensemble_average']['spearman_mean']
    
    # Create main comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ensemble vs Control Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Individual Model Performance (Pearson)
    ax1.scatter(ensemble_seeds, ensemble_pearson_means, alpha=0.7, s=100, 
                label='Individual Models', color='lightblue', edgecolor='navy')
    ax1.errorbar(ensemble_seeds, ensemble_pearson_means, yerr=ensemble_pearson_stds, 
                 fmt='none', alpha=0.5, color='navy')
    ax1.axhline(y=control_pearson, color='red', linestyle='--', linewidth=2, 
                label=f'Control (r={control_pearson:.3f})')
    ax1.axhline(y=ensemble_avg_pearson, color='green', linestyle='-', linewidth=2, 
                label=f'Ensemble Average (r={ensemble_avg_pearson:.3f})')
    ax1.set_xlabel('Model Seed')
    ax1.set_ylabel('Pearson Correlation')
    ax1.set_title('Individual Model Performance (Pearson)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ensemble_seeds)
    
    # Plot 2: Individual Model Performance (Spearman)
    ax2.scatter(ensemble_seeds, ensemble_spearman_means, alpha=0.7, s=100, 
                label='Individual Models', color='lightgreen', edgecolor='darkgreen')
    ax2.errorbar(ensemble_seeds, ensemble_spearman_means, yerr=ensemble_spearman_stds, 
                 fmt='none', alpha=0.5, color='darkgreen')
    ax2.axhline(y=control_spearman, color='red', linestyle='--', linewidth=2, 
                label=f'Control (ρ={control_spearman:.3f})')
    ax2.axhline(y=ensemble_avg_spearman, color='green', linestyle='-', linewidth=2, 
                label=f'Ensemble Average (ρ={ensemble_avg_spearman:.3f})')
    ax2.set_xlabel('Model Seed')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Individual Model Performance (Spearman)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ensemble_seeds)
    
    # Plot 3: Performance Comparison (Bar Chart)
    models = ['Control'] + [f'Model {s}' for s in ensemble_seeds] + ['Ensemble Avg']
    pearson_values = [control_pearson] + ensemble_pearson_means + [ensemble_avg_pearson]
    spearman_values = [control_spearman] + ensemble_spearman_means + [ensemble_avg_spearman]
    
    x_pos = range(len(models))
    bars1 = ax3.bar(x_pos, pearson_values, alpha=0.7, 
                    color=['red'] + ['blue'] * len(ensemble_seeds) + ['green'])
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Pearson Correlation')
    ax3.set_title('Performance Comparison (Pearson)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, pearson_values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Performance Comparison (Spearman)
    bars2 = ax4.bar(x_pos, spearman_values, alpha=0.7, 
                    color=['red'] + ['blue'] * len(ensemble_seeds) + ['green'])
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Spearman Correlation')
    ax4.set_title('Performance Comparison (Spearman)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, spearman_values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'ensemble_vs_control_comparison.svg')
    plt.savefig(plot_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Main comparison plot saved to: {plot_path}")
    
    # Create improvement analysis
    create_improvement_analysis(results, plots_dir)

def create_improvement_analysis(results, plots_dir):
    """Create improvement analysis over control."""
    print("Creating improvement analysis...")
    
    control_pearson = results['control']['pearson_mean']
    control_spearman = results['control']['spearman_mean']
    
    # Calculate improvements
    improvements_pearson = []
    improvements_spearman = []
    ensemble_seeds = results['ensemble_seeds']
    
    for r in results['ensemble_individual']:
        improvements_pearson.append(r['pearson_mean'] - control_pearson)
        improvements_spearman.append(r['spearman_mean'] - control_spearman)
    
    # Ensemble average improvement
    ensemble_avg_pearson_improvement = results['ensemble_average']['pearson_mean'] - control_pearson
    ensemble_avg_spearman_improvement = results['ensemble_average']['spearman_mean'] - control_spearman
    
    # Create improvement plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pearson improvement
    x_pos = range(len(ensemble_seeds))
    bars1 = ax1.bar(x_pos, improvements_pearson, alpha=0.7, color='lightblue', edgecolor='navy')
    ax1.axhline(y=ensemble_avg_pearson_improvement, color='green', linestyle='-', linewidth=2, 
                label=f'Ensemble Avg Improvement: {ensemble_avg_pearson_improvement:.3f}')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Control Baseline')
    ax1.set_xlabel('Model Seed')
    ax1.set_ylabel('Improvement in Pearson Correlation')
    ax1.set_title('Individual Model Improvement Over Control (Pearson)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ensemble_seeds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, improvement) in enumerate(zip(bars1, improvements_pearson)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{improvement:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Spearman improvement
    bars2 = ax2.bar(x_pos, improvements_spearman, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax2.axhline(y=ensemble_avg_spearman_improvement, color='green', linestyle='-', linewidth=2, 
                label=f'Ensemble Avg Improvement: {ensemble_avg_spearman_improvement:.3f}')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Control Baseline')
    ax2.set_xlabel('Model Seed')
    ax2.set_ylabel('Improvement in Spearman Correlation')
    ax2.set_title('Individual Model Improvement Over Control (Spearman)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ensemble_seeds)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, improvement) in enumerate(zip(bars2, improvements_spearman)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{improvement:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    improvement_plot_path = os.path.join(plots_dir, 'ensemble_improvement_analysis.svg')
    plt.savefig(improvement_plot_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Improvement analysis plot saved to: {improvement_plot_path}")

def create_dev_hk_performance_plot(results, plots_dir):
    """Create Dev and Hk performance comparison plot."""
    print("Creating Dev and Hk performance plot...")
    
    # Extract data for plotting
    control_pearson = results['control']['pearson_mean']
    ensemble_avg_pearson = results['ensemble_average']['pearson_mean']
    
    # Get student model performance if available
    student_performance = results.get('student', None)
    student_pearson = student_performance['pearson_mean'] if student_performance else None
    
    # Sort ensemble data by seed to ensure proper ordering (convert to int for numerical sorting)
    ensemble_data_sorted = sorted(results['ensemble_individual'], key=lambda x: int(x['seed']))
    ensemble_seeds = [r['seed'] for r in ensemble_data_sorted]
    ensemble_pearson_means = [r['pearson_mean'] for r in ensemble_data_sorted]
    ensemble_spearman_means = [r['spearman_mean'] for r in ensemble_data_sorted]
    
    # Get control and ensemble average Spearman values
    control_spearman = results['control']['spearman_mean']
    ensemble_avg_spearman = results['ensemble_average']['spearman_mean']
    
    # Get student model Spearman performance if available
    student_spearman = student_performance['spearman_mean'] if student_performance and student_performance.get('spearman_mean') is not None else None
    
    # Create figure with subplots for Pearson and Spearman correlation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Performance Comparison: Pearson vs Spearman Correlation', fontsize=16, fontweight='bold')
    
    # Use Pearson correlation for left plot and Spearman for right plot
    # since we don't have task-specific Dev/Hk data in the CSV
    
    # Dev task (left subplot)
    x_pos = range(len(ensemble_seeds))
    bars1 = ax1.bar(x_pos, ensemble_pearson_means, alpha=0.7, color='lightblue', edgecolor='navy', 
                    label='Individual Ensemble Models')
    
    # Add ensemble average line
    ensemble_line1 = ax1.axhline(y=ensemble_avg_pearson, color='green', linestyle='-', linewidth=3, 
                label=f'Ensemble Average (r={ensemble_avg_pearson:.3f})')
    
    # Add control line
    control_line1 = ax1.axhline(y=control_pearson, color='red', linestyle='--', linewidth=2, 
                label=f'Control (r={control_pearson:.3f})')
    
    # Add student model if available
    student_line1 = None
    if student_pearson is not None:
        student_line1 = ax1.axhline(y=student_pearson, color='black', linestyle='-', linewidth=4, 
                    label=f'Student Model (r={student_pearson:.3f})')
    
    ax1.set_xlabel('Model Seed')
    ax1.set_ylabel('Pearson Correlation')
    ax1.set_title('Pearson Correlation Performance')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ensemble_seeds)
    ax1.grid(True, alpha=0.3)
    
    # Set y-axis range to zoom into the top part for better visibility
    all_values = ensemble_pearson_means + [control_pearson, ensemble_avg_pearson]
    if student_pearson is not None:
        all_values.append(student_pearson)
    y_min = min(all_values) - 0.01
    y_max = max(all_values) + 0.01
    ax1.set_ylim(y_min, y_max)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, ensemble_pearson_means)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Spearman correlation (right subplot)
    bars2 = ax2.bar(x_pos, ensemble_spearman_means, alpha=0.7, color='lightgreen', edgecolor='darkgreen', 
                    label='Individual Ensemble Models')
    
    # Add ensemble average line
    ensemble_line2 = ax2.axhline(y=ensemble_avg_spearman, color='green', linestyle='-', linewidth=3, 
                label=f'Ensemble Average (ρ={ensemble_avg_spearman:.3f})')
    
    # Add control line
    control_line2 = ax2.axhline(y=control_spearman, color='red', linestyle='--', linewidth=2, 
                label=f'Control (ρ={control_spearman:.3f})')
    
    # Add student model if available
    student_line2 = None
    if student_spearman is not None:
        student_line2 = ax2.axhline(y=student_spearman, color='black', linestyle='-', linewidth=4, 
                    label=f'Student Model (ρ={student_spearman:.3f})')
    
    ax2.set_xlabel('Model Seed')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Spearman Correlation Performance')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ensemble_seeds)
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis range to zoom into the top part for better visibility
    all_values_spearman = ensemble_spearman_means + [control_spearman, ensemble_avg_spearman]
    if student_spearman is not None:
        all_values_spearman.append(student_spearman)
    y_min_spearman = min(all_values_spearman) - 0.01
    y_max_spearman = max(all_values_spearman) + 0.01
    ax2.set_ylim(y_min_spearman, y_max_spearman)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, ensemble_spearman_means)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Create a proper legend using the actual plot elements
    legend_elements = [bars1, control_line1, ensemble_line1]
    legend_labels = ['Individual Ensemble Models', 'Control Model', 'Ensemble Average']
    
    if student_line1 is not None:
        legend_elements.append(student_line1)
        legend_labels.append('Student Model')
    
    fig.legend(legend_elements, legend_labels, 
               loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    dev_hk_plot_path = os.path.join(plots_dir, 'dev_hk_performance_comparison.svg')
    plt.savefig(dev_hk_plot_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Dev and Hk performance plot saved to: {dev_hk_plot_path}")

def main():
    """Main function to create ensemble vs control performance plots."""
    print("Ensemble vs Control Performance Analysis")
    print("=" * 50)
    
    # Configuration
    output_dir = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/EvoAug_ensemble_10'
    plots_dir = '/grid/wsbs/home_norepl/pmantill/train_EvoAug/Distillation_plots'
    csv_path = os.path.join(output_dir, 'ensemble_performance.csv')
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load performance data from CSV
    df = load_train_summary(csv_path)
    if df is None:
        return
    
    # Process data for plotting
    results = process_performance_data(df)
    if results is None:
        return
    
    # Create plots
    #plot_performance(results, plots_dir)
    
    # Create Dev and Hk performance plot
    create_dev_hk_performance_plot(results, plots_dir)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Control Pearson: {results['control']['pearson_mean']:.4f}")
    print(f"Control Spearman: {results['control']['spearman_mean']:.4f}")
    print(f"Ensemble Avg Pearson: {results['ensemble_average']['pearson_mean']:.4f}")
    print(f"Ensemble Avg Spearman: {results['ensemble_average']['spearman_mean']:.4f}")
    
    if results.get('student') is not None:
        print(f"Student Model Pearson: {results['student']['pearson_mean']:.4f}")
        if results['student']['spearman_mean'] is not None:
            print(f"Student Model Spearman: {results['student']['spearman_mean']:.4f}")
        else:
            print("Student Model Spearman: Not available")
    else:
        print("Student Model: Not available")
    
    # Individual model performance
    print(f"\nIndividual Model Performance (Pearson):")
    for r in results['ensemble_individual']:
        print(f"  Model {r['seed']}: {r['pearson_mean']:.4f}")
    
    print(f"\nIndividual Model Performance (Spearman):")
    for r in results['ensemble_individual']:
        print(f"  Model {r['seed']}: {r['spearman_mean']:.4f}")
    
    print(f"\nImprovement over Control (Pearson):")
    control_pearson = results['control']['pearson_mean']
    for r in results['ensemble_individual']:
        improvement = r['pearson_mean'] - control_pearson
        print(f"  Model {r['seed']}: +{improvement:.4f}")
    
    ensemble_improvement = results['ensemble_average']['pearson_mean'] - control_pearson
    print(f"  Ensemble Average: +{ensemble_improvement:.4f}")
    
    if results.get('student') is not None:
        student_improvement = results['student']['pearson_mean'] - control_pearson
        print(f"  Student Model: +{student_improvement:.4f}")
    
    print(f"\nImprovement over Control (Spearman):")
    control_spearman = results['control']['spearman_mean']
    for r in results['ensemble_individual']:
        improvement = r['spearman_mean'] - control_spearman
        print(f"  Model {r['seed']}: +{improvement:.4f}")
    
    ensemble_improvement_spearman = results['ensemble_average']['spearman_mean'] - control_spearman
    print(f"  Ensemble Average: +{ensemble_improvement_spearman:.4f}")
    
    if results.get('student') is not None and results['student']['spearman_mean'] is not None:
        student_improvement_spearman = results['student']['spearman_mean'] - control_spearman
        print(f"  Student Model: +{student_improvement_spearman:.4f}")
    
    print(f"\nAll plots saved to: {plots_dir}")
    print("="*60)

if __name__ == "__main__":
    main()