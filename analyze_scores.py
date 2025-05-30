import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ScoreAnalyzer:
    """
    📊 Comprehensive score analysis tool for PlasmodiumClassification inference results.
    
    Features:
    - Load and parse score files
    - Calculate class-wise statistics
    - Generate confidence distributions
    - Analyze prediction patterns
    - Create visualization plots
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.scores_data = None
        self.class_names = []
        self.num_classes = 0
        
    def load_scores_file(self, scores_file_path):
        """Load and parse the scores file."""
        if self.verbose:
            print(f"📂 Loading scores from: {scores_file_path}")
        
        if not os.path.exists(scores_file_path):
            raise FileNotFoundError(f"❌ Scores file not found: {scores_file_path}")
        
        try:
            with open(scores_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                raise ValueError("❌ Scores file appears to be empty or corrupted")
            
            # Parse header to get class names
            header = lines[0].strip().split(',')
            score_headers = [h for h in header if h.startswith('score_')]
            self.class_names = [h.replace('score_', '') for h in score_headers]
            self.num_classes = len(self.class_names)
            
            if self.verbose:
                print(f"   📋 Found {self.num_classes} classes: {self.class_names}")
            
            # Parse data
            data = []
            for line_idx, line in enumerate(lines[1:], 1):
                parts = line.strip().split(',')
                
                if len(parts) < 3 + self.num_classes:
                    if self.verbose:
                        print(f"   ⚠️ Skipping malformed line {line_idx}")
                    continue
                
                try:
                    sample_path = parts[0]
                    true_label = int(parts[1])
                    predicted_label = int(parts[2])
                    scores = [float(parts[3 + i]) for i in range(self.num_classes)]
                    
                    # Create record
                    record = {
                        'sample_path': sample_path,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'scores': scores,
                        'max_score': max(scores),
                        'max_score_class': scores.index(max(scores)),
                        'true_class_score': scores[true_label] if 0 <= true_label < len(scores) else 0.0,
                        'predicted_class_score': scores[predicted_label] if 0 <= predicted_label < len(scores) else 0.0,
                        'correct_prediction': true_label == predicted_label
                    }
                    
                    # Add individual class scores
                    for i, class_name in enumerate(self.class_names):
                        record[f'score_{class_name}'] = scores[i]
                    
                    data.append(record)
                    
                except (ValueError, IndexError) as e:
                    if self.verbose:
                        print(f"   ⚠️ Error parsing line {line_idx}: {e}")
                    continue
            
            self.scores_data = pd.DataFrame(data)
            
            if self.verbose:
                print(f"   ✅ Loaded {len(self.scores_data)} samples successfully")
                print(f"   📊 Overall accuracy: {self.scores_data['correct_prediction'].mean():.4f}")
            
            return self.scores_data
            
        except Exception as e:
            raise RuntimeError(f"❌ Error loading scores file: {e}")
    
    def calculate_class_statistics(self):
        """Calculate comprehensive statistics for each class."""
        if self.scores_data is None:
            raise ValueError("❌ No scores data loaded. Call load_scores_file() first.")
        
        if self.verbose:
            print(f"\n📊 Calculating class-wise statistics...")
        
        stats = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            score_col = f'score_{class_name}'
            
            # All samples' scores for this class
            all_scores = self.scores_data[score_col]
            
            # Samples where this is the true class
            true_class_mask = self.scores_data['true_label'] == class_idx
            true_class_scores = self.scores_data[true_class_mask][score_col]
            
            # Samples where this was predicted
            pred_class_mask = self.scores_data['predicted_label'] == class_idx
            pred_class_scores = self.scores_data[pred_class_mask][score_col]
            
            # Correctly predicted samples of this class
            correct_mask = true_class_mask & self.scores_data['correct_prediction']
            correct_scores = self.scores_data[correct_mask][score_col]
            
            # Incorrectly predicted samples of this class (false positives)
            incorrect_pred_mask = pred_class_mask & (~self.scores_data['correct_prediction'])
            incorrect_pred_scores = self.scores_data[incorrect_pred_mask][score_col]
            
            # Missed samples of this class (false negatives)
            missed_mask = true_class_mask & (~self.scores_data['correct_prediction'])
            missed_scores = self.scores_data[missed_mask][score_col]
            
            # Calculate statistics
            class_stats = {
                'class_index': class_idx,
                'class_name': class_name,
                'total_samples': len(self.scores_data),
                'true_class_count': len(true_class_scores),
                'predicted_count': len(pred_class_scores),
                'correct_predictions': len(correct_scores),
                'false_positives': len(incorrect_pred_scores),
                'false_negatives': len(missed_scores),
                
                # Overall score statistics
                'mean_score_all': all_scores.mean(),
                'std_score_all': all_scores.std(),
                'min_score_all': all_scores.min(),
                'max_score_all': all_scores.max(),
                'median_score_all': all_scores.median(),
                
                # True class score statistics
                'mean_score_true_class': true_class_scores.mean() if len(true_class_scores) > 0 else 0.0,
                'std_score_true_class': true_class_scores.std() if len(true_class_scores) > 0 else 0.0,
                'min_score_true_class': true_class_scores.min() if len(true_class_scores) > 0 else 0.0,
                'max_score_true_class': true_class_scores.max() if len(true_class_scores) > 0 else 0.0,
                'median_score_true_class': true_class_scores.median() if len(true_class_scores) > 0 else 0.0,
                
                # Correct prediction score statistics
                'mean_score_correct': correct_scores.mean() if len(correct_scores) > 0 else 0.0,
                'std_score_correct': correct_scores.std() if len(correct_scores) > 0 else 0.0,
                'min_score_correct': correct_scores.min() if len(correct_scores) > 0 else 0.0,
                'max_score_correct': correct_scores.max() if len(correct_scores) > 0 else 0.0,
                
                # Incorrect prediction score statistics
                'mean_score_false_positive': incorrect_pred_scores.mean() if len(incorrect_pred_scores) > 0 else 0.0,
                'mean_score_false_negative': missed_scores.mean() if len(missed_scores) > 0 else 0.0,
                
                # Performance metrics
                'precision': len(correct_scores) / len(pred_class_scores) if len(pred_class_scores) > 0 else 0.0,
                'recall': len(correct_scores) / len(true_class_scores) if len(true_class_scores) > 0 else 0.0,
            }
            
            # Calculate F1 score
            if class_stats['precision'] + class_stats['recall'] > 0:
                class_stats['f1_score'] = 2 * (class_stats['precision'] * class_stats['recall']) / (class_stats['precision'] + class_stats['recall'])
            else:
                class_stats['f1_score'] = 0.0
            
            stats[class_name] = class_stats
        
        return stats
    
    def print_statistics_summary(self, stats):
        """Print a comprehensive statistics summary."""
        if self.verbose:
            print(f"\n📈 CLASS-WISE STATISTICS SUMMARY")
            print("=" * 80)
            
            # Header
            print(f"{'Class Name':<25} {'Samples':<8} {'Mean Score':<12} {'True Class':<12} {'Correct':<12} {'Precision':<10} {'Recall':<8} {'F1':<8}")
            print("-" * 80)
            
            # Class-wise rows
            for class_name, stat in stats.items():
                print(f"{class_name:<25} "
                      f"{stat['true_class_count']:<8} "
                      f"{stat['mean_score_all']:<12.4f} "
                      f"{stat['mean_score_true_class']:<12.4f} "
                      f"{stat['mean_score_correct']:<12.4f} "
                      f"{stat['precision']:<10.3f} "
                      f"{stat['recall']:<8.3f} "
                      f"{stat['f1_score']:<8.3f}")
            
            print("-" * 80)
            
            # Overall statistics
            total_samples = self.scores_data['correct_prediction'].sum()
            total_accuracy = self.scores_data['correct_prediction'].mean()
            avg_max_score = self.scores_data['max_score'].mean()
            avg_true_class_score = self.scores_data['true_class_score'].mean()
            
            print(f"\nOVERALL STATISTICS:")
            print(f"  Total samples: {len(self.scores_data)}")
            print(f"  Correct predictions: {total_samples}")
            print(f"  Overall accuracy: {total_accuracy:.4f}")
            print(f"  Average max score: {avg_max_score:.4f}")
            print(f"  Average true class score: {avg_true_class_score:.4f}")
            
            # Confidence analysis
            high_conf_threshold = 0.8
            low_conf_threshold = 0.5
            
            high_conf_mask = self.scores_data['max_score'] >= high_conf_threshold
            low_conf_mask = self.scores_data['max_score'] <= low_conf_threshold
            
            print(f"\nCONFIDENCE ANALYSIS:")
            print(f"  High confidence (≥{high_conf_threshold}): {high_conf_mask.sum()} samples ({high_conf_mask.mean()*100:.1f}%)")
            print(f"  Low confidence (≤{low_conf_threshold}): {low_conf_mask.sum()} samples ({low_conf_mask.mean()*100:.1f}%)")
            
            if high_conf_mask.sum() > 0:
                high_conf_acc = self.scores_data[high_conf_mask]['correct_prediction'].mean()
                print(f"  High confidence accuracy: {high_conf_acc:.4f}")
            
            if low_conf_mask.sum() > 0:
                low_conf_acc = self.scores_data[low_conf_mask]['correct_prediction'].mean()
                print(f"  Low confidence accuracy: {low_conf_acc:.4f}")
    
    def save_statistics_csv(self, stats, output_path):
        """Save statistics to CSV file."""
        try:
            # Convert stats to DataFrame
            stats_df = pd.DataFrame(stats).T
            stats_df.to_csv(output_path)
            
            if self.verbose:
                print(f"💾 Statistics saved to: {output_path}")
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving statistics: {e}")
    
    def create_visualizations(self, stats, output_dir):
        """Create comprehensive visualizations."""
        if self.verbose:
            print(f"\n🎨 Creating visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Class-wise mean scores heatmap
        self._plot_score_heatmap(stats, output_dir / "class_scores_heatmap.png")
        
        # 2. Score distributions by class
        self._plot_score_distributions(output_dir / "score_distributions.png")
        
        # 3. Confidence vs Accuracy analysis
        self._plot_confidence_analysis(output_dir / "confidence_analysis.png")
        
        # 4. Class performance metrics
        self._plot_performance_metrics(stats, output_dir / "performance_metrics.png")
        
        # 5. Score correlation matrix
        self._plot_score_correlations(output_dir / "score_correlations.png")
        
        if self.verbose:
            print(f"   ✅ Visualizations saved to: {output_dir}")
    
    def _plot_score_heatmap(self, stats, output_path):
        """Plot class-wise score statistics as heatmap."""
        # Prepare data for heatmap
        metrics = ['mean_score_all', 'mean_score_true_class', 'mean_score_correct']
        metric_labels = ['Mean Score (All)', 'Mean Score (True Class)', 'Mean Score (Correct)']
        
        data = []
        class_labels = []
        
        for class_name, stat in stats.items():
            class_labels.append(class_name)
            data.append([stat[metric] for metric in metrics])
        
        data = np.array(data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(self.class_names) * 0.5)))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels(class_labels)
        
        # Add text annotations
        for i in range(len(class_labels)):
            for j in range(len(metric_labels)):
                text = ax.text(j, i, f'{data[i, j]:.3f}', 
                             ha="center", va="center", color="black", fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score Value', rotation=270, labelpad=20)
        
        plt.title('Class-wise Score Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distributions(self, output_path):
        """Plot score distributions for each class."""
        n_classes = len(self.class_names)
        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, class_name in enumerate(self.class_names):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            score_col = f'score_{class_name}'
            all_scores = self.scores_data[score_col]
            
            # Plot histogram
            ax.hist(all_scores, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            ax.axvline(all_scores.mean(), color='red', linestyle='--', label=f'Mean: {all_scores.mean():.3f}')
            ax.axvline(all_scores.median(), color='green', linestyle='--', label=f'Median: {all_scores.median():.3f}')
            
            ax.set_title(f'{class_name}', fontweight='bold')
            ax.set_xlabel('Score')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_classes, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Score Distributions by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_analysis(self, output_path):
        """Plot confidence vs accuracy analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Max score distribution
        ax1.hist(self.scores_data['max_score'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(self.scores_data['max_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.scores_data["max_score"].mean():.3f}')
        ax1.set_title('Distribution of Maximum Scores')
        ax1.set_xlabel('Max Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence vs Accuracy
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        digitized = np.digitize(self.scores_data['max_score'], bins)
        accuracies = []
        counts = []
        
        for i in range(1, len(bins)):
            mask = digitized == i
            if mask.sum() > 0:
                acc = self.scores_data[mask]['correct_prediction'].mean()
                count = mask.sum()
            else:
                acc = 0
                count = 0
            accuracies.append(acc)
            counts.append(count)
        
        ax2.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=6)
        ax2.set_title('Accuracy vs Confidence')
        ax2.set_xlabel('Max Score (Confidence)')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. True class score vs prediction correctness
        correct_scores = self.scores_data[self.scores_data['correct_prediction']]['true_class_score']
        incorrect_scores = self.scores_data[~self.scores_data['correct_prediction']]['true_class_score']
        
        ax3.hist([correct_scores, incorrect_scores], bins=30, alpha=0.7, 
                label=['Correct', 'Incorrect'], color=['green', 'red'])
        ax3.set_title('True Class Scores: Correct vs Incorrect Predictions')
        ax3.set_xlabel('True Class Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Sample count per confidence bin
        ax4.bar(bin_centers, counts, width=0.08, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_title('Sample Count by Confidence Level')
        ax4.set_xlabel('Max Score (Confidence)')
        ax4.set_ylabel('Sample Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_metrics(self, stats, output_path):
        """Plot performance metrics comparison."""
        classes = list(stats.keys())
        precision = [stats[c]['precision'] for c in classes]
        recall = [stats[c]['recall'] for c in classes]
        f1_scores = [stats[c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_correlations(self, output_path):
        """Plot correlation matrix of class scores."""
        score_cols = [f'score_{name}' for name in self.class_names]
        score_data = self.scores_data[score_cols]
        
        correlation_matrix = score_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Score Correlation Matrix', fontsize=14, fontweight='bold')
        ax.set_xticklabels([name.replace('score_', '') for name in score_cols], rotation=45, ha='right')
        ax.set_yticklabels([name.replace('score_', '') for name in score_cols], rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_label_6_predictions(self):
        """Analyze predictions with label 6 and calculate confidence gap ratio."""
        if self.scores_data is None:
            raise ValueError("❌ No scores data loaded. Call load_scores_file() first.")
        
        if self.verbose:
            print(f"\n🔍 Analyzing predictions with label 6...")
        
        # Filter predictions where predicted_label is 6
        label_6_mask = self.scores_data['true_label'] == 6
        label_6_data = self.scores_data[label_6_mask].copy()
        
        if len(label_6_data) == 0:
            if self.verbose:
                print("❌ No predictions with label 6 found!")
            return None
        
        # Calculate max1 and max2 for each prediction
        scores_columns = [f'score_{name}' for name in self.class_names]
        label_6_scores = label_6_data[scores_columns].values
        
        # Get top 2 scores for each sample
        sorted_scores = np.sort(label_6_scores, axis=1)[:, ::-1]  # Sort descending
        max1 = sorted_scores[:, 0]  # Highest score
        max2 = sorted_scores[:, 1]  # Second highest score
        
        # Calculate ratio (max1 - max2) / max(max1, max2)
        max_of_max1_max2 = np.maximum(max1, max2)
        confidence_gap_ratio = (max1 - max2) / max_of_max1_max2
        
        # Calculate raw difference max1 - max2
        score_difference = max1 - max2
        
        # Calculate max2/max1 ratio (closer to 1 means more uncertain, closer to 0 means more confident)
        max2_to_max1_ratio = max2 / max1
        
        # Add to dataframe
        label_6_data['max1_score'] = max1
        label_6_data['max2_score'] = max2
        label_6_data['confidence_gap_ratio'] = confidence_gap_ratio
        label_6_data['score_difference'] = score_difference
        label_6_data['max2_to_max1_ratio'] = max2_to_max1_ratio
        
        # Calculate statistics
        stats = {
            'total_label_6_predictions': len(label_6_data),
            'correct_label_6_predictions': label_6_data['correct_prediction'].sum(),
            'accuracy_label_6': label_6_data['correct_prediction'].mean(),
            'confidence_gap_ratio_stats': {
                'mean': confidence_gap_ratio.mean(),
                'std': confidence_gap_ratio.std(),
                'min': confidence_gap_ratio.min(),
                'max': confidence_gap_ratio.max(),
                'median': np.median(confidence_gap_ratio),
                'q25': np.percentile(confidence_gap_ratio, 25),
                'q75': np.percentile(confidence_gap_ratio, 75)
            },
            'score_difference_stats': {
                'mean': score_difference.mean(),
                'std': score_difference.std(),
                'min': score_difference.min(),
                'max': score_difference.max(),
                'median': np.median(score_difference),
                'q25': np.percentile(score_difference, 25),
                'q75': np.percentile(score_difference, 75)
            },
            'max2_to_max1_ratio_stats': {
                'mean': max2_to_max1_ratio.mean(),
                'std': max2_to_max1_ratio.std(),
                'min': max2_to_max1_ratio.min(),
                'max': max2_to_max1_ratio.max(),
                'median': np.median(max2_to_max1_ratio),
                'q25': np.percentile(max2_to_max1_ratio, 25),
                'q75': np.percentile(max2_to_max1_ratio, 75)
            },
            'max1_stats': {
                'mean': max1.mean(),
                'std': max1.std(),
                'min': max1.min(),
                'max': max1.max(),
                'median': np.median(max1)
            },
            'max2_stats': {
                'mean': max2.mean(),
                'std': max2.std(),
                'min': max2.min(),
                'max': max2.max(),
                'median': np.median(max2)
            }
        }
        
        if self.verbose:
            self._print_label_6_analysis(stats, label_6_data)
        
        return {
            'statistics': stats,
            'detailed_data': label_6_data,
            'confidence_gap_ratios': confidence_gap_ratio,
            'score_differences': score_difference,
            'max2_to_max1_ratios': max2_to_max1_ratio
        }
    
    def _print_label_6_analysis(self, stats, label_6_data):
        """Print detailed analysis of label 6 predictions."""
        print(f"📊 LABEL 6 PREDICTIONS ANALYSIS")
        print("=" * 60)
        print(f"Total label 6 predictions: {stats['total_label_6_predictions']}")
        print(f"Correct label 6 predictions: {stats['correct_label_6_predictions']}")
        print(f"Label 6 accuracy: {stats['accuracy_label_6']:.4f}")
        
        print(f"\n🎯 CONFIDENCE GAP RATIO: (max1 - max2) / max(max1, max2)")
        gap_stats = stats['confidence_gap_ratio_stats']
        print(f"  Mean: {gap_stats['mean']:.4f}")
        print(f"  Std:  {gap_stats['std']:.4f}")
        print(f"  Min:  {gap_stats['min']:.4f}")
        print(f"  Max:  {gap_stats['max']:.4f}")
        print(f"  Median: {gap_stats['median']:.4f}")
        print(f"  Q25:  {gap_stats['q25']:.4f}")
        print(f"  Q75:  {gap_stats['q75']:.4f}")
        
        print(f"\n📏 SCORE DIFFERENCE: max1 - max2")
        diff_stats = stats['score_difference_stats']
        print(f"  Mean: {diff_stats['mean']:.4f}")
        print(f"  Std:  {diff_stats['std']:.4f}")
        print(f"  Min:  {diff_stats['min']:.4f}")
        print(f"  Max:  {diff_stats['max']:.4f}")
        print(f"  Median: {diff_stats['median']:.4f}")
        print(f"  Q25:  {diff_stats['q25']:.4f}")
        print(f"  Q75:  {diff_stats['q75']:.4f}")
        
        print(f"\n🔀 MAX2/MAX1 RATIO: max2 / max1")
        ratio_stats = stats['max2_to_max1_ratio_stats']
        print(f"  Mean: {ratio_stats['mean']:.4f}")
        print(f"  Std:  {ratio_stats['std']:.4f}")
        print(f"  Min:  {ratio_stats['min']:.4f}")
        print(f"  Max:  {ratio_stats['max']:.4f}")
        print(f"  Median: {ratio_stats['median']:.4f}")
        print(f"  Q25:  {ratio_stats['q25']:.4f}")
        print(f"  Q75:  {ratio_stats['q75']:.4f}")
        
        print(f"\n📈 MAX1 SCORE STATISTICS:")
        max1_stats = stats['max1_stats']
        print(f"  Mean: {max1_stats['mean']:.4f}")
        print(f"  Std:  {max1_stats['std']:.4f}")
        print(f"  Range: [{max1_stats['min']:.4f}, {max1_stats['max']:.4f}]")
        
        print(f"\n📉 MAX2 SCORE STATISTICS:")
        max2_stats = stats['max2_stats']
        print(f"  Mean: {max2_stats['mean']:.4f}")
        print(f"  Std:  {max2_stats['std']:.4f}")
        print(f"  Range: [{max2_stats['min']:.4f}, {max2_stats['max']:.4f}]")
        
        # Confidence gap ratio distribution
        gap_ratios = label_6_data['confidence_gap_ratio']
        print(f"\n📊 CONFIDENCE GAP RATIO DISTRIBUTION:")
        
        # Define ranges for ratio
        ranges = [
            (0.0, 0.1, "Very Low Gap"),
            (0.1, 0.2, "Low Gap"),
            (0.2, 0.3, "Medium Gap"),
            (0.3, 0.5, "High Gap"),
            (0.5, 1.0, "Very High Gap")
        ]
        
        for min_val, max_val, label in ranges:
            mask = (gap_ratios >= min_val) & (gap_ratios < max_val)
            count = mask.sum()
            percentage = count / len(gap_ratios) * 100
            if count > 0:
                avg_acc = label_6_data[mask]['correct_prediction'].mean()
                print(f"  {label:<15} [{min_val:.1f}-{max_val:.1f}): {count:4d} samples ({percentage:5.1f}%) - Accuracy: {avg_acc:.3f}")
            else:
                print(f"  {label:<15} [{min_val:.1f}-{max_val:.1f}): {count:4d} samples ({percentage:5.1f}%)")
        
        # Score difference distribution
        score_diffs = label_6_data['score_difference']
        print(f"\n📊 SCORE DIFFERENCE DISTRIBUTION:")
        
        # Define ranges for difference (0.0 to 1.0 typically)
        diff_ranges = [
            (0.0, 0.05, "Very Small Diff"),
            (0.05, 0.1, "Small Diff"),
            (0.1, 0.2, "Medium Diff"),
            (0.2, 0.3, "Large Diff"),
            (0.3, 1.0, "Very Large Diff")
        ]
        
        for min_val, max_val, label in diff_ranges:
            mask = (score_diffs >= min_val) & (score_diffs < max_val)
            count = mask.sum()
            percentage = count / len(score_diffs) * 100
            if count > 0:
                avg_acc = label_6_data[mask]['correct_prediction'].mean()
                avg_ratio = label_6_data[mask]['confidence_gap_ratio'].mean()
                print(f"  {label:<15} [{min_val:.2f}-{max_val:.2f}): {count:4d} samples ({percentage:5.1f}%) - Accuracy: {avg_acc:.3f}, Avg Ratio: {avg_ratio:.3f}")
            else:
                print(f"  {label:<15} [{min_val:.2f}-{max_val:.2f}): {count:4d} samples ({percentage:5.1f}%)")
        
        # Max2/Max1 ratio distribution
        max2_to_max1_ratios = label_6_data['max2_to_max1_ratio']
        print(f"\n📊 MAX2/MAX1 RATIO DISTRIBUTION:")
        
        # Define ranges for max2/max1 ratio (0.0 to 1.0, closer to 1 = more uncertain)
        ratio_ranges = [
            (0.0, 0.2, "Very Confident"),    # max2 is much smaller than max1
            (0.2, 0.4, "Confident"),        # clear winner
            (0.4, 0.6, "Moderate Conf"),    # some difference
            (0.6, 0.8, "Low Confidence"),   # close scores
            (0.8, 1.0, "Very Uncertain")    # very close scores
        ]
        
        for min_val, max_val, label in ratio_ranges:
            mask = (max2_to_max1_ratios >= min_val) & (max2_to_max1_ratios < max_val)
            count = mask.sum()
            percentage = count / len(max2_to_max1_ratios) * 100
            if count > 0:
                avg_acc = label_6_data[mask]['correct_prediction'].mean()
                avg_gap_ratio = label_6_data[mask]['confidence_gap_ratio'].mean()
                avg_score_diff = label_6_data[mask]['score_difference'].mean()
                print(f"  {label:<15} [{min_val:.1f}-{max_val:.1f}): {count:4d} samples ({percentage:5.1f}%) - Accuracy: {avg_acc:.3f}, Gap: {avg_gap_ratio:.3f}, Diff: {avg_score_diff:.3f}")
            else:
                print(f"  {label:<15} [{min_val:.1f}-{max_val:.1f}): {count:4d} samples ({percentage:5.1f}%)")


def analyze_scores_direct(
    scores_file_path,
    output_dir='score_analysis',
    create_plots=True,
    verbose=True,
    analyze_label_6=True
):
    """
    🎯 Direct analysis function without CLI - perfect for notebooks and direct execution!
    
    Args:
        scores_file_path (str): Path to the scores .txt file
        output_dir (str): Output directory for results
        create_plots (bool): Whether to generate visualization plots
        verbose (bool): Whether to print detailed progress
        analyze_label_6 (bool): Whether to perform label 6 specific analysis
    
    Returns:
        dict: Analysis results with stats and file paths
    """
    try:
        # Initialize analyzer
        analyzer = ScoreAnalyzer(verbose=verbose)
        
        # Load scores file
        if verbose:
            print(f"🔍 Starting direct analysis of: {scores_file_path}")
        
        scores_data = analyzer.load_scores_file(scores_file_path)
        
        # Calculate statistics
        stats = analyzer.calculate_class_statistics()
        
        # Print summary
        analyzer.print_statistics_summary(stats)
        
        # Analyze label 6 predictions if requested
        label_6_analysis = None
        if analyze_label_6:
            try:
                label_6_analysis = analyzer.analyze_label_6_predictions()
            except Exception as e:
                if verbose:
                    print(f"⚠️ Warning: Label 6 analysis failed: {e}")
        
        # Prepare output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save statistics to CSV
        stats_csv_path = output_dir / "class_statistics.csv"
        analyzer.save_statistics_csv(stats, stats_csv_path)
        
        # Save detailed data
        detailed_csv_path = output_dir / "detailed_scores.csv"
        scores_data.to_csv(detailed_csv_path, index=False)
        if verbose:
            print(f"💾 Detailed scores saved to: {detailed_csv_path}")
        
        # Create visualizations
        plots_dir = None
        if create_plots:
            plots_dir = output_dir / "plots"
            analyzer.create_visualizations(stats, plots_dir)
        
        # Return results
        results = {
            'analyzer': analyzer,
            'scores_data': scores_data,
            'statistics': stats,
            'label_6_analysis': label_6_analysis,
            'output_directory': str(output_dir),
            'files_created': {
                'class_statistics_csv': str(stats_csv_path),
                'detailed_scores_csv': str(detailed_csv_path),
                'plots_directory': str(plots_dir) if plots_dir else None
            },
            'summary': {
                'total_samples': len(scores_data),
                'num_classes': analyzer.num_classes,
                'class_names': analyzer.class_names,
                'overall_accuracy': scores_data['correct_prediction'].mean(),
                'avg_max_score': scores_data['max_score'].mean(),
                'avg_true_class_score': scores_data['true_class_score'].mean()
            }
        }
        
        if verbose:
            print(f"\n✨ Direct analysis completed successfully!")
            print(f"📁 Results saved to: {output_dir}")
            print(f"📊 Overall accuracy: {results['summary']['overall_accuracy']:.4f}")
            print(f"🎯 Average confidence: {results['summary']['avg_max_score']:.4f}")
            
        return results
        
    except Exception as e:
        if verbose:
            print(f"❌ Error during direct analysis: {e}")
            import traceback
            traceback.print_exc()
        raise e


def quick_stats(scores_file_path, verbose=True):
    """
    ⚡ Super quick stats without saving files - perfect for quick checks!
    
    Args:
        scores_file_path (str): Path to the scores .txt file
        verbose (bool): Whether to print results
    
    Returns:
        dict: Quick statistics summary
    """
    try:
        analyzer = ScoreAnalyzer(verbose=False)  # Quiet loading
        scores_data = analyzer.load_scores_file(scores_file_path)
        
        # Quick calculations
        overall_accuracy = scores_data['correct_prediction'].mean()
        avg_max_score = scores_data['max_score'].mean()
        avg_true_class_score = scores_data['true_class_score'].mean()
        
        # Confidence analysis
        high_conf_mask = scores_data['max_score'] >= 0.8
        low_conf_mask = scores_data['max_score'] <= 0.5
        
        high_conf_acc = scores_data[high_conf_mask]['correct_prediction'].mean() if high_conf_mask.sum() > 0 else 0
        low_conf_acc = scores_data[low_conf_mask]['correct_prediction'].mean() if low_conf_mask.sum() > 0 else 0
        
        # Per-class quick stats
        class_stats = {}
        for i, class_name in enumerate(analyzer.class_names):
            score_col = f'score_{class_name}'
            all_scores = scores_data[score_col]
            true_class_mask = scores_data['true_label'] == i
            true_class_scores = scores_data[true_class_mask][score_col]
            
            class_stats[class_name] = {
                'mean_score_all': all_scores.mean(),
                'mean_score_true_class': true_class_scores.mean() if len(true_class_scores) > 0 else 0.0,
                'true_class_count': len(true_class_scores),
                'std_score_all': all_scores.std()
            }
        
        quick_results = {
            'file_path': scores_file_path,
            'total_samples': len(scores_data),
            'num_classes': analyzer.num_classes,
            'class_names': analyzer.class_names,
            'overall_accuracy': overall_accuracy,
            'avg_max_score': avg_max_score,
            'avg_true_class_score': avg_true_class_score,
            'high_confidence_samples': high_conf_mask.sum(),
            'high_confidence_accuracy': high_conf_acc,
            'low_confidence_samples': low_conf_mask.sum(),
            'low_confidence_accuracy': low_conf_acc,
            'class_stats': class_stats
        }
        
        if verbose:
            print(f"\n⚡ QUICK STATS for {Path(scores_file_path).name}")
            print("=" * 60)
            print(f"📊 Total samples: {quick_results['total_samples']}")
            print(f"🎯 Overall accuracy: {overall_accuracy:.4f}")
            print(f"📈 Average confidence: {avg_max_score:.4f}")
            print(f"🔍 Average true class score: {avg_true_class_score:.4f}")
            print(f"🚀 High confidence (≥0.8): {quick_results['high_confidence_samples']} samples (acc: {high_conf_acc:.4f})")
            print(f"⚠️ Low confidence (≤0.5): {quick_results['low_confidence_samples']} samples (acc: {low_conf_acc:.4f})")
            
            print(f"\n📋 Class-wise mean scores:")
            for class_name, stats in class_stats.items():
                print(f"   {class_name:<30}: All={stats['mean_score_all']:.4f}, True={stats['mean_score_true_class']:.4f} (n={stats['true_class_count']})")
        
        return quick_results
        
    except Exception as e:
        if verbose:
            print(f"❌ Error in quick stats: {e}")
        raise e


def main():
    """Main function - now supports both direct execution and CLI."""
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        # CLI mode
        parser = argparse.ArgumentParser(description='📊 Analyze PlasmodiumClassification inference scores')
        parser.add_argument('scores_file', help='Path to the scores .txt file')
        parser.add_argument('-o', '--output', default='score_analysis', help='Output directory (default: score_analysis)')
        parser.add_argument('-v', '--verbose', action='store_true', default=True, help='Verbose output')
        parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
        
        args = parser.parse_args()
        
        try:
            # Use direct analysis function
            results = analyze_scores_direct(
                scores_file_path=args.scores_file,
                output_dir=args.output,
                create_plots=not args.no_plots,
                verbose=args.verbose
            )
            
        except Exception as e:
            print(f"❌ Error during CLI analysis: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Direct execution mode - customize these paths! 🎯
        print("🎯 PlasmodiumClassification Score Analysis - Direct Mode")
        print("=" * 60)
        
        # 🔧 CUSTOMIZE THESE PATHS FOR YOUR ANALYSIS!
        scores_file = r"X:\datn\6cls_results\mobilenetv4_hybrid_medium.ix_e550_r384_in1\PlasmodiumClassification-1\results_kaggle\mobilenetv4_hybrid_medium.ix_e550_r384_in1k\simple_inference_results\train_scores_6cls_vs_7cls.txt"
        output_directory = "detailed_score_analysis"
        
        # Check if file exists
        if not os.path.exists(scores_file):
            print(f"❌ Scores file not found: {scores_file}")
            print("\n🔧 Please update the 'scores_file' path in the main() function!")
            print("💡 Example paths to try:")
            print("   - inference_scores_test.txt")
            print("   - test_scores_6cls_vs_7cls.txt") 
            print("   - path/to/your/scores.txt")
            return
        
        try:
            # Option 1: Quick stats only (fast)
            print("🚀 Running quick stats analysis...")
            quick_results = quick_stats(scores_file, verbose=True)
            
            print("\n" + "="*60)
            
            # Option 2: Full analysis with plots (slower but comprehensive)
            print("🎨 Running full analysis with visualizations...")
            full_results = analyze_scores_direct(
                scores_file_path=scores_file,
                output_dir=output_directory,
                create_plots=True,
                verbose=True
            )
            
            print(f"\n🎉 Analysis complete! Check results in: {output_directory}")
            
        except Exception as e:
            print(f"❌ Error during direct execution: {e}")
            import traceback
            traceback.print_exc()


# Additional convenience functions for common use cases
def analyze_multiple_files(file_list, output_base_dir="multi_analysis", verbose=True):
    """
    🔄 Analyze multiple score files at once - perfect for comparing models!
    
    Args:
        file_list (list): List of score file paths
        output_base_dir (str): Base directory for outputs
        verbose (bool): Whether to print progress
    
    Returns:
        dict: Results for each file
    """
    results = {}
    
    for i, file_path in enumerate(file_list, 1):
        if verbose:
            print(f"\n📂 [{i}/{len(file_list)}] Analyzing: {Path(file_path).name}")
        
        try:
            file_name = Path(file_path).stem
            output_dir = Path(output_base_dir) / f"analysis_{file_name}"
            
            result = analyze_scores_direct(
                scores_file_path=file_path,
                output_dir=str(output_dir),
                create_plots=True,
                verbose=verbose
            )
            
            results[file_name] = result
            
        except Exception as e:
            if verbose:
                print(f"❌ Error analyzing {file_path}: {e}")
            results[Path(file_path).stem] = {'error': str(e)}
    
    if verbose:
        print(f"\n✨ Multi-file analysis complete! Results in: {output_base_dir}")
        print(f"📊 Successfully analyzed: {len([r for r in results.values() if 'error' not in r])}/{len(file_list)} files")
    
    return results


def compare_models_quick(file_dict, verbose=True):
    """
    ⚡ Quick comparison of multiple models - just the key metrics!
    
    Args:
        file_dict (dict): {'model_name': 'path/to/scores.txt'}
        verbose (bool): Whether to print comparison table
    
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, file_path in file_dict.items():
        try:
            quick_result = quick_stats(file_path, verbose=False)
            comparison_data.append({
                'Model': model_name,
                'Accuracy': quick_result['overall_accuracy'],
                'Avg_Confidence': quick_result['avg_max_score'],
                'Avg_True_Score': quick_result['avg_true_class_score'],
                'High_Conf_Samples': quick_result['high_confidence_samples'],
                'High_Conf_Acc': quick_result['high_confidence_accuracy'],
                'Low_Conf_Samples': quick_result['low_confidence_samples'],
                'Low_Conf_Acc': quick_result['low_confidence_accuracy'],
                'Total_Samples': quick_result['total_samples']
            })
        except Exception as e:
            if verbose:
                print(f"❌ Error with {model_name}: {e}")
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if verbose and not comparison_df.empty:
        print("\n🏆 MODEL COMPARISON TABLE")
        print("=" * 100)
        print(comparison_df.to_string(index=False, float_format='{:.4f}'.format))
        
        # Find best model
        best_acc_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_conf_model = comparison_df.loc[comparison_df['Avg_Confidence'].idxmax(), 'Model']
        
        print(f"\n🎯 Best accuracy: {best_acc_model}")
        print(f"🚀 Highest confidence: {best_conf_model}")
    
    return comparison_df


if __name__ == "__main__":
    # Examples of different usage patterns:
    
    # 1. CLI mode (if arguments provided)
    # 2. Direct execution mode (if no arguments)
    # 3. You can also import and use functions directly:
    #
    # from analyze_scores import quick_stats, analyze_scores_direct
    # results = quick_stats("my_scores.txt")
    # full_analysis = analyze_scores_direct("my_scores.txt", "my_output")
    
    main()
