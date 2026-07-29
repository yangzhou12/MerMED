import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate results and run statistical comparisons")
    parser.add_argument("--result_dir", type=str, default="/path/to/MerMED_Results_External")
    parser.add_argument("--output_dir", type=str, default="./aggregated_fairness_results")
    parser.add_argument("--train_sizes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 123, 2025])
    parser.add_argument("--confidence_level", type=float, default=0.95)
    # Non-parametric / bootstrap options
    parser.add_argument("--use_mwu", action="store_true", help="Use Mann-Whitney U for between-method comparisons")
    parser.add_argument("--use_bootstrap", action="store_true", help="Use bootstrap percentile CI for metric means")
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2025)
    return parser.parse_args()


args = parse_args()

result_dir = args.result_dir
output_dir = args.output_dir
train_sizes = args.train_sizes
seeds = args.seeds  # List of seeds to iterate through
compared_methods = [
    f"MerMED_{train_sizes}",
    # f"PanDerm_{train_sizes}",
    # f"RETFound_CPF_{train_sizes}",
    f"RETFound_OCT_{train_sizes}",
    # f"UNI_{train_sizes}",
    # f"Rad-Dino_{train_sizes}",
    # f"USFM_{train_sizes}",
    # f"Merlin_{train_sizes}",
    # f"UniMed_CLIP_{train_sizes}",
    f"BioMedCLIP_{train_sizes}",
    f"Dino_{train_sizes}",
]
# Datasets to aggregate fairness metrics over. Each needs per-subgroup metrics
# written by main_finetune_fairness.py (see its --sensitive_attr flag).
compared_datasets = [
    "DRCR_OCT",
]

metrics = ["AUCROC", "AUCPR", "Sensitivity", "Specificity", "F1", "Brier", "Acc", "BalancedAcc"]
confidence_level = args.confidence_level  # 95% confidence interval

# Fairness attributes to analyze
fairness_attributes = ["gender", "age_group"]


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, iters: int = 10000, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng(args.bootstrap_seed)
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return values[0], values[0]
    boot_means = np.empty(iters, dtype=float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(values[idx])
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return float(lower), float(upper)


def mann_whitney_analysis(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 1 or y.size < 1:
        return np.nan, np.nan, np.nan, np.nan
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)
    # Common-language effect size (AUC of x>y)
    auc_u = u_stat / (x.size * y.size)
    # Cliff's delta
    cliffs_delta = 2 * auc_u - 1
    return u_stat, p_val, auc_u, cliffs_delta

# all_result_dir = f"all_path_results_{train_sizes}"
# all_result_dir = f"all_skin_results_{train_sizes}"
all_result_dir = f"all_oct_results_{train_sizes}"
# all_result_dir = f"all_cpf_results_{train_sizes}"
# all_result_dir = f"all_cxr_results_{train_sizes}"
# all_result_dir = f"all_ct_results_{train_sizes}"
# all_result_dir = f"all_us_results_{train_sizes}"
# os.makedirs(os.path.join(result_dir, all_result_dir), exist_ok=True)
os.makedirs(os.path.join(output_dir, all_result_dir), exist_ok=True)

# Dictionary to store all results by dataset and method
dataset_method_results = {dataset: {method: [] for method in compared_methods} for dataset in compared_datasets}

# Dictionary to store fairness results by dataset, method, and attribute
fairness_groups_results = {
    dataset: {
        method: {
            attr: [] for attr in fairness_attributes
        } for method in compared_methods
    } for dataset in compared_datasets
}

fairness_disparities_results = {
    dataset: {
        method: {
            attr: [] for attr in fairness_attributes
        } for method in compared_methods
    } for dataset in compared_datasets
}

# Step 1: Collect all raw results by dataset, method, and seed
print("Collecting raw data across all seeds...")
for seed in seeds:
    print(f"Processing seed: {seed}")
    suffix = f"_seed{seed}_outputs"
    
    for dataset in compared_datasets:
        for method in compared_methods:
            result_folder = method + suffix
            result_path = os.path.join(result_dir, result_folder, dataset, "metrics_test.csv")
            
            try:
                results = pd.read_csv(result_path)
                # Round off all the decimal values
                results = results.round(4)
                results_dict = results.iloc[-1][1:].to_dict()
                results_dict["Seed"] = seed
                dataset_method_results[dataset][method].append(results_dict)
            except FileNotFoundError:
                print(f"Warning: File not found - {result_path}")
                continue
            
            # Collect fairness results for each attribute
            for attr in fairness_attributes:
                # Collect group-level fairness metrics
                fairness_groups_path = os.path.join(
                    result_dir, result_folder, dataset, f"fairness_{attr}_groups_test.csv"
                )
                try:
                    fairness_groups = pd.read_csv(fairness_groups_path)
                    fairness_groups = fairness_groups.round(4)
                    fairness_groups["Seed"] = seed
                    fairness_groups_results[dataset][method][attr].append(fairness_groups)
                except FileNotFoundError:
                    print(f"Warning: Fairness groups file not found - {fairness_groups_path}")
                
                # Collect disparity-level fairness metrics
                fairness_disparities_path = os.path.join(
                    result_dir, result_folder, dataset, f"fairness_{attr}_disparities_test.csv"
                )
                try:
                    fairness_disparities = pd.read_csv(fairness_disparities_path)
                    fairness_disparities = fairness_disparities.round(4)
                    fairness_disparities["Seed"] = seed
                    fairness_disparities_results[dataset][method][attr].append(fairness_disparities)
                except FileNotFoundError:
                    print(f"Warning: Fairness disparities file not found - {fairness_disparities_path}")

# --------------------- ORIGINAL CODE: Generating Comprehensive Summary CSV --------------------- #
print("\nGenerating comprehensive summary CSV file...")

# Create a list to store all summary rows
summary_rows = []

# Process each dataset
for dataset in compared_datasets:
    print(f"Processing dataset: {dataset} for summary")
    
    for method in compared_methods:
        seed_results = dataset_method_results[dataset][method]
        
        if not seed_results:
            continue
            
        # Create DataFrame from seed results
        seed_df = pd.DataFrame(seed_results)
        
        # Prepare a row for this dataset-method combination
        summary_row = {
            "Dataset": dataset,
            "Method": method
        }
        
        # Process each metric
        for metric in metrics:
            if metric not in seed_df.columns:
                summary_row[f"{metric}_mean"] = "N/A"
                if metric == "AUCROC":  # Only add CI for AUCROC
                    summary_row["AUCROC_ci_lower"] = "N/A"
                    summary_row["AUCROC_ci_upper"] = "N/A"
                continue
                
            values = seed_df[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            
            # Add mean value for all metrics
            summary_row[f"{metric}_mean"] = mean_val
            
            # Calculate and add CI only for AUCROC
            if metric == "AUCROC":
                if args.use_bootstrap:
                    ci_low, ci_up = bootstrap_ci(values, confidence=confidence_level, iters=args.bootstrap_iters)
                    summary_row["AUCROC_ci_lower"] = ci_low
                    summary_row["AUCROC_ci_upper"] = ci_up
                else:
                    n = len(values)
                    t_critical = stats.t.ppf((1 + confidence_level) / 2, n-1)
                    margin = t_critical * (std_val / np.sqrt(n))
                    summary_row["AUCROC_ci_lower"] = mean_val - margin
                    summary_row["AUCROC_ci_upper"] = mean_val + margin
        
        # Add to the summary rows
        summary_rows.append(summary_row)

# Create and save the comprehensive summary CSV
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_file_path = os.path.join(output_dir, all_result_dir, "comprehensive_summary.csv")
    summary_df.to_csv(summary_file_path, index=False)
    print(f"Comprehensive summary saved to: {summary_file_path}")
else:
    print("No data available for comprehensive summary.")

# --------------------- ORIGINAL CODE: Statistical Analysis for AUC and F1 only --------------------- #
print("\nGenerating statistical analysis for AUCROC and F1 only...")

# List to store statistical comparison results
stat_analysis_rows = []

# Focus only on AUCROC and F1 metrics
analysis_metrics = ["AUCROC", "F1", "BalancedAcc"]

# Process each dataset
for dataset in compared_datasets:
    print(f"Processing dataset: {dataset} for statistical analysis")
    
    # Process each of the two metrics
    for metric in analysis_metrics:
        print(f"  Analyzing {metric}")
        
        # Prepare data for each method
        method_stats = {}
        valid_methods = []
        
        for method in compared_methods:
            seed_results = dataset_method_results[dataset][method]
            
            if not seed_results:
                continue
            
            seed_df = pd.DataFrame(seed_results)
            
            # Check if we have this metric
            if metric not in seed_df.columns:
                continue
            
            # Calculate statistics for this method
            method_stats[method] = {}
            valid_methods.append(method)
            
            values = seed_df[metric].values
            method_stats[method]["values"] = values
            method_stats[method]["mean"] = np.mean(values)
            method_stats[method]["std"] = np.std(values, ddof=1)
            
            # Calculate 95% CI (t or bootstrap)
            if args.use_bootstrap:
                ci_low, ci_up = bootstrap_ci(values, confidence=confidence_level, iters=args.bootstrap_iters)
            else:
                n = len(values)
                std = method_stats[method]["std"]
                t_critical = stats.t.ppf((1 + confidence_level) / 2, n-1)
                margin = t_critical * (std / np.sqrt(n))
                ci_low = method_stats[method]["mean"] - margin
                ci_up = method_stats[method]["mean"] + margin
            method_stats[method]["ci_lower"] = ci_low
            method_stats[method]["ci_upper"] = ci_up
        
        if not valid_methods:
            print(f"    No valid methods found for {dataset} and {metric}, skipping.")
            continue
        
        # Identify best method for this metric and dataset
        # best_method = max(valid_methods, key=lambda m: method_stats[m]["mean"])
        best_method = compared_methods[0]
        
        # Compare best method against all others
        for method in valid_methods:
            if method == best_method:
                continue
            
            best_values = method_stats[best_method]["values"]
            other_values = method_stats[method]["values"]
            
            # Check if we have enough data
            if len(best_values) >= 2 and len(other_values) >= 2:
                # Always compute both tests
                if len(best_values) == len(other_values):
                    t_stat, t_p_value = stats.ttest_rel(best_values, other_values)
                    t_type = "Paired t-test"
                else:
                    t_stat, t_p_value = stats.ttest_ind(best_values, other_values, equal_var=False)
                    t_type = "Welch's t-test"
                u_res = stats.mannwhitneyu(best_values, other_values, alternative="two-sided")
                u_stat = float(u_res.statistic)
                u_p_value = float(u_res.pvalue)
                auc_u = u_stat / (len(best_values) * len(other_values))
                cliffs_delta = 2 * auc_u - 1
                test_type = f"{t_type} + Mann-Whitney U"
                
                # Calculate effect size (Cohen's d) and mean diff
                mean_diff = method_stats[best_method]["mean"] - method_stats[method]["mean"]
                pooled_std = np.sqrt((method_stats[best_method]["std"]**2 + method_stats[method]["std"]**2) / 2)
                cohens_d = (mean_diff / pooled_std) if (not np.isnan(pooled_std) and pooled_std != 0) else float('inf')
                
                # Use asterisks to represent significance level based on t-test p-value
                if t_p_value < 0.001:
                    sig_symbol = "***"
                elif t_p_value < 0.01:
                    sig_symbol = "**"
                elif t_p_value < 0.05:
                    sig_symbol = "*"
                else:
                    sig_symbol = "ns"
                
                # Create a row for the statistical analysis
                analysis_row = {
                    "Dataset": dataset,
                    "Metric": metric,
                    "Best_Method": best_method,
                    "Best_Method_Mean": method_stats[best_method]["mean"],
                    "Best_Method_Std": method_stats[best_method]["std"],
                    "Best_Method_CI_Lower": method_stats[best_method]["ci_lower"],
                    "Best_Method_CI_Upper": method_stats[best_method]["ci_upper"],
                    "Compared_Method": method,
                    "Compared_Method_Mean": method_stats[method]["mean"],
                    "Compared_Method_Std": method_stats[method]["std"],
                    "Compared_Method_CI_Lower": method_stats[method]["ci_lower"],
                    "Compared_Method_CI_Upper": method_stats[method]["ci_upper"],
                    "Mean_Difference": mean_diff,
                    "Test_Type": test_type,
                    "t_statistic": t_stat,
                    "t_p_value": t_p_value,
                    "u_statistic": u_stat,
                    "u_p_value": u_p_value,
                    "p_value": t_p_value,
                    "Significance": sig_symbol,
                    "Cohens_d": cohens_d,
                    "MWU_AUC": auc_u,
                    "Cliffs_delta": cliffs_delta,
                    "Improvement_Percentage": (mean_diff / method_stats[method]["mean"]) * 100 if method_stats[method]["mean"] != 0 else float('inf')
                }
                
                stat_analysis_rows.append(analysis_row)

# Create and save the statistical analysis CSV
if stat_analysis_rows:
    stat_analysis_df = pd.DataFrame(stat_analysis_rows)
    stat_analysis_file_path = os.path.join(output_dir, all_result_dir, "auc_f1_statistical_analysis.csv")
    stat_analysis_df.to_csv(stat_analysis_file_path, index=False)
    print(f"AUC and F1 statistical analysis saved to: {stat_analysis_file_path}")
else:
    print("No data available for statistical analysis.")

# --------------------- NEW CODE: Fairness Analysis --------------------- #
print("\n" + "="*80)
print("FAIRNESS ANALYSIS")
print("="*80)

# Process fairness group metrics
print("\nGenerating fairness group-level summary...")
fairness_group_summary_rows = []

for dataset in compared_datasets:
    print(f"Processing dataset: {dataset} for fairness groups")
    
    for method in compared_methods:
        for attr in fairness_attributes:
            seed_fairness_groups = fairness_groups_results[dataset][method][attr]
            
            if not seed_fairness_groups:
                continue
            
            # Concatenate all seed results
            all_groups_df = pd.concat(seed_fairness_groups, ignore_index=True)
            
            # Get unique groups
            unique_groups = all_groups_df['Group'].unique()
            
            for group in unique_groups:
                group_df = all_groups_df[all_groups_df['Group'] == group]
                
                summary_row = {
                    "Dataset": dataset,
                    "Method": method,
                    "Attribute": attr,
                    "Group": group,
                    "Count_mean": group_df['count'].mean(),
                    "Count_std": group_df['count'].std(ddof=1) if len(group_df) > 1 else 0,
                }
                
                # Calculate mean and std for each metric
                fairness_metrics = ['acc', 'balanced_acc', 'auc_roc', 'auc_pr', 'f1', 'ece']
                for metric in fairness_metrics:
                    if metric in group_df.columns:
                        summary_row[f"{metric}_mean"] = group_df[metric].mean()
                        summary_row[f"{metric}_std"] = group_df[metric].std(ddof=1) if len(group_df) > 1 else 0
                
                fairness_group_summary_rows.append(summary_row)

# Save fairness group summary
if fairness_group_summary_rows:
    fairness_group_summary_df = pd.DataFrame(fairness_group_summary_rows)
    fairness_group_summary_path = os.path.join(output_dir, all_result_dir, "fairness_groups_summary.csv")
    fairness_group_summary_df.to_csv(fairness_group_summary_path, index=False)
    print(f"Fairness group summary saved to: {fairness_group_summary_path}")
else:
    print("No data available for fairness group summary.")

# Process fairness disparity metrics
print("\nGenerating fairness disparity-level summary...")
fairness_disparity_summary_rows = []

for dataset in compared_datasets:
    print(f"Processing dataset: {dataset} for fairness disparities")
    
    for method in compared_methods:
        for attr in fairness_attributes:
            seed_fairness_disparities = fairness_disparities_results[dataset][method][attr]
            
            if not seed_fairness_disparities:
                continue
            
            # Concatenate all seed results
            all_disparities_df = pd.concat(seed_fairness_disparities, ignore_index=True)
            
            summary_row = {
                "Dataset": dataset,
                "Method": method,
                "Attribute": attr,
            }
            
            # Calculate mean and std for disparity metrics
            disparity_metrics = [
                'acc_max_diff', 'acc_std', 'balanced_acc_max_diff', 'balanced_acc_std',
                'auc_roc_max_diff', 'auc_roc_std', 'f1_max_diff', 'f1_std',
                'ece_max_diff', 'ece_std', 'demographic_parity_diff'
            ]
            
            for metric in disparity_metrics:
                if metric in all_disparities_df.columns:
                    values = all_disparities_df[metric].values
                    summary_row[f"{metric}_mean"] = np.mean(values)
                    summary_row[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0
                    
                    # Calculate CI for key disparity metrics
                    if metric.endswith('_max_diff') or metric == 'demographic_parity_diff':
                        if args.use_bootstrap:
                            ci_low, ci_up = bootstrap_ci(values, confidence=confidence_level, iters=args.bootstrap_iters)
                        else:
                            n = len(values)
                            std = summary_row[f"{metric}_std"]
                            if n > 1:
                                t_critical = stats.t.ppf((1 + confidence_level) / 2, n-1)
                                margin = t_critical * (std / np.sqrt(n))
                                ci_low = summary_row[f"{metric}_mean"] - margin
                                ci_up = summary_row[f"{metric}_mean"] + margin
                            else:
                                ci_low = ci_up = summary_row[f"{metric}_mean"]
                        summary_row[f"{metric}_ci_lower"] = ci_low
                        summary_row[f"{metric}_ci_upper"] = ci_up
            
            fairness_disparity_summary_rows.append(summary_row)

# Save fairness disparity summary
if fairness_disparity_summary_rows:
    fairness_disparity_summary_df = pd.DataFrame(fairness_disparity_summary_rows)
    fairness_disparity_summary_path = os.path.join(output_dir, all_result_dir, "fairness_disparities_summary.csv")
    fairness_disparity_summary_df.to_csv(fairness_disparity_summary_path, index=False)
    print(f"Fairness disparity summary saved to: {fairness_disparity_summary_path}")
else:
    print("No data available for fairness disparity summary.")

# Statistical comparison of fairness disparities
print("\nGenerating fairness disparity statistical analysis...")
fairness_stat_rows = []

for dataset in compared_datasets:
    print(f"Processing dataset: {dataset} for fairness disparity statistical analysis")
    
    for attr in fairness_attributes:
        # Key disparity metrics to analyze
        key_disparity_metrics = ['auc_roc_max_diff', 'f1_max_diff', 'demographic_parity_diff']
        
        for disparity_metric in key_disparity_metrics:
            print(f"  Analyzing {attr} - {disparity_metric}")
            
            # Prepare data for each method
            method_stats = {}
            valid_methods = []
            
            for method in compared_methods:
                seed_fairness_disparities = fairness_disparities_results[dataset][method][attr]
                
                if not seed_fairness_disparities:
                    continue
                
                all_disparities_df = pd.concat(seed_fairness_disparities, ignore_index=True)
                
                if disparity_metric not in all_disparities_df.columns:
                    continue
                
                method_stats[method] = {}
                valid_methods.append(method)
                
                values = all_disparities_df[disparity_metric].values
                method_stats[method]["values"] = values
                method_stats[method]["mean"] = np.mean(values)
                method_stats[method]["std"] = np.std(values, ddof=1)
                
                # Calculate CI
                if args.use_bootstrap:
                    ci_low, ci_up = bootstrap_ci(values, confidence=confidence_level, iters=args.bootstrap_iters)
                else:
                    n = len(values)
                    std = method_stats[method]["std"]
                    if n > 1:
                        t_critical = stats.t.ppf((1 + confidence_level) / 2, n-1)
                        margin = t_critical * (std / np.sqrt(n))
                        ci_low = method_stats[method]["mean"] - margin
                        ci_up = method_stats[method]["mean"] + margin
                    else:
                        ci_low = ci_up = method_stats[method]["mean"]
                method_stats[method]["ci_lower"] = ci_low
                method_stats[method]["ci_upper"] = ci_up
            
            if not valid_methods:
                continue
            
            # For fairness, lower disparity is better
            best_method = compared_methods[0]
            
            # Compare best method against all others
            for method in valid_methods:
                if method == best_method:
                    continue
                
                best_values = method_stats[best_method]["values"]
                other_values = method_stats[method]["values"]
                
                if len(best_values) >= 2 and len(other_values) >= 2:
                    # Perform statistical tests
                    if len(best_values) == len(other_values):
                        t_stat, t_p_value = stats.ttest_rel(best_values, other_values)
                        t_type = "Paired t-test"
                    else:
                        t_stat, t_p_value = stats.ttest_ind(best_values, other_values, equal_var=False)
                        t_type = "Welch's t-test"
                    
                    u_res = stats.mannwhitneyu(best_values, other_values, alternative="two-sided")
                    u_stat = float(u_res.statistic)
                    u_p_value = float(u_res.pvalue)
                    auc_u = u_stat / (len(best_values) * len(other_values))
                    cliffs_delta = 2 * auc_u - 1
                    test_type = f"{t_type} + Mann-Whitney U"
                    
                    # Calculate effect size
                    mean_diff = method_stats[best_method]["mean"] - method_stats[method]["mean"]
                    pooled_std = np.sqrt((method_stats[best_method]["std"]**2 + method_stats[method]["std"]**2) / 2)
                    cohens_d = (mean_diff / pooled_std) if (not np.isnan(pooled_std) and pooled_std != 0) else float('inf')
                    
                    # Significance
                    if t_p_value < 0.001:
                        sig_symbol = "***"
                    elif t_p_value < 0.01:
                        sig_symbol = "**"
                    elif t_p_value < 0.05:
                        sig_symbol = "*"
                    else:
                        sig_symbol = "ns"
                    
                    fairness_row = {
                        "Dataset": dataset,
                        "Attribute": attr,
                        "Disparity_Metric": disparity_metric,
                        "Best_Method": best_method,
                        "Best_Method_Mean": method_stats[best_method]["mean"],
                        "Best_Method_Std": method_stats[best_method]["std"],
                        "Best_Method_CI_Lower": method_stats[best_method]["ci_lower"],
                        "Best_Method_CI_Upper": method_stats[best_method]["ci_upper"],
                        "Compared_Method": method,
                        "Compared_Method_Mean": method_stats[method]["mean"],
                        "Compared_Method_Std": method_stats[method]["std"],
                        "Compared_Method_CI_Lower": method_stats[method]["ci_lower"],
                        "Compared_Method_CI_Upper": method_stats[method]["ci_upper"],
                        "Mean_Difference": mean_diff,
                        "Test_Type": test_type,
                        "t_statistic": t_stat,
                        "t_p_value": t_p_value,
                        "u_statistic": u_stat,
                        "u_p_value": u_p_value,
                        "p_value": t_p_value,
                        "Significance": sig_symbol,
                        "Cohens_d": cohens_d,
                        "MWU_AUC": auc_u,
                        "Cliffs_delta": cliffs_delta,
                        "Improvement_Percentage": (mean_diff / method_stats[method]["mean"]) * 100 if method_stats[method]["mean"] != 0 else float('inf')
                    }
                    
                    fairness_stat_rows.append(fairness_row)

# Save fairness disparity statistical analysis
if fairness_stat_rows:
    fairness_stat_df = pd.DataFrame(fairness_stat_rows)
    fairness_stat_path = os.path.join(output_dir, all_result_dir, "fairness_disparities_statistical_analysis.csv")
    fairness_stat_df.to_csv(fairness_stat_path, index=False)
    print(f"Fairness disparity statistical analysis saved to: {fairness_stat_path}")
else:
    print("No data available for fairness disparity statistical analysis.")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
print("\nGenerated files:")
print(f"1. {os.path.join(output_dir, all_result_dir, 'comprehensive_summary.csv')}")
print(f"2. {os.path.join(output_dir, all_result_dir, 'auc_f1_statistical_analysis.csv')}")
print(f"3. {os.path.join(output_dir, all_result_dir, 'fairness_groups_summary.csv')}")
print(f"4. {os.path.join(output_dir, all_result_dir, 'fairness_disparities_summary.csv')}")
print(f"5. {os.path.join(output_dir, all_result_dir, 'fairness_disparities_statistical_analysis.csv')}")