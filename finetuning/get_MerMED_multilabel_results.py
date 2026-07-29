import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multilabel results and run statistical comparisons")
    parser.add_argument("--result_dir", type=str, default="/path/to/MerMED_Results")
    parser.add_argument("--output_dir", type=str, default="./aggregated_multilabel_results")
    parser.add_argument("--train_sizes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 123, 2025])
    parser.add_argument("--confidence_level", type=float, default=0.95)
    # Non-parametric / bootstrap options
    parser.add_argument("--use_mwu", action="store_true", help="Use Mann-Whitney U for between-method comparisons")
    parser.add_argument("--use_bootstrap", action="store_true", help="Use bootstrap percentile CI for metric means")
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2025)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        metavar="NAME:NUM_CLASSES",
        help="Multi-label datasets to aggregate, e.g. --datasets my_cxr:5 my_other:14",
    )
    return parser.parse_args()


def parse_dataset_class_counts(specs):
    """Turn ["name:5", ...] into {"name": 5, ...}."""
    counts = {}
    for spec in specs:
        name, _, num_classes = spec.rpartition(":")
        if not name or not num_classes.isdigit():
            raise SystemExit(f"--datasets expects NAME:NUM_CLASSES entries, got {spec!r}")
        counts[name] = int(num_classes)
    return counts

args = parse_args()

result_dir = args.result_dir
output_dir = args.output_dir
train_sizes = args.train_sizes
seeds = args.seeds  # List of seeds to iterate through
compared_methods = [
    # f"MedFM_Balanced_Medium_ViT_{train_sizes}",
    f"MerMED_{train_sizes}",
    f"Rad-Dino_{train_sizes}",
    # f"UniMed_CLIP_{train_sizes}",
    f"BioMedCLIP_{train_sizes}",
    f"Dino_{train_sizes}",
]

# Dataset-specific number of classes, from --datasets NAME:NUM_CLASSES
dataset_class_counts = parse_dataset_class_counts(args.datasets)

metrics = ["Acc", "BalancedAcc", "AUCROC", "AUCPR", "Sensitivity", "Specificity", "F1", "Brier"]
# metrics = ["Macro_Accuracy", "Macro_AUC_ROC", "Macro_AUC_PR", "Macro_F1", "Macro_Sensitivity", "Macro_Specificity"]
confidence_level = args.confidence_level  # 95% confidence interval


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

all_result_dir = f"all_cxr_ml_{train_sizes}"
os.makedirs(os.path.join(output_dir, all_result_dir), exist_ok=True)

# Dictionary to store all results by dataset and method
dataset_method_results = {dataset: {method: [] for method in compared_methods} for dataset in dataset_class_counts}

# Step 1: Collect all raw results by dataset, method, and seed
print("Collecting raw data across all seeds...")
for seed in seeds:
    print(f"Processing seed: {seed}")
    suffix = f"_seed{seed}_outputs"
    
    for dataset, num_classes in dataset_class_counts.items():
        for method in compared_methods:
            result_folder = method + suffix
            result_path = os.path.join(result_dir, result_folder, dataset, "metrics_test.csv")
            # result_path = os.path.join(result_dir, result_folder, dataset, "results_test.csv")
            
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

# --------------------- NEW CODE: Generating Comprehensive Summary CSV --------------------- #
print("\nGenerating comprehensive summary CSV file...")

# Create a list to store all summary rows
summary_rows = []

# Process each dataset
for dataset, num_classes in dataset_class_counts.items():
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
        
        # Dynamically create per-class metrics based on number of classes
        per_class_metrics = []
        for class_idx in range(num_classes):
            per_class_metrics.extend([
                f"Class_{class_idx}_Acc",
                f"Class_{class_idx}_F1",
                f"Class_{class_idx}_AUCROC",
                f"Class_{class_idx}_AUCPR",
                f"Class_{class_idx}_Sensitivity",
                f"Class_{class_idx}_Specificity"
            ])
        
        # Process each metric - both macro metrics and per-class metrics
        all_metrics = metrics + per_class_metrics
        for metric in all_metrics:
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
            # if metric == "Macro_AUC_ROC":
            if "AUCROC" in metric:
                if args.use_bootstrap:
                    ci_low, ci_up = bootstrap_ci(values, confidence=confidence_level, iters=args.bootstrap_iters)
                    summary_row[f"{metric}_ci_lower"] = ci_low
                    summary_row[f"{metric}_ci_upper"] = ci_up
                else:
                    n = len(values)
                    t_critical = stats.t.ppf((1 + confidence_level) / 2, n-1)
                    margin = t_critical * (std_val / np.sqrt(n))
                    summary_row[f"{metric}_ci_lower"] = mean_val - margin
                    summary_row[f"{metric}_ci_upper"] = mean_val + margin
        
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

# --------------------- NEW CODE: Statistical Analysis for AUC and F1 only --------------------- #
print("\nGenerating statistical analysis for AUCROC and F1 only...")

# List to store statistical comparison results
stat_analysis_rows = []

# # Process each dataset
# for dataset, num_classes in dataset_class_counts.items():
#     print(f"Processing dataset: {dataset} for statistical analysis")

#     # Focus only on AUCROC and F1 metrics
#     analysis_metrics = ["Macro_AUC_ROC", "Macro_F1"]

#     # Dynamically create per-class metrics based on number of classes
#     per_class_metrics = []
#     for class_idx in range(num_classes):
#         per_class_metrics.extend([
#             f"Class_{class_idx}_F1",
#             f"Class_{class_idx}_AUC_ROC",
#         ])
    
#     # Process each metric - both macro metrics and per-class metrics
#     analysis_metrics = analysis_metrics + per_class_metrics

#     # Process each of the two metrics
#     for metric in analysis_metrics:
#         print(f"  Analyzing {metric}")
        
#         # Prepare data for each method
#         method_stats = {}
#         valid_methods = []
        
#         for method in compared_methods:
#             seed_results = dataset_method_results[dataset][method]
            
#             if not seed_results:
#                 continue
            
#             seed_df = pd.DataFrame(seed_results)
            
#             # Check if we have this metric
#             if metric not in seed_df.columns:
#                 continue
            
#             # Calculate statistics for this method
#             method_stats[method] = {}
#             valid_methods.append(method)
            
#             values = seed_df[metric].values
#             method_stats[method]["values"] = values
#             method_stats[method]["mean"] = np.mean(values)
#             method_stats[method]["std"] = np.std(values, ddof=1)
            
#             # Calculate 95% CI
#             n = len(values)
#             std = method_stats[method]["std"]
#             t_critical = stats.t.ppf((1 + confidence_level) / 2, n-1)
#             margin = t_critical * (std / np.sqrt(n))
            
#             method_stats[method]["ci_lower"] = method_stats[method]["mean"] - margin
#             method_stats[method]["ci_upper"] = method_stats[method]["mean"] + margin
        
#         if not valid_methods:
#             print(f"    No valid methods found for {dataset} and {metric}, skipping.")
#             continue
        
#         # Identify best method for this metric and dataset
#         best_method = max(valid_methods, key=lambda m: method_stats[m]["mean"])
        
#         # Compare best method against all others
#         for method in valid_methods:
#             if method == best_method:
#                 continue
            
#             best_values = method_stats[best_method]["values"]
#             other_values = method_stats[method]["values"]
            
#             # Check if we have enough data
#             if len(best_values) >= 2 and len(other_values) >= 2:
#                 # Perform t-test (paired if same length, otherwise independent)
#                 if len(best_values) == len(other_values):
#                     t_stat, p_value = stats.ttest_rel(best_values, other_values)
#                     test_type = "Paired t-test"
#                 else:
#                     t_stat, p_value = stats.ttest_ind(best_values, other_values, equal_var=False)  # Welch's t-test
#                     test_type = "Welch's t-test"
                
#                 # Calculate effect size (Cohen's d)
#                 mean_diff = method_stats[best_method]["mean"] - method_stats[method]["mean"]
#                 pooled_std = np.sqrt((method_stats[best_method]["std"]**2 + method_stats[method]["std"]**2) / 2)
#                 cohens_d = mean_diff / pooled_std if pooled_std != 0 else float('inf')
                
#                 # Use asterisks to represent significance level
#                 if p_value < 0.001:
#                     sig_symbol = "***"
#                 elif p_value < 0.01:
#                     sig_symbol = "**"
#                 elif p_value < 0.05:
#                     sig_symbol = "*"
#                 else:
#                     sig_symbol = "ns"
                
#                 # Create a row for the statistical analysis
#                 analysis_row = {
#                     "Dataset": dataset,
#                     "Metric": metric,
#                     "Best_Method": best_method,
#                     "Best_Method_Mean": method_stats[best_method]["mean"],
#                     "Best_Method_Std": method_stats[best_method]["std"],
#                     "Best_Method_CI_Lower": method_stats[best_method]["ci_lower"],
#                     "Best_Method_CI_Upper": method_stats[best_method]["ci_upper"],
#                     "Compared_Method": method,
#                     "Compared_Method_Mean": method_stats[method]["mean"],
#                     "Compared_Method_Std": method_stats[method]["std"],
#                     "Compared_Method_CI_Lower": method_stats[method]["ci_lower"],
#                     "Compared_Method_CI_Upper": method_stats[method]["ci_upper"],
#                     "Mean_Difference": mean_diff,
#                     "Test_Type": test_type,
#                     "t_statistic": t_stat,
#                     "p_value": p_value,
#                     "Significance": sig_symbol,
#                     "Cohens_d": cohens_d,
#                     "Improvement_Percentage": (mean_diff / method_stats[method]["mean"]) * 100 if method_stats[method]["mean"] != 0 else float('inf')
#                 }
                
#                 stat_analysis_rows.append(analysis_row)

# # Create and save the statistical analysis CSV
# if stat_analysis_rows:
#     stat_analysis_df = pd.DataFrame(stat_analysis_rows)
#     stat_analysis_file_path = os.path.join(result_dir, all_result_dir, "auc_f1_statistical_analysis.csv")
#     stat_analysis_df.to_csv(stat_analysis_file_path, index=False)
#     print(f"AUC and F1 statistical analysis saved to: {stat_analysis_file_path}")
# else:
#     print("No data available for statistical analysis.")

# print("Analysis complete!")

# For storing final results
simplified_tables = []

# Process each dataset
for dataset, num_classes in dataset_class_counts.items():
    print(f"Processing dataset: {dataset}")
    
    # Dynamically create per-class metrics based on number of classes
    per_class_metrics = []
    for class_idx in range(num_classes):
        per_class_metrics.extend([
            f"Class_{class_idx}_F1",
            f"Class_{class_idx}_AUCROC",
        ])
    
    # Combine macro and per-class metrics
    analysis_metrics = metrics + per_class_metrics
    
    # Process each metric
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
        
        if not valid_methods:
            print(f"    No valid methods found for {dataset} and {metric}, skipping.")
            continue
        
        # Identify best method for this metric and dataset
        # best_method = max(valid_methods, key=lambda m: method_stats[m]["mean"])
        best_method = compared_methods[0]
        
        # Create a table with methods as rows and p-values in one column
        table_data = {
            "Method": [],
            f"{metric}_Mean": [],
            "t_statistic": [],
            "Cohens_d": [],
            "p-value": []
        }
        # Add Mann-Whitney U columns if needed
        if args.use_mwu or "AUCROC" in metric:
            table_data["u_statistic"] = []
            table_data["u_p_value"] = []
            table_data["MWU_AUC"] = []
            table_data["Cliffs_delta"] = []
        
        # Add all methods in the original order, marking the best with "Ref."
        for method in compared_methods:
            if method not in valid_methods:
                continue
                
            table_data["Method"].append(method)
            table_data[f"{metric}_Mean"].append(method_stats[method]["mean"])
            
            if method == best_method:
                table_data["p-value"].append("Ref.")
                table_data["t_statistic"].append("Ref.")
                table_data["Cohens_d"].append("Ref.")
                if args.use_mwu or "AUCROC" in metric:
                    table_data["u_statistic"].append("Ref.")
                    table_data["u_p_value"].append("Ref.")
                    table_data["MWU_AUC"].append("Ref.")
                    table_data["Cliffs_delta"].append("Ref.")
            else:
                best_values = method_stats[best_method]["values"]
                other_values = method_stats[method]["values"]
                
                # Calculate p-value
                if len(best_values) >= 2 and len(other_values) >= 2:
                    # Always compute both tests
                    if len(best_values) == len(other_values):
                        t_stat, t_p_value = stats.ttest_rel(best_values, other_values)
                        t_type = "Paired t-test"
                    else:
                        t_stat, t_p_value = stats.ttest_ind(best_values, other_values, equal_var=False)
                        t_type = "Welch's t-test"
                    
                    # Mann-Whitney U test (always compute for AUCROC, or if use_mwu is set)
                    if args.use_mwu or "AUCROC" in metric:
                        u_stat, u_p_value, auc_u, cliffs_delta = mann_whitney_analysis(best_values, other_values)
                    
                    # Calculate effect size (Cohen's d)
                    mean_diff = method_stats[best_method]["mean"] - method_stats[method]["mean"]
                    pooled_std = np.sqrt((method_stats[best_method]["std"]**2 + method_stats[method]["std"]**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std != 0 else float('inf')

                    # Format p-value as requested (use t-test p-value for main p-value column)
                    if t_p_value < 0.001:
                        p_value_formatted = "<0.001"
                    else:
                        p_value_formatted = f"{t_p_value:.3f}"
                else:
                    p_value_formatted = "N/A"
                    t_stat = np.nan
                    cohens_d = np.nan
                    if args.use_mwu or "AUCROC" in metric:
                        u_stat = np.nan
                        u_p_value = np.nan
                        auc_u = np.nan
                        cliffs_delta = np.nan
                
                table_data["p-value"].append(p_value_formatted)
                table_data["t_statistic"].append(t_stat)
                table_data["Cohens_d"].append(cohens_d)
                if args.use_mwu or "AUCROC" in metric:
                    table_data["u_statistic"].append(u_stat)
                    table_data["u_p_value"].append(u_p_value)
                    table_data["MWU_AUC"].append(auc_u)
                    table_data["Cliffs_delta"].append(cliffs_delta)
        
        # Create DataFrame for this metric
        metric_table = pd.DataFrame(table_data)
        
        # Add dataset and metric information
        metric_table["Dataset"] = dataset
        metric_table["Metric"] = metric
        
        # Add to final results
        simplified_tables.append(metric_table)

# Combine all tables and save
if simplified_tables:
    combined_table = pd.concat(simplified_tables)
    
    # Reorganize columns
    base_columns = ["Dataset", "Metric", "Method"] + [col for col in combined_table.columns if "_Mean" in col] + ["p-value", "t_statistic", "Cohens_d"]
    # Add Mann-Whitney U columns if they exist
    mwu_columns = ["u_statistic", "u_p_value", "MWU_AUC", "Cliffs_delta"]
    final_columns = base_columns + [col for col in mwu_columns if col in combined_table.columns]
    combined_table = combined_table[final_columns]
    
    # Save to CSV
    output_path = os.path.join(output_dir, all_result_dir, "simplified_pvalue_analysis.csv")
    combined_table.to_csv(output_path, index=False)
    print(f"Simplified p-value analysis saved to: {output_path}")
else:
    print("No data available for simplified analysis.")

print("Analysis complete!")