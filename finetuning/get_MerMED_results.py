import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate results and run statistical comparisons")
    parser.add_argument("--result_dir", type=str, default="/path/to/MerMED_Results")
    parser.add_argument("--output_dir", type=str, default="./aggregated_results")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[10, 30, 50, 100], help="List of train sizes to process")
    parser.add_argument("--modality", type=str, nargs="+", default=['path', 'skin', 'cpf', 'oct', 'cxr', 'ct', 'us'], choices=['path', 'skin', 'cpf', 'oct', 'cxr', 'ct', 'us'], help="List of modalities to process")
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
train_sizes_list = args.train_sizes  # List of train sizes to iterate through
modalities_list = args.modality  # List of modalities to iterate through
seeds = args.seeds  # List of seeds to iterate through
# compared_methods = [
#     f"MedFM_Balanced_Medium_ViT_{train_sizes}",
#     # f"PanDerm_{train_sizes}",
#     # f"RETFound_CPF_{train_sizes}",
#     # f"RETFound_OCT_{train_sizes}",
#     f"UNI_{train_sizes}",
#     # f"Rad-Dino_{train_sizes}",
#     # f"USFM_{train_sizes}",
#     # f"Merlin_{train_sizes}",
#     f"UniMed_CLIP_{train_sizes}",
#     f"BioMedCLIP_{train_sizes}",
#     f"Dino_{train_sizes}",
# ]
# compared_datasets = [
#     # "APTOS2019", "CRFO-v4", "Glaucoma_fundus", "IDRiD", "JSIEC",
#     # "MESSIDOR2", "PAPILA",
#     # "FM-AMD", "FM-DR", "FM-Glaucoma", "FM-MMD", "DRCR_CFP",
#     # "FM-CKD", "Seed_Cataract", 
#     # "OCTDL", "OCTID", "DRCR_OCT",
#     # "COVIDx-CXR4", 
#     # "TBX11K", "rsna_pneumonia", "siim_acr_pneumothorax", "CBIS_DDSM", 
#     # "chest-ctscan-images", "IQ-OTHNCCD", "SARS-COV-2", "HRCTCov19", "iCTCF", "RAPIER_CT",
#     "CRC-VAL-HE-7K", "PanNuke", "BreakHis", "Chaoyang", "LC25000", "RAPIER_Gastric",
#     # "TCGA",
#     # "BUSC", "BUSI", "US3M", "BrEaST",
#     # "BCN20000", "Derm7pt", "Dermnet", "HAM10000_clean", "pad-ufes"
# ]

def get_single_modality_methods(train_size):
    return {
        "cpf": f"RETFound_CPF_{train_size}",
        "oct": f"RETFound_OCT_{train_size}",
        "cxr": f"Rad-Dino_{train_size}",
        "ct": f"Merlin_{train_size}",
        "path": f"UNI_{train_size}",
        "us": f"USFM_{train_size}",
        "skin": f"PanDerm_{train_size}"
    }

def get_compared_methods(modality, train_size):
    single_modality_methods = get_single_modality_methods(train_size)
    return [
        f"MedFM_Balanced_Medium_ViT_{train_size}",
        single_modality_methods[modality],
        f"UniMed_CLIP_{train_size}",
        f"BioMedCLIP_{train_size}",
        f"Dino_{train_size}",
    ]

dataset_group = {
    "cpf": ["APTOS2019", "CRFO-v4", "Glaucoma_fundus", "IDRiD", "JSIEC", "MESSIDOR2", "PAPILA",
            "FM-AMD", "FM-DR", "FM-Glaucoma", "FM-MMD", "DRCR_CFP"],
    # "FM-CKD", "Seed_Cataract", 
    "oct": ["OCTDL", "OCTID", "DRCR_OCT"],
    # "COVIDx-CXR4", 
    "cxr": ["TBX11K", "rsna_pneumonia", "siim_acr_pneumothorax", "CBIS_DDSM"], 
    "ct": ["chest-ctscan-images", "IQ-OTHNCCD", "SARS-COV-2", "HRCTCov19", "iCTCF", "RAPIER_CT"],
    "path": ["CRC-VAL-HE-7K", "PanNuke", "BreakHis", "Chaoyang", "LC25000", "RAPIER_Gastric", "MIDOG25", "AMi-Br"],
    # "TCGA",
    "us": ["BUSC", "BUSI", "US3M", "BrEaST"],
    "skin": ["BCN20000", "Derm7pt", "Dermnet", "HAM10000_clean", "pad-ufes"]
}

# "BalancedAcc"
metrics = ["AUCROC", "AUCPR", "Sensitivity", "Specificity", "F1", "Brier", "Acc", "BalancedAcc"]
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

# Main processing loop over all combinations of train sizes and modalities
for train_size in train_sizes_list:
    for modality in modalities_list:
        print(f"\n{'='*80}")
        print(f"Processing: Modality={modality}, Train Size={train_size}")
        print(f"{'='*80}\n")
        
        # Get methods and datasets for this modality
        compared_methods = get_compared_methods(modality, train_size)
        compared_datasets = dataset_group[modality]
        
        all_result_dir = f"all_{modality}_results_{train_size}"
        os.makedirs(os.path.join(output_dir, all_result_dir), exist_ok=True)
        
        # Dictionary to store all results by dataset and method
        dataset_method_results = {dataset: {method: [] for method in compared_methods} for dataset in compared_datasets}

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

        # --------------------- NEW CODE: Generating Comprehensive Summary CSV --------------------- #
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
                    "Method": f"MerMED_{train_size}" if "MedFM_" in method else method
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

        # --------------------- NEW CODE: Statistical Analysis for AUC and F1 only --------------------- #
        print("\nGenerating statistical analysis for AUCROC and F1 only...")
        
        # List to store statistical comparison results
        stat_analysis_rows = []
        
        # Focus only on AUCROC and F1 metrics
        # analysis_metrics = ["AUCROC", "F1", "BalancedAcc"]
        analysis_metrics = ["AUCROC", "F1"]
        
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
                            "Best_Method": f"MerMED_{train_size}" if "MedFM_" in best_method else best_method,
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
        
        print(f"Analysis complete for Modality={modality}, Train Size={train_size}!\n")

print("\n" + "="*80)
print("All analyses complete!")
print("="*80)