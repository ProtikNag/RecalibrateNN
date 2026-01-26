import os
import sys
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple
from openpyxl.styles import Alignment

class TCAVAnalyzer:
    """Analyzer for TCAV (Testing with Concept Activation Vectors) sensitivity scores."""
    
    def __init__(self, root_directory: str):
        """Initialize the analyzer with root directory."""
        self.root_directory = root_directory
        self.csv_files = []
        self.results = {}
        
    def find_csv_files(self) -> List[str]:
        """Navigate through directory tree and list all CSV files."""
        csv_files = []
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        self.csv_files = csv_files
        print(f"Found {len(csv_files)} CSV files")
        return csv_files
    
    def load_and_preprocess_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV file and drop 'Full filepath' column."""
        df = pd.read_csv(csv_path)
        if 'Full filepath' in df.columns:
            df = df.drop(columns=['Full filepath'])
        return df
    
    def group_by_class(self, df: pd.DataFrame):
        """Group data based on Full Class Index."""
        return df.groupby('Full Class Index')
    
    def count_sensitivity_scores(self, group_df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
        """Count positive and negative sensitivity scores for each sensitivity column."""
        sensitivity_cols = [col for col in group_df.columns if col.startswith('sensitivity')]
        counts = {}
        
        for col in sensitivity_cols:
            positive_count = (group_df[col] > 0).sum()
            negative_count = (group_df[col] < 0).sum()
            counts[col] = (positive_count, negative_count)
        
        return counts
    
    def compute_tcav_score(self, positive_count: int, negative_count: int) -> float:
        """Compute TCAV score as positive / (positive + negative)."""
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        return positive_count / total
    
    def perform_hypothesis_test(self, tcav_score: float, total_count: int) -> Dict[str, float]:
        """
        Perform binomial proportion hypothesis test.
        H0: p = 0.5
        H1: p != 0.5
        Returns p-values for both one-sided tests.
        """
        if total_count == 0:
            return {'p_value_positive': 1.0, 'p_value_negative': 1.0, 'p_value_two_sided': 1.0}
        
        # Number of successes (positive counts)
        successes = int(tcav_score * total_count)
        
        # Two-sided test
        p_value_two_sided = stats.binomtest(successes, total_count, p=0.5, alternative='two-sided').pvalue
        
        # One-sided tests
        p_value_positive = stats.binomtest(successes, total_count, p=0.5, alternative='greater').pvalue  # H1: p > 0.5
        p_value_negative = stats.binomtest(successes, total_count, p=0.5, alternative='less').pvalue     # H1: p < 0.5
        
        return {'p_value_positive': p_value_positive, 'p_value_negative': p_value_negative, 'p_value_two_sided': p_value_two_sided}
    
    
    def classify_hypothesis(self, p_value_positive: float, p_value_negative: float, alpha: float = 0.05) -> str:
        """Classify layer based on hypothesis test results."""
        if p_value_positive < alpha:
            return "Positive Sensitivity"
        elif p_value_negative < alpha:
            return "Negative Sensitivity"
        else:
            return "No Significant Sensitivity"
    #This method is not used currently but kept for future reference
    def compute_correlations(self, class_results: Dict) -> pd.DataFrame:
        """Perform pairwise statistical comparisons between artifacts using two-proportion z-test and Fisher's exact test."""
        comparison_results = []
        
        # Process each class
        for class_name, results in class_results.items():
            df = results['layer_results']
            
            # Check if 'Artifact' column exists
            if 'Artifact' not in df.columns:
                print(f"Warning: 'Artifact' column not found in {class_name}")
                continue
            
            # Get unique artifacts
            artifacts = df['Artifact'].unique()
            
            # Pairwise comparison between artifacts
            for i, artifact1 in enumerate(artifacts):
                for artifact2 in artifacts[i+1:]:
                    # Get data for both artifacts
                    df1 = df[df['Artifact'] == artifact1]
                    df2 = df[df['Artifact'] == artifact2]
                    
                    # Find common layers
                    common_layers = set(df1['Layer']).intersection(set(df2['Layer']))
                    
                    for layer in common_layers:
                        layer_data1 = df1[df1['Layer'] == layer].iloc[0]
                        layer_data2 = df2[df2['Layer'] == layer].iloc[0]
                        
                        pos1 = layer_data1['Positive Activations']
                        neg1 = layer_data1['Negative Activations']
                        total1 = pos1 + neg1
                        
                        pos2 = layer_data2['Positive Activations']
                        neg2 = layer_data2['Negative Activations']
                        total2 = pos2 + neg2
                        
                        if total1 > 0 and total2 > 0:
                            # Proportions
                            prop1 = pos1 / total1
                            prop2 = pos2 / total2
                            
                            # Two-proportion z-test
                            pooled_prop = (pos1 + pos2) / (total1 + total2)
                            se = (pooled_prop * (1 - pooled_prop) * (1/total1 + 1/total2)) ** 0.5
                            
                            if se > 0:
                                z_stat = (prop1 - prop2) / se
                                z_p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                            else:
                                z_stat = 0
                                z_p_value = 1.0
                            
                            # Fisher's exact test
                            contingency_table = [[pos1, neg1], [pos2, neg2]]
                            fisher_result = stats.fisher_exact(contingency_table, alternative='two-sided')
                            fisher_p_value = fisher_result[1]
                            odds_ratio = fisher_result[0]
                            
                            comparison_results.append({
                                'Class': class_name,
                                'Layer': layer,
                                'Artifact 1': artifact1,
                                'Artifact 2': artifact2,
                                'Proportion 1 (P1)': prop1,
                                'Proportion 2 (P2)': prop2,
                                'Difference (P1 - P2)': prop1 - prop2,
                                'Z-statistic': z_stat,
                                'Z-test P-value': z_p_value,
                                'Fisher Odds Ratio': odds_ratio,
                                'Fisher P-value': fisher_p_value,
                                'Significant (α=0.05)': 'Yes' if min(z_p_value, fisher_p_value) < 0.05 else 'No'
                            })
        
        return pd.DataFrame(comparison_results)
    
    def process_single_csv(self, csv_path: str) -> Dict:
        """Process a single CSV file and return results."""
        print(f"Processing: {csv_path}")
        
        # Load and preprocess
        df = self.load_and_preprocess_csv(csv_path)
        
        # Group by class
        grouped = self.group_by_class(df)
        
        class_results = {}
        
        for class_idx, group_df in grouped:
            # Count sensitivity scores
            sensitivity_counts = self.count_sensitivity_scores(group_df)
            
            layer_results = []
            tcav_scores = {}
            
            for layer_name, (pos_count, neg_count) in sensitivity_counts.items():
                # Compute TCAV score
                tcav_score = self.compute_tcav_score(pos_count, neg_count)
                total_count = pos_count + neg_count
                
                # Perform hypothesis test
                hypothesis_results = self.perform_hypothesis_test(tcav_score, total_count)
                
                # Classify
                classification = self.classify_hypothesis(
                    hypothesis_results['p_value_positive'],
                    hypothesis_results['p_value_negative']
                )
                
                # Remove 'sensitivityscore_before' prefix if present
                clean_layer_name = layer_name.replace('sensitivityscore_before_', '').replace('sensitivityscore_before_', '')
                
                layer_results.append({
                    'Layer': clean_layer_name,
                    'Positive Activations': pos_count,
                    'Negative Activations': neg_count,
                    'Proportion of Positive Activations': tcav_score,
                    'P-value (Positive)': hypothesis_results['p_value_positive'],
                    'P-value (Negative)': hypothesis_results['p_value_negative'],
                    'P-value (Two-sided)': hypothesis_results['p_value_two_sided'],
                    'Hypothesis Test Result': classification
                })
                
                tcav_scores[clean_layer_name] = tcav_score
            
            class_results[f'Class {int(class_idx)}'] = {
                'layer_results': pd.DataFrame(layer_results),
                'tcav_scores': tcav_scores
            }
        
        return class_results
    
    def save_results_to_excel(self, class_results: Dict, output_path: str, artifact_name: str = ""):
        """Save results to Excel file with multiple sheets."""
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Save each class results in separate sheets
            for class_name, results in class_results.items():
                results['layer_results'].to_excel(writer, sheet_name=class_name, index=False)
            
            # Compute and save correlations
            #correlations_df = self.compute_correlations(class_results)
            #print(correlations_df)
            #correlations_df.to_excel(writer, sheet_name='Correlations', index=False)
            
            # Format all sheets: left align and autofit columns
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name+'_'+ artifact_name if artifact_name else sheet_name]
                
                # Autofit column widths and left align
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        # Left align all cells
                        cell.alignment = Alignment(horizontal='left', vertical='top')
                        
                        # Calculate max length for autofit
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    
                    # Set column width with some padding
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"Results saved to: {output_path}")
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        # Find all CSV files
        csv_files = self.find_csv_files()
        
        if not csv_files:
            print("No CSV files found!")
            return
        
        # Process each CSV file
        for csv_file in csv_files:
            try:
                class_results = self.process_single_csv(csv_file)
                
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                output_path = os.path.join(
                    os.path.dirname(csv_file),
                    f"{base_name}_tcav_results.xlsx"
                )
                
                # Save results
                self.save_results_to_excel(class_results, output_path)
                
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")


    def consolidate_results(self,artifact_name: str = ""):
        """Consolidate all TCAV results from xlsx files into a single workbook."""
        # Step 1: Find all xlsx files
        xlsx_files = []
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('.xlsx') and 'tcav_results' in file:
                    xlsx_files.append(os.path.join(root, file))
                        
        if not xlsx_files:
            print("No XLSX result files found!")
            return
            
        print(f"Found {len(xlsx_files)} XLSX files to consolidate")
                        
        # Step 2: Create consolidated workbook
        root_dir_name = os.path.basename(os.path.normpath(self.root_directory))
        consolidated_path = os.path.join(self.root_directory, f"{root_dir_name}_consolidated_results.xlsx")
                        
        with pd.ExcelWriter(consolidated_path, engine='openpyxl') as writer:
            # Step 3 & 4: Process each xlsx file
            for xlsx_file in xlsx_files:
                # Get directory name (model name)
                model_dir = os.path.basename(os.path.dirname(xlsx_file))
                # Read the workbook
                xls = pd.ExcelFile(xlsx_file)
                # Copy each sheet with renamed name
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
                    # Rename sheet: model_name_class0 format
                    new_sheet_name = f"{model_dir}_{sheet_name}".replace(" ", "_")
                    new_sheet_name = new_sheet_name + '_' + artifact_name if artifact_name else sheet_name
                    # Excel sheet names have a 31 character limit
                    if len(new_sheet_name) > 31:
                        new_sheet_name = new_sheet_name[:31]
                    df.to_excel(writer, sheet_name=new_sheet_name, index=False)
                    print(f"Copied {sheet_name} from {model_dir} as {new_sheet_name}")
                            
            # Format all sheets: left align and autofit columns
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                    
                # Autofit column widths and left align
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                        
                    for cell in column:
                        # Left align all cells
                        cell.alignment = Alignment(horizontal='left', vertical='top')
                            
                        # Calculate max length for autofit
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    
                    # Set column width with some padding
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
        print(f"Consolidated results saved to: {consolidated_path}")



def main():
    """Main function to run the TCAV analyzer."""
    if len(sys.argv) < 3:
        print("Usage: python script.py <root_directory> <artifact_name>")
        sys.exit(1)
    
    root_directory = sys.argv[1]
    artifact_name = sys.argv[2] 
    
    if not os.path.exists(root_directory):
        print(f"Error: Directory '{root_directory}' does not exist!")
        sys.exit(1)
    
    # Initialize and run analyzer
    analyzer = TCAVAnalyzer(root_directory)
    analyzer.run_analysis()
    analyzer.consolidate_results(artifact_name=artifact_name)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()