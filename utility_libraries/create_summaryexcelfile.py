import pandas as pd
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <excel_file>")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    
    # Read CSPI sheets
    cspi_0 = pd.read_excel(excel_file, sheet_name='Class_0_cspi')
    cspi_1 = pd.read_excel(excel_file, sheet_name='Class_1_cspi')
    cspi_2 = pd.read_excel(excel_file, sheet_name='Class_2_cspi')
    
    # Read Delta sheets
    delta_0 = pd.read_excel(excel_file, sheet_name='Class_0_delta')
    delta_1 = pd.read_excel(excel_file, sheet_name='Class_1_delta')
    delta_2 = pd.read_excel(excel_file, sheet_name='Class_2_delta')
    
    # Read Moving Average sheets
    ma_0 = pd.read_excel(excel_file, sheet_name='Class_0_moving_avg')
    ma_1 = pd.read_excel(excel_file, sheet_name='Class_1_moving_avg')
    ma_2 = pd.read_excel(excel_file, sheet_name='Class_2_moving_avg')
    
    # Rename unnamed rows with Layer_Idx
    for df in [cspi_0, cspi_1, cspi_2]:
        df.index = df.index.fillna("Layer_Idx")

    # Drop rows 2 and 3 from moving average dataframes
    print(ma_0.loc[[0, 1]])
    print(ma_1.loc[[0, 1]])
    print(ma_2.loc[[0, 1]])
    
    # Drop rows 2 and 3 from moving average dataframes
    ma_0 = ma_0.drop([0, 1], errors='ignore')
    ma_1 = ma_1.drop([0, 1], errors='ignore')
    ma_2 = ma_2.drop([0, 1], errors='ignore')
    
    # Create summary for CSPI
    cspi_summary = pd.concat(
        [cspi_0.add_suffix('_Class_0'), 
         cspi_1.add_suffix('_Class_1'), 
         cspi_2.add_suffix('_Class_2')],
        axis=1
    )
    
    # Create summary for Delta
    delta_summary = pd.concat(
        [delta_0.add_suffix('_Class_0'), 
         delta_1.add_suffix('_Class_1'), 
         delta_2.add_suffix('_Class_2')],
        axis=1
    )
    
    # Create summary for Moving Average
    ma_summary = pd.concat(
        [ma_0.add_suffix('_Class_0'), 
         ma_1.add_suffix('_Class_1'), 
         ma_2.add_suffix('_Class_2')],
        axis=1
    )
    
    # Write summary sheets to new Excel file
    output_file = excel_file.replace('.xlsx', '_summary.xlsx')
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        cspi_summary.to_excel(writer, sheet_name='CSPI_Summary', index=False)
        delta_summary.to_excel(writer, sheet_name='Delta_Summary', index=False)
        ma_summary.to_excel(writer, sheet_name='MovingAvg_Summary', index=False)
    
    print(f"Summary file created: {output_file}")

if __name__ == "__main__":
    main()
