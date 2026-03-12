import os
import glob
import pandas as pd

input_data_dir          = os.environ.get('INPUT_DATA_DIR')
cycle_result_dir        = os.environ.get('CYCLE_RESULT_DIR')
cycle_number            = os.environ.get('CYCLE_NUMBER')

prediction_results_dir  = os.path.join(cycle_result_dir, "prediction_results")
master_csv_path         = os.path.join(input_data_dir,   'master', 'master.csv')

def concat_master_csv(prediction_results_dir, cycle_number, master_csv_path):
    master_cycle_csv_path = os.path.join(prediction_results_dir, f"master_cycle_{cycle_number}.csv")
    temp_files_pattern    = os.path.join(prediction_results_dir, f"master_cycle_{cycle_number}.csv.*")

    master_df   = pd.read_csv(master_csv_path)
    temp_files  = glob.glob(temp_files_pattern)
    combined_df = None
    
    for temp_file in sorted(temp_files):
        temp_df = pd.read_csv(temp_file)

        if combined_df is None:
            combined_df = temp_df
        else:
            combined_df = pd.merge(combined_df, temp_df, left_index=True, right_index=True, how='outer')

    combined_df    = pd.concat([master_df[['Sequence']], combined_df], axis=1)
    columns_sorted = ['Sequence'] + sorted([col for col in combined_df.columns if col != 'Sequence'])
    combined_df    = combined_df[columns_sorted]

    combined_df.to_csv(master_cycle_csv_path, index=False)

    print(f"All temporary files have been concatenated and sorted into {master_cycle_csv_path}")


if __name__ == "__main__":
    concat_master_csv(prediction_results_dir, cycle_number, master_csv_path)
