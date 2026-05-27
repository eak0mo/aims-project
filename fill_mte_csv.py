import os
import csv
import math

def get_exp_folder(exp_name):
    mapping = {
        "Dynamic_Motion": "dymo",
        "Multiple_Safety_Conditions": "mulsafe",
        "Cluttered_tabletop_custom": "custom_table",
        "Cluttered_tabletop": "table"
    }
    return mapping.get(exp_name)

def get_model_parts(model_name):
    parts = model_name.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return model_name, ""

def main():
    csv_path = os.path.join("results", "mte_raw_data.csv")
    if not os.path.exists(csv_path):
        print("CSV file not found.")
        return

    # Read the current user CSV
    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for r in reader:
            rows.append(r)

    # Add new column names to fieldnames if they aren't there
    new_cols = ["MTE_Mean_Last_10s", "MTE_Std_Last_10s"]
    for col in new_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    updated_count = 0
    missing_count = 0

    for idx, row in enumerate(rows):
        model = row["Model"].strip()
        exp = row["Experiment"].strip()
        pv = row["Prompt_Version"].strip()
        
        # Parse MTE_Mean if it exists
        mte_mean_str = row.get("MTE_Mean")
        mte_mean_val = None
        if mte_mean_str and mte_mean_str.strip():
            try:
                mte_mean_val = float(mte_mean_str)
            except ValueError:
                pass

        exp_folder = get_exp_folder(exp)
        model_parent, model_size = get_model_parts(model)

        if not exp_folder or not model_parent:
            row["MTE_Mean_Last_10s"] = ""
            row["MTE_Std_Last_10s"] = ""
            continue

        # Look in results/new/{exp_folder}/{model_parent}/{model_size}/{pv}/
        target_dir = os.path.join("results", "new", exp_folder, model_parent, model_size, pv)
        
        best_match_10s_mean = ""
        best_match_10s_std = ""
        
        if os.path.exists(target_dir):
            csv_files = [f for f in os.listdir(target_dir) if f.endswith("_results.csv")]
            
            all_runs = []
            for csv_file in csv_files:
                file_path = os.path.join(target_dir, csv_file)
                try:
                    with open(file_path, mode="r", encoding="utf-8") as f:
                        rdr = csv.DictReader(f)
                        for r_row in rdr:
                            mean_key = "MTE_mean_all" if "MTE_mean_all" in r_row else "MTE_mean"
                            std_key = "MTE_std_all" if "MTE_std_all" in r_row else "MTE_std"
                            mean_10s_key = "MTE_mean_last_10s" if "MTE_mean_last_10s" in r_row else None
                            std_10s_key = "MTE_std_last_10s" if "MTE_std_last_10s" in r_row else None

                            if mean_key in r_row:
                                all_runs.append({
                                    "mean": float(r_row[mean_key]) if r_row[mean_key] else 0.0,
                                    "std": float(r_row[std_key]) if r_row[std_key] else 0.0,
                                    "mean_10s": float(r_row[mean_10s_key]) if mean_10s_key and r_row[mean_10s_key] else None,
                                    "std_10s": float(r_row[std_10s_key]) if std_10s_key and r_row[std_10s_key] else None
                                })
                except Exception as e:
                    pass

            if all_runs:
                if mte_mean_val is not None:
                    best_run = min(all_runs, key=lambda run: abs(run["mean"] - mte_mean_val))
                    if abs(best_run["mean"] - mte_mean_val) < 0.05:
                        if best_run["mean_10s"] is not None:
                            best_match_10s_mean = f"{best_run['mean_10s']:.3f}"
                            best_match_10s_std = f"{best_run['std_10s']:.3f}"
                else:
                    first_run = all_runs[0]
                    row["MTE_Mean"] = f"{first_run['mean']:.3f}"
                    row["MTE_Std"] = f"{first_run['std']:.3f}"
                    if first_run["mean_10s"] is not None:
                        best_match_10s_mean = f"{first_run['mean_10s']:.3f}"
                        best_match_10s_std = f"{first_run['std_10s']:.3f}"

        if best_match_10s_mean != "":
            row["MTE_Mean_Last_10s"] = best_match_10s_mean
            row["MTE_Std_Last_10s"] = best_match_10s_std
            updated_count += 1
        else:
            row["MTE_Mean_Last_10s"] = ""
            row["MTE_Std_Last_10s"] = ""
            missing_count += 1

    # Write the updated CSV back
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregation complete. Updated {updated_count} rows. Missing data for {missing_count} rows.")

if __name__ == "__main__":
    main()
