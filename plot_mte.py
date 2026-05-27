import os
import matplotlib.pyplot as plt


def main():
    # Attempt to import pandas and seaborn
    try:
        import pandas as pd
        import seaborn as sns
    except ImportError:
        print("This plotting script requires 'pandas' and 'seaborn' to be installed.")
        print("Please install them in your Python environment by running:")
        print("  pip install pandas seaborn")
        return

    # Attempt to import set_style from your project's visualization module
    try:
        from barriertransformer.visualization import set_style
    except ImportError:
        # Fallback inline implementation of set_style if imports fail
        def set_style():
            my_pal = [
                "#000000",
                "#29AF8C",
                "#97BE49",
                "#3D9CCC",
                "#7C60C6",
                "#D58C2E",
                "#C9492C",
                "#44546A",
            ]
            sns.reset_defaults()
            sns.set_theme(
                context="paper",
                style="ticks",
                palette=my_pal,
                rc={
                    "pdf.fonttype": 42,
                    "svg.fonttype": "none",
                    "figure.facecolor": "white",
                    "figure.dpi": 200,
                    "axes.facecolor": "None",
                    "axes.spines.left": True,
                    "axes.spines.bottom": True,
                    "axes.spines.right": False,
                    "axes.spines.top": False,
                },
            )

    csv_path = os.path.join("results", "mte_raw_data.csv")

    # Check if results folder and raw data CSV exist
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(csv_path):
        # Create a sample template CSV file with multiple trials for demonstration
        print(f"CSV file not found at {csv_path}. Creating a template...")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(
                [
                    "Model",
                    "Experiment",
                    "Prompt_Version",
                    "MTE_Mean",
                    "MTE_Std",
                    "MTE_Mean_Last_10s",
                    "MTE_Std_Last_10s",
                ]
            )
            # Sample trials for Dynamic_Motion
            for trial in range(5):
                writer.writerow(
                    [
                        "llama3.1_8b",
                        "Dynamic_Motion",
                        "v1",
                        str(0.08 + trial * 0.005),
                        "0.05",
                        str(0.07 + trial * 0.004),
                        "0.04",
                    ]
                )
                writer.writerow(
                    [
                        "llama3.1_8b",
                        "Dynamic_Motion",
                        "v1.5",
                        str(0.06 + trial * 0.004),
                        "0.04",
                        str(0.05 + trial * 0.003),
                        "0.03",
                    ]
                )
                writer.writerow(
                    [
                        "llama3.1_8b",
                        "Dynamic_Motion",
                        "v2",
                        str(0.05 + trial * 0.003),
                        "0.03",
                        str(0.04 + trial * 0.002),
                        "0.02",
                    ]
                )
                writer.writerow(
                    [
                        "llama3.1_70b",
                        "Dynamic_Motion",
                        "v1",
                        str(0.07 + trial * 0.005),
                        "0.04",
                        str(0.06 + trial * 0.004),
                        "0.03",
                    ]
                )
                writer.writerow(
                    [
                        "llama3.1_70b",
                        "Dynamic_Motion",
                        "v1.5",
                        str(0.055 + trial * 0.004),
                        "0.03",
                        str(0.048 + trial * 0.003),
                        "0.025",
                    ]
                )
                writer.writerow(
                    [
                        "llama3.1_70b",
                        "Dynamic_Motion",
                        "v2",
                        str(0.045 + trial * 0.003),
                        "0.02",
                        str(0.038 + trial * 0.002),
                        "0.015",
                    ]
                )
            print(f"Created template CSV file at: {csv_path}")
            print(
                "Please open it, replace the sample rows with your actual experimental results, and run this script again."
            )
            return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return

    # Clean and normalize string column content
    for col in ["Model", "Experiment", "Prompt_Version"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Ensure numeric columns are parsed as numbers (coercing strings/empty cells to NaN)
    for col in ["MTE_Mean", "MTE_Std", "MTE_Mean_Last_10s", "MTE_Std_Last_10s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # Get unique models and sort them in the requested custom order from left to right
    custom_model_order = [
        "llama3.1_latest",
        "llama3.1_8b",
        "llama3.1_70b",
        "gemma4_e4b",
        "gemma4_31b",
        "qwen3.5_9b",
        "qwen3.5_27b",
        "qwen3.5_35b"
    ]
    model_order_dict = {model: i for i, model in enumerate(custom_model_order)}
    unique_models = sorted(
        list(df["Model"].unique()),
        key=lambda x: model_order_dict.get(x, 99)
    )

    # Find all unique combinations of Experiment and Prompt_Version in the data
    combinations = df.groupby(["Experiment", "Prompt_Version"]).size().reset_index()

    if len(combinations) == 0:
        print("No data found in CSV.")
        return

    # Check if Last 10s columns are present in CSV
    has_last_10s = "MTE_Mean_Last_10s" in df.columns and "MTE_Std_Last_10s" in df.columns

    print(f"Found {len(combinations)} Experiment + Prompt combinations. Generating separate box plots...")

    for _, row_comb in combinations.iterrows():
        exp = row_comb["Experiment"]
        pv = row_comb["Prompt_Version"]

        # Filter data for this specific experiment and prompt version
        df_sub = df[(df["Experiment"] == exp) & (df["Prompt_Version"] == pv)]

        # Apply style settings from visualization.py
        set_style()

        # Initialize a new figure
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

        # Build boxplot stats manually using MTE_Mean/Std and MTE_Mean_Last_10s/Std_Last_10s
        bxpstats = []
        box_colors = []
        box_alphas = []
        positions_list = []
        
        my_pal = [
            "#29AF8C",
            "#3D9CCC",
            "#7C60C6",
            "#D58C2E",
            "#C9492C",
            "#44546A",
            "#97BE49"
        ]
        model_colors = {model: my_pal[i % len(my_pal)] for i, model in enumerate(unique_models)}

        for m_idx, model in enumerate(unique_models):
            df_sub_model = df_sub[df_sub["Model"] == model]
            if len(df_sub_model) == 0:
                continue

            # --- 1. All Time MTE ---
            mean_val = df_sub_model["MTE_Mean"].mean()
            std_val = df_sub_model["MTE_Std"].mean()
            if pd.isna(std_val):
                std_val = 0.0
            half_std = 0.5 * std_val
            
            bxpstats.append({
                "label": model,
                "med": mean_val,
                "q1": mean_val - half_std,
                "q3": mean_val + half_std,
                "whislo": mean_val - std_val,
                "whishi": mean_val + std_val,
                "fliers": []
            })
            box_colors.append(model_colors[model])
            box_alphas.append(0.85)
            positions_list.append(m_idx - 0.2 if has_last_10s else m_idx)

            # --- 2. Last 10s MTE (if available) ---
            if has_last_10s:
                mean_val_10s = df_sub_model["MTE_Mean_Last_10s"].mean()
                std_val_10s = df_sub_model["MTE_Std_Last_10s"].mean()
                
                # Check if this specific model has data for 10s
                if not pd.isna(mean_val_10s):
                    if pd.isna(std_val_10s):
                        std_val_10s = 0.0
                    half_std_10s = 0.5 * std_val_10s
                    
                    bxpstats.append({
                        "label": f"{model}_10s",
                        "med": mean_val_10s,
                        "q1": mean_val_10s - half_std_10s,
                        "q3": mean_val_10s + half_std_10s,
                        "whislo": mean_val_10s - std_val_10s,
                        "whishi": mean_val_10s + std_val_10s,
                        "fliers": []
                    })
                    box_colors.append(model_colors[model])
                    box_alphas.append(0.40)  # semi-transparent for last 10s
                    positions_list.append(m_idx + 0.2)

        if not bxpstats:
            plt.close(fig)
            continue

        # Draw the boxplot using ax.bxp at explicit positions
        box_width = 0.25 if has_last_10s else 0.4
        artists = ax.bxp(bxpstats, positions=positions_list, showmeans=False, patch_artist=True, widths=box_width)

        # Apply styling, palette colors, and alphas
        for patch, color, alpha in zip(artists["boxes"], box_colors, box_alphas):
            patch.set_facecolor(color)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.0)
            patch.set_alpha(alpha)

        for median in artists["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.5)

        for whisker in artists["whiskers"]:
            whisker.set_color("#444444")
            whisker.set_linewidth(1.0)
            whisker.set_linestyle("--")

        for cap in artists["caps"]:
            cap.set_color("#444444")
            cap.set_linewidth(1.0)

        # Title & Labels
        clean_exp = exp.replace("_", " ")
        ax.set_title(
            fr"{clean_exp} Example (prompt {pv}) - $\overline{{TE}}$",
            fontsize=11,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("LLM Model", fontsize=9, fontweight="bold", labelpad=8)
        ax.set_ylabel(
            r"$\overline{TE}$ (m)", fontsize=9, fontweight="bold", labelpad=8
        )

        # Clean axes lines and grid
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Ensure all models are represented on X-axis, leaving gaps for missing data
        ax.set_xticks(range(len(unique_models)))
        ax.set_xticklabels(unique_models, rotation=45, ha="right")
        ax.set_xlim(-0.6, len(unique_models) - 0.4)

        # Add custom legend to differentiate All Time and Last 10s if both are plotted
        if has_last_10s:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#666666", edgecolor="#333333", alpha=0.85, label=r"All Time $\overline{TE}$"),
                Patch(facecolor="#666666", edgecolor="#333333", alpha=0.40, label=r"Last 10s $\overline{{TE}}$")
            ]
            ax.legend(handles=legend_elements, frameon=True, facecolor="white", edgecolor="none", fontsize=9)

        # Save output for this specific experiment and prompt combination
        safe_exp = exp.lower().replace(" ", "_")
        safe_pv = pv.lower().replace(" ", "_")
        filename = f"mte_boxplot_{safe_exp}_{safe_pv}"
        png_path = os.path.join("results", f"{filename}.png")
        pdf_path = os.path.join("results", f"{filename}.pdf")

        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")

        print(f"Saved: {png_path} and {pdf_path}")
        plt.close(fig)

    print("All box plots generated successfully!")


if __name__ == "__main__":
    main()


