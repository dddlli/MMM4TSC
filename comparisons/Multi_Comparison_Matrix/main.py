import pandas as pd

from MCM import MCM

if __name__ == "__main__":
    path_res = "lts.csv"
    output_dir = "/"

    df_results = pd.read_csv(path_res)

    MCM.compare(
        df_results=df_results,
        pdf_savename="heatmap",
        png_savename="heatmap",
    )

    MCM.compare(
        df_results=df_results,
        excluded_col_comparates=["clf1", "clf3"],
        png_savename="heatline_vertical",
        include_ProbaWinTieLoss=True
    )

    MCM.compare(
        df_results=df_results,
        excluded_row_comparates=["clf1", "clf3"],
        png_savename="heatline_horizontal",
        tex_savename="heatline_horizontal",
    )
