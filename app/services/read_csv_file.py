import pandas as pd

# read in csv, remove anime with weird genres, and drop irrelevant columns
def read_csv(path):
    df = pd.read_csv(
        path, dtype={"Episodes": "object", "Ranked": "float64"}, engine="python"
    )
    df.columns = df.columns.str.strip()
    df["Episodes"] = pd.to_numeric(df["Episodes"], errors="coerce")
    df["Episodes"] = df["Episodes"].fillna(0)
    df["synopsis"] = df["synopsis"].fillna("")
    df["Type"] = df["Type"].str.strip().str.lower()

    # drop all anime whose english name is listed as "unknown"
    df = df[df["Name"] != "Unknown"]
    df = df[df["Type"] != "movie"]
    df.reset_index(drop=True, inplace=True)
    
    # No one would want to see these.
    unwanted_genres = ["Ecchi", "Harem", "Hentai"]

    query_str = " & ".join(
        [
            f'~Genres.str.contains("{genre}", case=False, na=False)'
            for genre in unwanted_genres
        ]
    )
    df_filtered = df.query(query_str)

    columns_to_keep = ["English name", "Score", "Genres", "synopsis", "Rating"]

    df_filtered = df_filtered[columns_to_keep]

    return df
