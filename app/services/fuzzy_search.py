from fuzzywuzzy import process
import numpy as np

def fuzzy_match_anime(df, query):
    # Get top 10 matches for both 'Name' and 'English name' columns
    matches_name = process.extract(query, df["Name"], limit=10)
    matches_english_name = process.extract(query, df["English name"], limit=10)

    # Combine the results and remove duplicates
    matches = {match[0]: match[1] for match in matches_name + matches_english_name}

    # Sort the matches by score in descending order
    sorted_matches = sorted(matches.items(), key=lambda item: item[1], reverse=True)

    # Extract the corresponding rows from the dataframe and make a copy
    suggestions = df[
        df["Name"].isin([match[0] for match in sorted_matches])
        | df["English name"].isin([match[0] for match in sorted_matches])
    ].copy()

    # Clean the suggestions DataFrame to handle any NaN or infinite values
    suggestions.replace([np.inf, -np.inf], np.nan, inplace=True)
    suggestions.dropna(inplace=True)

    return suggestions.iloc[:10].to_dict(orient="records")
