import pandas as pd, json, glob, os

# config constants
TARGET_PER_CLASS = 2000
OUTPUT_FILENAME = "final_koppen_dataset.csv"

def parse_geo(geo_str):
    """extract Lat/Lon from earth engine's .geo JSON string"""
    try:
        data = json.loads(geo_str)
        # GeoJSON is [Longitude, Latitude]
        return data['coordinates'][0], data['coordinates'][1]
    except:
        return None, None

def main():
    # 1. load all CSV files in the directory
    ## robust path-finding RELATIVE TO FILE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(script_dir, "**", "*.csv")
    ## search and RETRIEVE the files
    csv_files = glob.glob(search_path, recursive=True)
    ## (avoid reading output file)
    csv_files = [f for f in csv_files if "final_koppen_dataset" not in os.path.basename(f)]
    
    if not csv_files:
        print("no CSV files found! download data from drive first")
        return

    print(f"found files: {csv_files}")

    df_list = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"WARNING!! could not read {f} due to error ({e})")

    if not df_list:
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"raw total rows: {len(full_df)}")

    # 2. extract coordinates (crucial for earth engine upload)
    print("parsing coordinates...")
    ## apply the parser to the .geo column
    coords = full_df['.geo'].apply(parse_geo)
    full_df['longitude'] = [c[0] for c in coords]
    full_df['latitude'] = [c[1] for c in coords]
    
    ## drop rows with bad coordinates
    full_df = full_df.dropna(subset=['latitude', 'longitude'])

    # 3. STRICT de-duplication
    ## we round to 5 decimal places (~1 meter) to catch near-duplicates
    full_df['lat_round'] = full_df['latitude'].round(5)
    full_df['lon_round'] = full_df['longitude'].round(5)
    ## print result
    before_dedup = len(full_df)
    full_df = full_df.drop_duplicates(subset=['classification', 'lat_round', 'lon_round'])
    print(f"removed {before_dedup - len(full_df)} duplicates")

    # 4. balancing
    final_dfs = []
    stats = []

    print("\nclass balance REPORT\n")
    print(f"{'class':<6} | {'found':<8} | {'action':<20} | {'final':<8}")
    print("-" * 50)

    ## loop through all classes
    all_classes = sorted(full_df['classification'].unique())
    for class_id in all_classes:
        subset = full_df[full_df['classification'] == class_id]
        count = len(subset)
        if count >= TARGET_PER_CLASS:
            # downsample: randomly pick exactly 2000
            sampled = subset.sample(n=TARGET_PER_CLASS, random_state=42)
            action = "trimmed"
        else:
            # rare class so keep all
            sampled = subset
            action = "kept all"
        final_dfs.append(sampled)
        stats.append({'class': class_id, 'count': len(sampled)})
        print(f"{int(class_id):<6} | {count:<8} | {action:<20} | {len(sampled):<8}")

    # 5. final assembly
    master_df = pd.concat(final_dfs)
    
    # shuffle rows so classes not ordered 1, 1... 2, 2... for NO position memorization
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. save clean CSV for earth engine upload
    ## we only need these 3 columns for the next step
    export_cols = ['classification', 'latitude', 'longitude']
    output_path = os.path.join(script_dir, "class_samples", OUTPUT_FILENAME)
    master_df[export_cols].to_csv(output_path, index=False)

    print("-" * 50)
    print(f"SUCCESS!! saved {len(master_df)} PERFECTLY BALANCED class samples to '{OUTPUT_FILENAME}'")
    print("NEXT STEP: upload this file to earth engine assets")

if __name__ == "__main__":
    main()