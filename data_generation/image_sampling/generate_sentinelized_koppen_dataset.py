import ee

# initialize
ee.Authenticate()
ee.Initialize(project='grad-ml-project')

# config
ASSET_PATH = "projects/grad-ml-project/assets/final_koppen_dataset"
OUTPUT_BASE_NAME = "koppen_shard"
NUM_SHARDS = 1500  # split 60,000 points into 300 files (200 points each)
PATCH_SIZE = 128 # 128 pixels * 10m = 1.28km patch

# 13 spectral bands
## B1 (aerosol), B2-B4 (RGB), B5-B7 (red edge), B8 (NIR), B8A (narrow NIR)
## B9 (water vapor), B10 (cirrus clouds [imperfect data!]), B11-B12 (SWIR)
ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

def run_patch_exporter():
    print(f"loading points from: {ASSET_PATH}...")

    # 1. add random column (0 to 1) to "scatter" points into batches
    ## ensures every batch gets a mix of climates
    table = ee.FeatureCollection(ASSET_PATH)
    table = table.randomColumn(seed=42)

    # 2. define image collection (Sentinel-2 Harmonized)
    ## filter for low cloud cover (<10%) to get clean ground data
    ## median pixel = cloud-free composite
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterDate('2023-01-01', '2023-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .median() \
        .select(ALL_BANDS)

    # 3. define the neighborhood (the "camera lens")
    ## this creates a square kernel of 128 x 128 pixels
    kernel = ee.Kernel.square(radius=PATCH_SIZE/2, units='pixels')

    # 4. convert image to arrays
    ## 'neighborhoodToArray' turns each pixel into a list of its neighbors.
    ### result: image where every pixel contains a 128x128x13 data cube
    patches = s2.neighborhoodToArray(kernel)

    for i in range(NUM_SHARDS):
        ## calculate the slice: e.g., batch 0 is 0.00 to 0.033
        lower = i / NUM_SHARDS
        upper = (i + 1) / NUM_SHARDS
        
        ## filter table to just this 3% slice of data
        shard_table = table.filter(ee.Filter.And(
            ee.Filter.gte('random', lower),
            ee.Filter.lt('random', upper)
        ))

        # 5. sample at previously found points
        print("sampling regions (this happens on the server)...")
        samples = patches.sampleRegions(
            collection=shard_table,
            scale=10,# force 10m resolution (upscales 20m/60m bands automatically)
            projection='EPSG:3857', # web mercator (standard for ML)
            tileScale=8, # prevents "user memory limit exceeded"
            geometries=True # keep lat/lon in the export
        )

        # 6. export to ggl drive in TFRecord format
        ## TFRecord is the only format that handles 3D array data efficiently
        task_name = f"{OUTPUT_BASE_NAME}_part_{i+1}_of_{NUM_SHARDS}"
        task = ee.batch.Export.table.toDrive(
            collection=samples,
            description=task_name,
            folder='CS6140_Project_Data',
            fileFormat='TFRecord', 
            selectors=ALL_BANDS + ['classification', '.geo'] 
        )
        task.start()
        print(f"shard {i + 1} submitted: {task_name}")
    
    print("-" * 50)
    print("THIRTY exports started!")
    print("check progress at: https://code.earthengine.google.com/tasks")
    print("output will be a set of .tfrecord files in my drive folder")

if __name__ == "__main__":
    run_patch_exporter()