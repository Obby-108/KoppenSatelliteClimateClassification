import ee

# initialize
ee.Authenticate()
ee.Initialize(project='grad-ml-project')

# config constants
ASSET_ID = "projects/grad-ml-project/assets/Koppen_Global_1980_2016"
POINTS_PER_CLASS = 50000
OUTPUT_NAME = "primary_koppen_dataset"

def generate_blind_samples():
    print(f"loading validated map: {ASSET_ID}...")
    
    # 1. load image and rename, select only relevant column
    img = ee.Image(ASSET_ID).select([0], ['classification'])
    
    # 2. get the native projection, avoid "missing statistics" errors
    native_proj = img.projection()
    
    # 3. define the region of interest (no antarctica, sentinel doesn't see)
    zones = [
        ee.Geometry.Rectangle([-180, 0, -60, 85], 'EPSG:4326', False), # north america
        ee.Geometry.Rectangle([-60, 0, 60, 85], 'EPSG:4326', False), # europe/africa
        ee.Geometry.Rectangle([60, 0, 180, 85], 'EPSG:4326', False), # asia
        ee.Geometry.Rectangle([-180, -60, -60, 0], 'EPSG:4326', False), # south america
        ee.Geometry.Rectangle([-60, -60, 60, 0], 'EPSG:4326', False), # southern africa
        ee.Geometry.Rectangle([60, -60, 180, 0], 'EPSG:4326', False), # australia/south asia
    ]
    
    print("starting projection-based sampling!!!")
    
    all_samples = []

    # 4. the loop (classes 1-30)
    for class_id in range(1, 31):
        # mask: keep only pixels that equal the current class ID
        ## .selfMask() makes everything else transparent (ignored)
        class_img = img.eq(class_id).selfMask().rename('classification')
        
        # stratification
        for i, zone in enumerate(zones):
            try:
                # use .sample() strictly with projection, NO factor
                points = class_img.sample(
                    region=zone,
                    projection=native_proj, # force native resolution
                    numPixels=POINTS_PER_CLASS,
                    geometries=True, # we need Lat/Lon
                    dropNulls=True
                )

                # check if we found points (takes ~1-2 seconds)
                count = points.size().getInfo()
                
                if count > 0:
                    print(f"yas, class {class_id}: found {count} points in zone {i + 1}")
                    
                    # add a random column, sort by it (shuffling), take top 2000
                    points = points.randomColumn().sort('random').limit(2500)
                
                    # force the class label to be explicit (just in case)
                    points = points.map(lambda f: f.set('classification', class_id))

                    # get final points
                    final_count = points.size().getInfo()
                    print(f"trimmed to {final_count} points")
                    
                    # collect data points
                    all_samples.append(points)
                else:
                    print(f"WARNING!! class {class_id} has no samples in this zone")
                
            except Exception as e:
                print(f"CRITICAL: class {class_id} with error ({e})")

    # 5. merge and export
    if len(all_samples) > 0:
        print("\nmerging collections...")
        flat_collection = ee.FeatureCollection(all_samples).flatten()
        
        total = flat_collection.size().getInfo()
        print(f"WOOP DE WOOPP total dataset size: {total} pointsss")
        
        print("starting export 2 drive...")
        task = ee.batch.Export.table.toDrive(
            collection=flat_collection,
            description=OUTPUT_NAME,
            folder='CS6140_Project_Data',
            fileFormat='CSV',
            selectors=['classification', '.geo']
        )
        task.start()
        print("check progress @ https://code.earthengine.google.com/tasks LOVE")
    else:
        print("no points 💀...")

if __name__ == "__main__":
    generate_blind_samples()