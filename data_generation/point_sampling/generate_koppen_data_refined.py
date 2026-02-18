import ee, sys, time

# initialize
ee.Authenticate()
ee.Initialize(project='grad-ml-project')

# config constants
ASSET_ID = "projects/grad-ml-project/assets/Koppen_Global_1980_2016"
TARGET_PIXEL_CLASSES = {
    # 1000000 : [23, 15, 25, 12, 30, 19], ## covered in run 1
    # 2000000 : [21, 9] -> run 3, # ...8, 22] -> covered in run 2 
    # 4000000 : [17, 24, 16], ## ...28 -> covered in run 5, ...18] -> covered in run 4
    # 6000000 : [20, 13] ## rest completed on run 6
    # 25000000: [10, 13] ## need extremely high amount -> completed on run 7
}
SESSION_ID = int(time.time()) # timestamp for unique filenames

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
    
    print("starting DYNAMIC projection-based sampling!!! now unsure about zero samples")

    # 4. the loop (targeting rare classes)
    for POINTS_PER_CLASS, classes in TARGET_PIXEL_CLASSES.items():
        print(f"NOW COUNTING {POINTS_PER_CLASS} PIXELS...")
        for class_id in classes:
            print(f"now searching over class {class_id}!")
            
            class_samples = [] # store points for THIS class only

            # mask: keep only pixels that equal the current class ID
            ## .selfMask() makes everything else transparent (ignored)
            class_img = img.eq(class_id).selfMask().rename('classification')
            
            # stratification
            for i, zone in enumerate(zones):
                # VERBOSE LOGGING: print BEFORE network call
                print(f"querying zone {i+1}/6...", end=" ") 
                sys.stdout.flush() # force print to screen immediately

                try:
                    # use .sample() strictly with projection, NO factor
                    points = class_img.sample(
                        region=zone,
                        projection=native_proj, # force native resolution
                        numPixels=POINTS_PER_CLASS,
                        geometries=True, # we need Lat/Lon
                        dropNulls=True
                    )

                    print(f"OK, finished sampling points in zone {i + 1} for class {class_id}")
                    
                    # add a random column, sort by it (shuffling), take top 2000
                    points = points.randomColumn().sort('random').limit(2500)
                
                    # force the class label to be explicit (just in case)
                    points = points.map(lambda f: f.set('classification', class_id))
                    
                    # collect data points
                    class_samples.append(points)
                    
                except Exception as e:
                    print(f"NO, class {class_id} encountered error ({e})")

            # 5. merge and export
            if len(class_samples) > 0:
                print("\nmerging collections...")
                flat_class_collection = ee.FeatureCollection(class_samples).flatten()
                
                print(f"WOOP DE WOOPP, points found for class {class_id}")
                
                fname = f"secondary_koppen__{class_id}_session_{SESSION_ID}_data"

                print("starting export 2 drive...")
                task = ee.batch.Export.table.toDrive(
                    collection=flat_class_collection,
                    description=fname,
                    folder='CS6140_Project_Data',
                    fileFormat='CSV',
                    selectors=['classification', '.geo']
                )
                task.start()
                print("check progress @ https://code.earthengine.google.com/tasks LOVE")
            else:
                print("no points 💀.. nothing to save!")

if __name__ == "__main__":
    generate_blind_samples()