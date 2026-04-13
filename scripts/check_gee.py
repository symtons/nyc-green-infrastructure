"""Minimal Google Earth Engine authentication and connectivity check.

Run this before attempting any data collection. If this fails, fix GEE
before doing anything else.
"""
import sys


def main():
    print("=" * 60)
    print("GOOGLE EARTH ENGINE — CONNECTIVITY CHECK")
    print("=" * 60)

    # 1. Can we import the library?
    try:
        import ee
        print("✓ earthengine-api imported")
    except ImportError as e:
        print(f"✗ Cannot import earthengine-api: {e}")
        print("  Run: pip install earthengine-api")
        sys.exit(1)

    # 2. Can we initialize? This is where auth happens.
    PROJECT_ID = "pivotal-equinox-445505-b3"
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"✓ Earth Engine initialized with project: {PROJECT_ID}")
    except Exception as e:
        print(f"✗ Initialize failed: {e}")
        print()
        print("  If this says 'Please authorize access', run:")
        print("    python -c \"import ee; ee.Authenticate()\"")
        print("  then re-run this script.")
        print()
        print("  If the project ID is wrong or expired, go to")
        print("    https://console.cloud.google.com/")
        print("  and check that the project still exists.")
        sys.exit(1)

    # 3. Can we actually do a trivial query?
    try:
        nyc_point = ee.Geometry.Point([-74.0, 40.7])
        collection = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(nyc_point)
            .filterDate("2024-06-01", "2024-08-31")
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
        )
        count = collection.size().getInfo()
        print(f"✓ Landsat 9 query returned {count} scenes over NYC (summer 2024)")
    except Exception as e:
        print(f"✗ Landsat query failed: {e}")
        sys.exit(1)

    # 4. WorldCover check
    try:
        wc = ee.ImageCollection("ESA/WorldCover/v200").first()
        _ = wc.bandNames().getInfo()
        print("✓ ESA WorldCover v200 accessible")
    except Exception as e:
        print(f"✗ WorldCover access failed: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED — GEE is ready for Step 3b")
    print("=" * 60)


if __name__ == "__main__":
    main()