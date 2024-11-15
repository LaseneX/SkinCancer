import sys
import gdown
from constants import VERSION_URLS

def download_version(version):
    if version not in VERSION_URLS:
        print(f"Error: Version '{version}' is not available.")
        print("Available versions:", ", ".join(VERSION_URLS.keys()))
        return
    url = VERSION_URLS[version]
    output_filename = f"{version}.h5"

    print(f"Downloading version {version}...")
    gdown.download(url, output_filename)
    print(f"Download complete: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download.py <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    download_version(version)