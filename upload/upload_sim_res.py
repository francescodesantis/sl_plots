import os
from pathlib import Path

from upload.config import bucket_name, client, results_dir
from tqdm import tqdm


def upload_to_gcs(local_file: Path):
    """Uploads a file to Google Cloud Storage"""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(local_file)
    blob.upload_from_filename(str(local_file))
    print(f"Uploaded {local_file} to GCS")


def download_from_gcs(blob_name: str, save_dir: Path):
    """Downloads a file from Google Cloud Storage"""
    bucket = client.bucket(bucket_name)
    print(bucket)
    blob = bucket.blob(blob_name)
    print(blob)
    blob.download_to_filename(str(save_dir + Path(blob_name).name))
    print(f"Downloaded {blob_name} to {save_dir}")


def show_blob_info(blob):
    """Prints information about a blob in a single line"""
    print(f"Name: {blob.name}, Size: {blob.size}, Updated: {blob.updated}")

def download_blobs(blob_names:list[str] = None, save_dir=None, start_from = 0):
    """Downloads a list of blobs from Google Cloud Storage"""
    if blob_names is None:
        blob_names = [blob.name for blob in  client.list_blobs(bucket_name)]
        print(f"Downloading {len(blob_names[start_from:])} blobs from {bucket_name}")

    for i, blob_name in enumerate(tqdm(blob_names[start_from:])):
        download_from_gcs(blob_name, save_dir)
        print(f"Downloaded blob {i+1} of {len(blob_names[start_from:])}: {Path(blob_name).name}")
        
def list_blob_names():
    """Lists all the blobs in a bucket"""
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    for i, blob in enumerate(blobs):
        print(f"{i}: ", end="")  # Use end="" to prevent newline
        show_blob_info(blob)

    

def delete_from_gcs(blob_name: str):
    """Deletes a file from Google Cloud Storage"""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()
    print(f"Deleted {blob_name}")

def delete_blobs(blob_names:list[str] = None):
    """Deletes a list of blobs from Google Cloud Storage"""
    if blob_names is None:
        blob_names = [blob.name for blob in  client.list_blobs(bucket_name)]
        print(f"Deleting all blobs from {bucket_name}")

    for blob_name in blob_names:
        delete_from_gcs(blob_name)


def test_uploads():
    for experiment in range(1, 61):
        # Run your simulation (replace this with your actual code)
        print(f"Running experiment {experiment}")
        results_file = results_dir / f"result_{experiment}.txt"
        results_file.write_bytes(os.urandom(2048))  # Example file creation
        print(f"Uploading {results_file} to GCS")
        # Upload results
        upload_to_gcs(results_file)
        print(f"Uploaded {results_file} to GCS")

        # Cleanup local file
        # results_file.unlink()


if __name__ == "__main__":
    test_uploads()
