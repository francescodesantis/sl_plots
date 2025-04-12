from pathlib import Path

from google.cloud import storage

bucket_name = "sim-results-localization"
key_file = Path("upload/secrets/key2.json")
results_dir = Path("mock_results")
client = storage.Client.from_service_account_json(str(key_file))
