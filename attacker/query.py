import time
import requests
from config import API_URL

_session = requests.Session()


def query_victim(images, account_id: str = "attacker_001"):
    labels = []
    probs = []

    for i, img in enumerate(images):
        payload = {
            "account_id": account_id,
            "image": img.tolist()
        }

        response = _session.post(API_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            labels.append(data["pred"])
            probs.append(data["probs"])
        else:
            print(f"  [warning] query {i} failed: {response.status_code}")
            labels.append(-1)
            probs.append([])

        if i > 0 and i % 500 == 0:  # i > 0 guards against sleeping before the first request (0 % 500 == 0 is true). 500-query interval and 0.1s sleep are empirical to avoid overwhelming the FastAPI server.
            time.sleep(0.1)

    return labels, probs
