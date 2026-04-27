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

        if i > 0 and i % 500 == 0:
            time.sleep(0.1)

    return labels, probs
