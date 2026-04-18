import requests
from config import API_URL


def query_victim(images, account_id: str = "attacker_001"):
    """
    Send images to victim API one by one, collect predicted labels.
    
    Args:
        images: np.ndarray of shape (N, 28, 28) float32
        account_id: simulated attacker account
    
    Returns:
        labels: list of int (victim's predicted class for each image)
        probs:  list of list (victim's predicted probabilities)
    """
    labels = []
    probs  = []

    for i, img in enumerate(images):
        payload = {
            "account_id": account_id,
            "image": img.tolist()
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            labels.append(data["pred"])
            probs.append(data["probs"])
        else:
            print(f"  [warning] query {i} failed: {response.status_code}")
            labels.append(-1)
            probs.append([])

    return labels, probs