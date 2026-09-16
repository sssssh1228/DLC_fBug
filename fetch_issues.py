import json
import time
import requests

TOKEN = "****"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "issue-fetcher"
}

SEARCH_URL = "https://api.github.com/search/issues"
ISSUE_URL = "https://api.github.com/search/issues?q=repo:pytorch/pytorch+is:issue+state:closed+created:2025-01-01..2026-01-31+label:%22module:%20dynamo%22"


OUTPUT_FILE = "issues.json"
Num_issue_with_pr = 0

def fetch_linked_prs(issue_number, retry_interval=20, max_retry=None):
    """
    Search PRs related to an issue.

    retry_interval:
        Seconds to wait before retrying after an error.

    max_retry:
        None = retry forever
        int  = maximum retry times
    """

    query = (
        f"repo:pytorch/pytorch "
        f"is:pr "
        f"{issue_number}"
    )

    params = {"q": query,"per_page": 100}
    retry = 0

    while True:
        try:
            resp = requests.get(
                SEARCH_URL,
                headers=HEADERS,
                params=params,
                timeout=30
            )

            resp.raise_for_status()
            data = resp.json()
            linked_prs = []
            for item in data.get("items", []):
                linked_prs.append({
                    "number": item["number"],
                    "title": item["title"],
                    "state": item["state"],
                    "body": item["body"]
                })

            return linked_prs

        except Exception as e:
            retry += 1
            print(f"    PR Search Error: {e}")
            if max_retry is not None and retry > max_retry:
                raise
            print(f"    Retry after {retry_interval} seconds...")
            time.sleep(retry_interval)

# Step 1. Fetch Issues
print("Fetching issues...")
raw_issues = []

for page in range(1, 10):
    params = {"per_page": 100, "page": page}
    resp = requests.get(ISSUE_URL, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items", [])
    raw_issues.extend(items)

print(f"Fetched {len(raw_issues)} issues.")

# Step 2. Process Issues (comments + linked PR)

processed = []
failed_pr_retry = []

for idx, issue in enumerate(raw_issues):
    issue_number = issue["number"]
    print(f"[{idx+1}/{len(raw_issues)}] Issue #{issue_number}")
    
    # comments
    comments = []
    if issue["comments"] > 0:
        try:
            resp = requests.get(
                issue["comments_url"],
                headers=HEADERS
            )
            resp.raise_for_status()

            for c in resp.json():
                comments.append(c["body"])

        except Exception as e:
            print(f"    Comment Error: {e}")


    # labels

    labels = []
    for label in issue["labels"]:
        labels.append({
            "name": label["name"],
            "description": label["description"]
        })

    # linked PR

    linked_prs = fetch_linked_prs(issue_number)
    if len(linked_prs) > 0:
        issue["linked_prs"] = linked_prs
        Num_issue_with_pr += 1
    else:
        issue["linked_prs"] = "None"
    
    print(f"    Found PRs: {len(linked_prs)}")

    # save
    processed.append({
        "number": issue_number,
        "title": issue["title"],
        "state": issue["state"],
        "labels": labels,
        "description": issue["body"],
        "comments": comments,
        "linked_prs": linked_prs
    })
    time.sleep(1)

# Step 4. Save

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)


print(f"Saved {len(processed)} issues to {OUTPUT_FILE}, {Num_issue_with_pr} with related PR")
