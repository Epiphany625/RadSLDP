from openai import OpenAI
import json
import os
import time

# ========== CONFIG ==========
MODEL = "gpt-4.1-mini"   # low cost + very stable for summarization
TEMPERATURE = 0
SLEEP_SEC = 0.3          # avoid sending requests too fast
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are summarizing clusters of radiology reason-for-visit text.

Your goal is to produce a single generalized description that captures
the shared semantic meaning of the cluster.

Rules:
- Be concise (one sentence).
- Use neutral clinical language.
- Do not add new medical information.
- The output should be reusable as a category label.
- Do NOT add implied actions, purposes, or medical workflow context.
"""

def summarize_cluster(representative, samples):
    user_prompt = f"""
Representative:
{representative}

Samples:
""" + "\n".join(f"- {s}" for s in samples)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()


def summarize_all(input_path, output_path=None, max_retries=5):
    """Summarize all clusters in a cluster_details.json file.

    Args:
        input_path: path to cluster_details.json
        output_path: path to write results (default: same dir / cluster_summaries.json)
    """
    from pathlib import Path
    from tqdm import tqdm

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / "cluster_summaries.json"
    else:
        output_path = Path(output_path)

    with open(input_path) as f:
        clusters = json.load(f)

    # Load existing results to support resuming
    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing summaries, resuming...")

    # Skip clusters already done with a successful summary; retry failed ones (summary=None)
    todo = {cid: c for cid, c in clusters.items()
            if cid not in results or results[cid].get("summary") is None}
    print(f"Total clusters: {len(clusters)}, already done: {len(clusters) - len(todo)}, remaining: {len(todo)}")

    for cid, cluster in tqdm(todo.items(), desc="Summarizing clusters"):
        for attempt in range(max_retries):
            try:
                summary = summarize_cluster(
                    cluster["representative"], cluster["samples"]
                )
                results[cid] = {
                    "representative": cluster["representative"],
                    "cluster_size": cluster["cluster_size"],
                    "summary": summary,
                }
                break
            except Exception as e:
                wait = 2 ** attempt * SLEEP_SEC
                print(f"\n  Cluster {cid} attempt {attempt+1} failed: {e}. "
                      f"Retrying in {wait:.1f}s...")
                time.sleep(wait)
        else:
            print(f"\n  Cluster {cid} FAILED after {max_retries} retries, skipping.")
            results[cid] = {
                "representative": cluster["representative"],
                "cluster_size": cluster["cluster_size"],
                "summary": None,
            }

        # Save incrementally so progress is never lost
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(SLEEP_SEC)


    print(f"\nDone. {len(results)} summaries written to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize clusters via LLM")
    parser.add_argument("input", help="Path to cluster_details.json")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: <input_dir>/cluster_summaries.json)")
    args = parser.parse_args()

    summarize_all(args.input, args.output)
