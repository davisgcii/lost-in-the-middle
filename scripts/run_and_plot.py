# run_and_plot.py
import subprocess
import matplotlib.pyplot as plt
import time


def main():
    gold_indices = [0, 629, 1254, 1879, 2499]
    num_keys = 2500  # replace with your actual number of keys
    accuracies = []

    for gold_index in gold_indices:
        # Run get_kv_responses_from_gpt4turbo.py
        # subprocess.run(
        #     [
        #         "python",
        #         "-u",
        #         "./scripts/get_kv_responses_from_gpt4turbo.py",
        #         "--input-path",
        #         f"kv_retrieval_data/kv-retrieval-{num_keys}_keys.jsonl.gz",
        #         "--gold-index",
        #         str(gold_index),
        #         "--model",
        #         "gpt-4-1106-preview",
        #         "--output-path",
        #         f"kv_predictions/kv-retrieval-{num_keys}_keys_gold_at_{gold_index}-gpt4turbo-predictions.jsonl.gz",
        #     ]
        # )

        # Run evaluate_kv_responses.py
        result = subprocess.run(
            [
                "python",
                "-u",
                "./scripts/evaluate_kv_responses.py",
                "--input-path",
                f"kv_predictions/kv-retrieval-{num_keys}_keys_gold_at_{gold_index}-gpt4turbo-predictions.jsonl.gz",
                "--output-path",
                f"kv_predictions/kv-retrieval-{num_keys}_keys_gold_at_{gold_index}-gpt4turbo-predictions-scored.jsonl.gz",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        accuracy = float(result.stdout)
        accuracies.append(accuracy)

    # about 51.2 tokens per key-value pair on average over the 128k tokens
    modified_indicies = [i * 51.2 for i in gold_indices]

    plt.figure(figsize=(10, 6))  # adjust as needed
    plt.plot(modified_indicies, accuracies, marker="o")
    plt.xlabel("Token Position")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Key Position (for {num_keys * 51.2} tokens of provided context)")
    plt.ylim(0, 1)  # set y-axis limits
    plt.axhline(0, color="black")  # draw x-axis at y=0
    plt.figtext(
        0.5,
        0.01,
        "This plot cost me $32 :( ",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5},
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("kv_predictions/accuracy_vs_gold_index.png")


if __name__ == "__main__":
    main()
