from datasets import load_dataset

# Load the dataset with streaming to handle large data efficiently
dataset = load_dataset("oscar-corpus/OSCAR-2301",
                       language="en",
                       streaming=True,
                       split="train")

# Initialize variables for data collection and size tracking
collected_data = []
total_size_bytes = 0
max_size_bytes = 1 * 1024 * 1024 * 1024

print("Collecting data...")

for d in dataset:
    # Assume each document's language needs to be checked
    is_eng = True
    sentence_data = d.get("meta", {}).get("sentence_identifications", [])
    for data in sentence_data:
        if data is not None and data["label"] != "en":
            is_eng = False
            break  # Stop checking if a non-English sentence is found

    # Collect data if it's entirely in English
    if is_eng:
        text_bytes = d["text"].encode('utf-8')  # Encode text to bytes to check size accurately
        total_size_bytes += len(text_bytes)
        if total_size_bytes > max_size_bytes:
            break  # Stop if the data size limit is exceeded
        collected_data.append(d["text"])

        print(f"Collected {len(collected_data)} documents, {total_size_bytes / 1024 / 1024 / 1024:.2f} GB", end="\r")

# Convert list of texts to a single string for saving
final_data = "\n".join(collected_data)

# Save the data to a file
with open('data/oscar_input.txt', 'w', encoding='utf-8') as f:
    f.write(final_data)
