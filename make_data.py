import json

if __name__ == '__main__':
    with open("data/generated.txt", "r") as f:
        lines = f.readlines()
        all_locations = set()
        sentences = []
        data = []
        for line in lines:
            locations, sentence = line.split(":")
            locations = [s.strip().lower() for s in locations.split(',') if s.strip()]
            all_locations.update(locations)
            sentences.append(sentence.strip())
        for sentence in sentences:
            label = [1 if w and w.lower() in all_locations else 0 for w in sentence.split()]
            assert sum(label) > 0, sentence
            data.append({"sentence": sentence, "label": label})
        with open("data/train_data.json", "w") as f:
            json.dump(data[:-20], f)
        with open("data/test_data.json", "w") as f:
            json.dump(data[-20:], f)

