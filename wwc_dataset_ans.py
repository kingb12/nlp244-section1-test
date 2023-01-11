


# ================================================================== #
#                Input pipeline for custom dataset                 #
# ================================================================== #

# You should build your custom dataset as below.
import json
from typing import List, Dict, Union, Set
from typing_extensions import TypedDict

import torch


class RawItem(TypedDict):
    sentence: str
    label: List[int]

def preprocess_sentence(sentence):
  return sentence.lower().split()


class WWCDataset(torch.utils.data.Dataset):

    data: List[RawItem]
    vocabulary: Dict[str, int]

    def __init__(self, data_file: str):
        # Load the data from the target file
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        # TODO: Construct a Vocabulary: a dictionary of all (processed) words + <unk> and <pad> to an integer between
        #  0 and |V|
        all_words: Set[str] = set()
        for item in self.data:
            for word in preprocess_sentence(item['sentence']):
                all_words.add(word)
        all_words.add("<unk>")
        all_words.add("<pad>")
        self.vocabulary = {word: i for i, word in enumerate(all_words)}

    def __getitem__(self, index: int):
        # TODO: Select the item from our set for index
        # TODO: preprocess sentence with the function above
        # TODO: convert our sentence to a tensor of integers using the vocabulary
        # TODO: return a dict {"sentence" : Tensor, "label": Tensor}
        item: RawItem = self.data[index]
        unk_id: int = self.vocabulary["<unk>"]
        processed_sentence: List[int] = [self.vocabulary.get(word, unk_id) for word in
                                         preprocess_sentence(item['sentence'])]
        return {"sentence": torch.tensor(processed_sentence), "label": torch.tensor(item['label'])}


    def __len__(self):
        # TODO: change 0 to the total size of your dataset.
        return len(self.data)

if __name__ == '__main__':
    train_dataset: WWCDataset = WWCDataset("data/train_data.json")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)
    for item in train_loader:
        print(item)