# ================================================================== #
#                Input pipeline for custom dataset                 #
# ================================================================== #

# You should build your custom dataset as below.
import torch


def preprocess_sentence(sentence):
  return sentence.lower().split()


class WWCDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO: Load the data from the target file
        # TODO: Construct a Vocabulary: a dictionary of all (processed) words to an integer between 0 and |V|
        # TODO: Add tokens <unk> and <pad> to this vocabulary
        pass

    def __getitem__(self, index: int):
        # TODO: Select the item from our set for index
        # TODO: preprocess sentence with the function above
        # TODO: convert our sentence to a tensor of integers using the vocabulary
        # TODO: return a dict {"sentence" : Tensor, "label": Tensor}
        pass

    def __len__(self):
        # TODO: change 0 to the total size of your dataset.
        return 0

if __name__ == '__main__':
    train_dataset: WWCDataset = WWCDataset("data/train_data.json")
    print(train_dataset[11])