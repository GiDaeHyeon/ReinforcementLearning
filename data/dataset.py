"""
Neural Machine Translation Dataset
"""
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset

SPECIAL_TOKEN = {
    "BOS": "[BOS]",
    "EOS": "[EOS]",
    "SEP": "[SEP]",
    "PAD": "[PAD]"
}


class NMTDataset(Dataset):
    def __init__(self, weight: str, phase: str = "train", max_length: int = 64) -> None:
        super().__init__()
        data = load_dataset("msarmi9/korean-english-multitarget-ted-talks-task")
        if phase not in ["train", "validation", "test"]:
            raise ValueError
        self.dataset = data[phase]
        self.tokenizer = AutoTokenizer.from_pretrained(weight)
        for st in SPECIAL_TOKEN.values():
            self.tokenizer.add_tokens(st, special_tokens=True)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def tokenize(self, text: str) -> dict:
        return self.tokenizer(text, return_tensors="pt", max_length=self.max_length,
                              truncation=True, padding="max_length")

    def __getitem__(self, index: int) -> tuple:
        data = self.dataset[index]
        kor = data["korean"]
        eng = data["english"]
        return (self.tokenize(f"[BOS]{kor}[EOS]")["input_ids"], f"[BOS]{kor}[EOS]"),\
            (self.tokenize(f"[BOS]{eng}[EOS]")["input_ids"], f"{eng}[EOS]")
