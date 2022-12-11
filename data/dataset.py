"""
Neural Machine Translation Dataset
"""
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset


class NMTDataset(Dataset):
    def __init__(self, weight: str, phase: str = "train", max_length: int = 64) -> None:
        super().__init__()
        data = load_dataset("msarmi9/korean-english-multitarget-ted-talks-task")
        if phase not in ["train", "validation", "test"]:
            raise ValueError
        self.dataset = data[phase]
        self.tokenizer = AutoTokenizer.from_pretrained(weight)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bos = self.tokenizer.bos_token
        self.eos = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def tokenize(self, text: str) -> dict:
        return self.tokenizer(text, return_tensors="pt", max_length=self.max_length,
                              truncation=True, padding="max_length")

    def __getitem__(self, index) -> tuple:
        data = self.dataset[index]
        kor = data["korean"]
        eng = data["english"]
        kor_text = f"{self.bos}{kor}{self.eos}"
        eng_text = f"{self.bos}{eng}{self.eos}"
        return (self.tokenize(kor_text), self.tokenize(eng_text)["input_ids"])
