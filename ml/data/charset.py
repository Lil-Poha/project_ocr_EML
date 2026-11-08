import json

BLANK = "<blank>"

class Charset:
    """Алфавит: blank = 0, символы с 1, пробел входит."""

    def __init__(self, chars):
        self.itos = [BLANK] + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts):
        return cls(sorted(set("".join(texts))))

    @classmethod
    def load(cls, path):
        itos = json.load(open(path, encoding="utf-8"))
        assert itos[0] == BLANK, "blank должен быть под номером 0"
        return cls(itos[1:])

    def save(self, path):
        json.dump(self.itos, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    def __len__(self):
        return len(self.itos)          # число классов для выходного слоя модели (с blank)

    def encode(self, text):
        return [self.stoi[c] for c in text]   # KeyError = символа нет в алфавите

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i != 0)

    def decode_ctc(self, ids):
        """Жадное CTC-декодирование: схлопнуть повторы, убрать blank."""
        out, prev = [], None
        for i in ids:
            if i != prev and i != 0:
                out.append(self.itos[i])
            prev = i
        return "".join(out)