"""
BPE-подобный токенизатор на символьном уровне с словарём по книге Raschka.
"""
import re
import json
from pathlib import Path
from collections import Counter
from loguru import logger


class SimpleTokenizer:
    """
    Простой word-piece токенизатор.
    Строим словарь из обучающих данных.
    """

    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<SEP>": 4,   # разделитель claim | evidence
        "<SUP>": 5,   # SUPPORTED
        "<REF>": 6,   # REFUTED
        "<NEI>": 7,   # NOT_ENOUGH_INFO
    }

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size   = vocab_size
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

    def build_vocab(self, texts: list[str]):
        """Строим словарь из списка текстов."""
        logger.info(f"Строим словарь из {len(texts)} текстов...")

        # Считаем частоту слов
        counter = Counter()
        for text in texts:
            tokens = self._basic_tokenize(text)
            counter.update(tokens)

        # Начинаем со специальных токенов
        self.token2id = dict(self.SPECIAL_TOKENS)

        # Добавляем самые частые слова
        n_special = len(self.SPECIAL_TOKENS)
        n_vocab   = min(self.vocab_size - n_special, len(counter))

        for word, _ in counter.most_common(n_vocab):
            if word not in self.token2id:
                self.token2id[word] = len(self.token2id)

        # Обратный словарь
        self.id2token = {v: k for k, v in self.token2id.items()}

        logger.info(f"Словарь: {len(self.token2id)} токенов")

    def _basic_tokenize(self, text: str) -> list[str]:
        """Простая токенизация: по пробелам и пунктуации."""
        text   = text.lower()
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens

    def encode(
        self,
        claim: str,
        evidence: str,
        max_length: int = 512,
    ) -> tuple[list[int], list[int]]:
        """
        Кодирует пару (claim, evidence) в token ids.
        Формат: <BOS> claim_tokens <SEP> evidence_tokens <EOS>
        """
        claim_tokens    = self._basic_tokenize(claim)
        evidence_tokens = self._basic_tokenize(evidence)

        ids = [self.SPECIAL_TOKENS["<BOS>"]]

        for tok in claim_tokens:
            ids.append(
                self.token2id.get(tok, self.SPECIAL_TOKENS["<UNK>"])
            )

        ids.append(self.SPECIAL_TOKENS["<SEP>"])

        for tok in evidence_tokens:
            ids.append(
                self.token2id.get(tok, self.SPECIAL_TOKENS["<UNK>"])
            )

        ids.append(self.SPECIAL_TOKENS["<EOS>"])

        # Truncate
        if len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.SPECIAL_TOKENS["<EOS>"]]

        # Attention mask
        mask = [1] * len(ids)

        # Pad
        pad_len = max_length - len(ids)
        ids  += [self.SPECIAL_TOKENS["<PAD>"]] * pad_len
        mask += [0] * pad_len

        return ids, mask

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "token2id":   self.token2id,
            }, f, ensure_ascii=False, indent=2)
        logger.success(f"Токенизатор сохранён: {path}")

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.token2id = data["token2id"]
        tok.id2token = {v: k for k, v in tok.token2id.items()}
        logger.info(f"Токенизатор загружен: {len(tok.token2id)} токенов")
        return tok