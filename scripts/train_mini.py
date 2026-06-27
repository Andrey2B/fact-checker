import sys
sys.path.append(".")
from pathlib import Path
from src.llm.mini_lm.trainer import train
from src.llm.mini_lm.tokenizer import SimpleTokenizer
import json
from pathlib import Path

# Перестраиваем токенизатор под новый большой датасет
tok_path = Path("data/mini_lm/tokenizer.json")
tok_path.unlink(missing_ok=True)  # удаляем старый

train(
    n_epochs   = 15,
    batch_size = 64,    # больше батч для большего датасета
    lr         = 1e-4,  # меньше lr для большего датасета
    max_length = 256,
)