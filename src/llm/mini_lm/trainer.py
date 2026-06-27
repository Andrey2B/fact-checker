"""
Обучение MiniVerifier на данных из нашего графа.
"""
import sys
import json
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from loguru import logger

sys.path.append(".")
from src.llm.mini_lm.model     import MiniVerifier
from src.llm.mini_lm.tokenizer import SimpleTokenizer

# Пути
MODEL_DIR  = Path("data/mini_lm")
MODEL_PATH = MODEL_DIR / "model.pt"
TOK_PATH   = MODEL_DIR / "tokenizer.json"
TRAIN_PATH = Path("data/mini_lm_train.json")

LABEL2ID = {
    "SUPPORTED":        0,
    "REFUTED":          1,
    "NOT_ENOUGH_INFO":  2,
}


class VerificationDataset(Dataset):
    """
    Датасет для обучения верификатора.
    Каждый пример: (claim, evidence, label)
    """
    def __init__(
        self,
        samples:    list[dict],
        tokenizer:  SimpleTokenizer,
        max_length: int = 512,
    ):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample   = self.samples[idx]
        claim    = sample["claim"]
        evidence = sample["evidence"]
        label    = LABEL2ID[sample["label"]]

        ids, mask = self.tokenizer.encode(claim, evidence, self.max_length)

        return {
            "input_ids":      torch.tensor(ids,  dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def generate_training_data(output_path: Path, n_samples: int = 2000):
    """
    Генерируем обучающие данные из нашего графа.
    Используем EvidenceMatcher чтобы получить реальные доказательства.
    """
    from src.graph.neo4j_client import Neo4jClient
    from src.verification.models import AtomicClaim

    logger.info("Генерируем обучающие данные из графа...")
    client = Neo4jClient.get_instance()

    # Получаем все факты
    facts = client.run_query("""
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        RETURN s.name AS subject, r.type AS predicate, o.name AS object
    """)

    # Все значения по предикату для генерации REFUTED
    values_by_pred: dict[str, list] = {}
    for f in facts:
        p = f["predicate"]
        if p not in values_by_pred:
            values_by_pred[p] = []
        if f["object"] not in values_by_pred[p]:
            values_by_pred[p].append(f["object"])

    samples = []

    for fact in facts:
        subj = fact["subject"]
        pred = fact["predicate"]
        obj  = fact["object"]

        # Доказательства — все факты о субъекте
        evidences = client.run_query("""
            MATCH (s:Entity {name: $name})-[r:RELATION]->(o:Entity)
            RETURN s.name AS subject, r.type AS predicate, o.name AS object
            LIMIT 5
        """, {"name": subj})

        if not evidences:
            continue

        evidence_text = " | ".join(
            f"{e['subject']} {e['predicate']} {e['object']}"
            for e in evidences
        )

        # SUPPORTED
        claim_text = f"{subj} {pred} {obj}"
        samples.append({
            "claim":    claim_text,
            "evidence": evidence_text,
            "label":    "SUPPORTED",
        })

        # REFUTED — подменяем объект
        candidates = [
            v for v in values_by_pred.get(pred, [])
            if v != obj
        ]
        if candidates:
            wrong_obj   = random.choice(candidates)
            wrong_claim = f"{subj} {pred} {wrong_obj}"
            samples.append({
                "claim":    wrong_claim,
                "evidence": evidence_text,
                "label":    "REFUTED",
            })

        # NOT_ENOUGH_INFO — факты о другой сущности как доказательство
        other_facts = [
            f for f in facts
            if f["subject"] != subj
        ]
        if other_facts:
            other = random.choice(other_facts)
            other_evidence = (
                f"{other['subject']} {other['predicate']} {other['object']}"
            )
            samples.append({
                "claim":    claim_text,
                "evidence": other_evidence,
                "label":    "NOT_ENOUGH_INFO",
            })

    random.shuffle(samples)
    samples = samples[:n_samples]

    # Баланс классов
    by_label: dict[str, list] = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    logger.info("Распределение классов:")
    for label, items in by_label.items():
        logger.info(f"  {label}: {len(items)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.success(f"Обучающие данные: {output_path} ({len(samples)} примеров)")
    return samples


def train(
    n_epochs:    int   = 20,
    batch_size:  int   = 32,
    lr:          float = 3e-4,
    max_length:  int   = 256,
    val_split:   float = 0.15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Устройство: {device}")

    # ── Данные ──────────────────────────────────────────────────
    if not TRAIN_PATH.exists():
        samples = generate_training_data(TRAIN_PATH, n_samples=3000)
    else:
        with open(TRAIN_PATH, encoding="utf-8") as f:
            samples = json.load(f)
        logger.info(f"Загружено {len(samples)} примеров")

    # ── Токенизатор ─────────────────────────────────────────────
    if TOK_PATH.exists():
        tokenizer = SimpleTokenizer.load(TOK_PATH)
    else:
        tokenizer = SimpleTokenizer(vocab_size=8192)
        texts = [s["claim"] + " " + s["evidence"] for s in samples]
        tokenizer.build_vocab(texts)
        tokenizer.save(TOK_PATH)

    # ── Train / Val split ────────────────────────────────────────
    random.shuffle(samples)
    n_val    = int(len(samples) * val_split)
    val_data = samples[:n_val]
    trn_data = samples[n_val:]

    train_ds = VerificationDataset(trn_data, tokenizer, max_length)
    val_ds   = VerificationDataset(val_data, tokenizer, max_length)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=0
    )
    val_dl   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0
    )

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Модель ──────────────────────────────────────────────────
    model = MiniVerifier().to(device)
    logger.info(f"Параметров: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # ── Обучение ─────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            loss   = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        # Validation
        model.eval()
        val_loss    = 0.0
        correct     = 0
        total       = 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["label"].to(device)

                logits  = model(input_ids, attn_mask)
                loss    = criterion(logits, labels)
                val_loss += loss.item()

                preds   = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_loss /= len(val_dl)
        val_acc   = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{n_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.1%}"
        )

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "config":      MiniVerifier.DEFAULT_CONFIG,
                "val_acc":     val_acc,
                "epoch":       epoch,
            }, MODEL_PATH)
            logger.success(f"  Новый лучший: val_acc={val_acc:.1%} → {MODEL_PATH}")

    # Сохраняем историю
    with open(MODEL_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.success(f"Обучение завершено. Лучший val_acc: {best_val_acc:.1%}")
    return history


if __name__ == "__main__":
    import torch.nn as nn
    train()