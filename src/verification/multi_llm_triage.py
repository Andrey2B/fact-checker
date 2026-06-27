# src/verification/multi_llm_triage.py
class MultiLLMTriage:
    """
    Опрашиваем несколько моделей Ollama,
    если все согласны — принимаем,
    если нет — углублённая проверка.
    """
    MODELS = ["llama3", "mistral", "gemma2"]

    def triage(self, claim, evidence) -> dict:
        verdicts = {}
        for model in self.MODELS:
            verdict = self._ask_model(model, claim, evidence)
            verdicts[model] = verdict

        # Все согласны?
        unique = set(v["verdict"] for v in verdicts.values())
        if len(unique) == 1:
            return {
                "verdict":    list(unique)[0],
                "confidence": 1.0,
                "method":     "consensus",
            }

        # Расходятся — нужна углублённая проверка
        return {
            "verdict":    "NEEDS_DEEP_VERIFY",
            "confidence": 0.5,
            "method":     "disagreement",
            "details":    verdicts,
        }