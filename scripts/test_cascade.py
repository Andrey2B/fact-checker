import sys
sys.path.append(".")

from loguru import logger
from src.verification.cascade_pipeline import CascadePipeline

pipeline = CascadePipeline()

tests = [
    "Эйфелева башня построена в 1889 году в Париже",
    "Эйфелева башня построена в 1950 году в Лондоне",
    "Марс населён синими существами",
    "Альберт Эйнштейн родился в 1879 году в Германии",
    "Москва является столицей России",
]

for text in tests:
    print("\n" + "="*60)
    print(f"Текст: {text}")
    result = pipeline.verify(text)
    print(f"Вердикт:    {result.overall_verdict}")
    print(f"Уверенность: {result.overall_confidence}")
    print(f"Итог:        {result.summary}")