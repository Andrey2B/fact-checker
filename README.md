# Система верификации фактов на основе GraphRAG

## Описание
Автоматическая верификация текстовых утверждений 
с использованием графа знаний Wikidata и LLM Llama 3.

## Архитектура
- ClaimDecomposer — декомпозиция через LLM
- EvidenceMatcher — поиск в Neo4j
- VerdictGenerator — вердикт через LLM
- MiniVerifier — быстрый фильтр (~5M параметров)

## Результаты
| Модель       | Accuracy | Macro F1 | Время |
|--------------|----------|----------|-------|
| Ollama 8B    | 78.5%    | 79.1%    | 9.07с |
| MiniVerifier | 48.1%    | 41.6%    | 0.01с |
| Каскад       | 78.5%    | 79.1%    | 2.80с |

## Запуск
```bash
docker-compose up -d    # Neo4j
ollama serve            # LLM
python main.py          # API
'''

## Основной пайплайн

```mermaid
flowchart TD
    A([Входной текст]) --> B[ClaimDecomposer\nLLM Llama 3.1 8B]
    B --> C[Атомарные триплеты\nsubject · predicate · object]
    C --> D[EvidenceMatcher\nNeo4j Cypher]
    D --> E[(Граф знаний\nNeo4j\n~1200 узлов)]
    E --> D
    D --> F[Доказательства\nmax 7 фактов]
    F --> G[VerdictGenerator\nLLM Llama 3.1 8B]
    G --> H[Вердикт по триплету\nSUPPORTED · REFUTED\nNEI · CONFLICTING]
    H --> I{Ещё\nтриплеты?}
    I -->|Да| D
    I -->|Нет| J[Агрегация\nвердиктов]
    J --> K([VerificationReport\noverall_verdict\nconfidence · summary])
```

## Каскадный пайплайн

```mermaid
flowchart TD
    A([Входной текст]) --> B{Субъект\nв графе?}

    B -->|ДА| C[VerificationPipeline\nосновной пайплайн\n~9 с]
    B -->|НЕТ| D[MiniVerifier\n~5M параметров\n~0.01 с]

    D --> E{confidence\n> 0.90?}

    E -->|ДА\nNEI| F([NOT_ENOUGH_INFO\nБыстрый ответ\n0.01 с])
    E -->|НЕТ| C

    C --> G([VerificationReport\nSUPPORTED · REFUTED · NEI\n~9 с])

    style F fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
    style D fill:#bbf,stroke:#333
    style C fill:#fdb,stroke:#333
```