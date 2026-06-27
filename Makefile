.PHONY: setup run test seed clean

# Установка всего окружения
setup:
	pip install -r requirements.txt
	docker-compose up neo4j -d
	@echo "⏳ Ждём запуска Neo4j..."
	sleep 20
	bash scripts/setup_ollama.sh
	python scripts/seed_graph.py
	@echo "✅ Всё готово!"

# Запуск API
run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Тесты
test:
	pytest tests/ -v

# Заполнить граф
seed:
	python scripts/seed_graph.py

# Очистить
clean:
	docker-compose down -v