#!/bin/bash
set -e

echo "🔧 Настройка Ollama..."

# Проверяем, установлен ли Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama не установлен!"
    echo "Скачай: https://ollama.com/download"
    exit 1
fi

echo "📥 Скачиваем LLM модель..."
ollama pull llama3.1:8b-instruct-q4_K_M

echo "📥 Скачиваем embedding модель (опционально)..."
ollama pull nomic-embed-text

echo ""
echo "✅ Модели готовы!"
echo ""
echo "Доступные модели:"
ollama list
echo ""
echo "Для проверки: ollama run llama3.1:8b-instruct-q4_K_M 'Привет!'"