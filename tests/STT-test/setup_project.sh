#!/bin/bash

echo "⚙️ Oppretter prosjektstruktur..."

# Root-level directories
mkdir -p app
mkdir -p app/utils
mkdir -p app/pipeline
mkdir -p client/web
mkdir -p client/desktop
mkdir -p models/whisper
mkdir -p models/mistral
mkdir -p scripts
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p docs

# Create basic app files
touch app/main.py
touch app/api.py
touch app/config.py

# Pipeline files
touch app/pipeline/stt.py
touch app/pipeline/summarize.py
touch app/pipeline/pipeline.py

# Utils
touch app/utils/audio_io.py
touch app/utils/file_manager.py
touch app/utils/validators.py

# Client
touch client/web/index.html

# Scripts
touch scripts/install_deps.sh
touch scripts/start.sh
touch scripts/run_stt_test.sh
touch scripts/validate_models.sh

# Tests
touch tests/unit/test_pipeline.py
touch tests/unit/test_stt.py
touch tests/unit/test_summarize.py
touch tests/integration/test_full_flow.py
