#!/bin/bash

echo "=================================="
echo "Starting AI Servers..."
echo "=================================="

# ==================================
# START PHOWHISPER
# ==================================

gnome-terminal -- bash -c "
source /home/amber/thesis/depthai-core/pho_env/bin/activate
python3 /home/amber/thesis/depthai-core/code/pho_server.py
exec bash
"

sleep 5

# ==================================
# START WHISPER
# ==================================

gnome-terminal -- bash -c "
source /home/amber/thesis/depthai-core/pho_env/bin/activate
python3 /home/amber/thesis/depthai-core/code/whisper_server.py
exec bash
"

sleep 5


# ==================================
# START GEMMA SERVER
# ==================================

gnome-terminal -- bash -c "
cd /home/amber//gemma/depthai-core/code/llama.cpp

./build/bin/llama-server \
--jinja \
-m /home/amber/gemma/depthai-core/code/models/gemma4_currency_Q4.gguf \
--mmproj /home/amber/gemma/depthai-core/code/models/mmproj-F16.gguf \
-ngl 999 \
-c 512

exec bash
"



echo ""
echo "=================================="
echo "All AI servers started!"
echo "Now run your Python program."
echo "=================================="

