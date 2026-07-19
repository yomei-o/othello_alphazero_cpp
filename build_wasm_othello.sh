#!/bin/bash
# Build the WASM Othello UI. NET / SIZE are overridable so a freshly trained net
# can be swapped in without editing anything:  NET=best8.bin SIZE=8 ./build_wasm_othello.sh
set -e; cd "$(dirname "$0")"

# --- emscripten env (portable to this machine's emsdk layout) ---
EMSDK="${EMSDK:-/c/prog/emsdk/emsdk}"
export EM_CONFIG="$EMSDK/.emscripten"
export PATH="$EMSDK/upstream/emscripten:$EMSDK/upstream/bin:$EMSDK/node/22.16.0_64bit/bin:$EMSDK/python/3.13.3_64bit:$PATH"

NET="${NET:-best8.bin}"     # embedded as net.bin
SIZE="${SIZE:-8}"           # board side; MUST match the net
OUT="${OUT:-wasmdist}"; mkdir -p "$OUT"

emcc -O3 -std=c++17 -I. -DBOARD_N="$SIZE" \
  wasm_othello.cpp game.cpp net.cpp mcts.cpp autograd.cpp \
  --embed-file "$NET@net.bin" \
  -sMODULARIZE=1 -sEXPORT_NAME=createOthello -sENVIRONMENT=web -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_FUNCTIONS=_oth_init,_oth_size,_oth_pass,_oth_player,_oth_terminal,_oth_count,_oth_reset,_oth_board,_oth_mask,_oth_is_legal,_oth_apply,_oth_ai_move,_oth_value,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=cwrap,ccall,getValue,HEAP32,HEAPF32,HEAPU8 \
  -o "$OUT/othello.js"
echo "built $OUT/othello.js (+.wasm) with NET=$NET SIZE=$SIZE"
