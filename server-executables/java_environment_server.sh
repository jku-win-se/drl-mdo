#!/bin/bash

# Global variables
JAVA_PID=""
PORT=""
if [[ ! -d "$PROJECT_DIR/build/classes" ]]; then
  echo "Error: server not built. Run server-executables/build.sh first." >&2
  exit 1
fi
SERVER_CLASSPATH="$PROJECT_DIR/build/classes:$PROJECT_DIR/build/lib/*"

# Python case names are the folder names; the server knows the sprint planning case as scrum.
java_case() {
  case "$1" in
    sp) echo "scrum" ;;
    sp_uc) echo "scrum-uc" ;;
    *) echo "$1" ;;
  esac
}

# start_server CASE N_ENVS TRAIN_PATH VAL_PATH TEST_PATH OUT_PATH [PORT] [EVAL_K] [ENV_ASSIGNMENT]
#   Validation and test datasets (or NONE) are served on PORT + 1 and PORT + 2.
start_server() {
  local case="$1"
  local n_envs="$2"
  local train_models_path="$3"
  local val_models_path="$4"
  local test_models_path="$5"
  local out_path="$6"
  local port="${7:-8000}"
  local eval_k="${8:-1}"
  local env_assignment="${9:-fixed}"

  # Store port globally for cleanup()
  PORT="$port"

  echo "Run on port $PORT @ start_server"
  # Start server in background, capture PID
  cd "$PROJECT_DIR" || exit
  "$JAVA_BIN" -cp "$SERVER_CLASSPATH" learnapi.MultiModelEnvServer \
    --case "$(java_case "$case")" \
    --n_envs "$n_envs" \
    --models_path "$train_models_path" \
    --val_models_path "$val_models_path" \
    --test_models_path "$test_models_path" \
    --eval_envs "$eval_k" \
    --env_assignment "$env_assignment" \
    --out_path "$out_path" \
    --port "$PORT" \
    > server_output.log 2>&1 &

  JAVA_PID=$!

  echo "Waiting for server to be ready on port $PORT..."
  while ! nc -z localhost "$PORT"; do
    if ! kill -0 "$JAVA_PID" 2>/dev/null; then
      echo "Error: the server exited during startup, see $PROJECT_DIR/server_output.log" >&2
      exit 1
    fi
    sleep 1
  done

  echo "Server is ready on port $PORT."
}

cleanup() {
  echo "Cleaning up..."

  # Kill child processes of JAVA_PID
  if [[ -n "$JAVA_PID" ]] && ps -p "$JAVA_PID" > /dev/null 2>&1; then
    echo "Killing child processes of $JAVA_PID"
    pkill -P "$JAVA_PID" 2>/dev/null
  fi

  # Kill process listening on PORT
  if [[ -n "$PORT" ]]; then
    local PORT_PID
    PORT_PID=$(lsof -ti :"$PORT" -sTCP:LISTEN)

    if [[ -n "$PORT_PID" ]] && ps -p "$PORT_PID" > /dev/null 2>&1; then
      echo "Killing process on port $PORT (PID: $PORT_PID)"
      kill "$PORT_PID" 2>/dev/null
      sleep 5
      if ps -p "$PORT_PID" > /dev/null 2>&1; then
        echo "Force killing process $PORT_PID"
        kill -9 "$PORT_PID"
      fi
    else
      echo "No process on port $PORT"
    fi
  fi

  # Kill main server process
  if [[ -n "$JAVA_PID" ]] && ps -p "$JAVA_PID" > /dev/null 2>&1; then
    echo "Killing server process PID $JAVA_PID"
    kill "$JAVA_PID" 2>/dev/null
    sleep 5
    if ps -p "$JAVA_PID" > /dev/null 2>&1; then
      echo "Force killing server process $JAVA_PID"
      kill -9 "$JAVA_PID"
    fi
  else
    echo "No server process to kill"
  fi

  # Ensure port is free
  if [[ -n "$PORT" ]]; then
    while nc -z localhost "$PORT" 2>/dev/null; do
      sleep 1
    done
  fi

  JAVA_PID=""
  PORT=""
}

trap cleanup EXIT INT TERM

