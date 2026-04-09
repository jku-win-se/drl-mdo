#!/bin/bash

# Global variables
JAVA_PID=""
PORT=""
RUN_SERVER_SET_CMD="$JAVA_BIN -jar $PROJECT_DIR/rl-environment-server.jar"
RUN_SERVER_SET_INFERENCE_CMD="$JAVA_BIN -jar $PROJECT_DIR/rl-environment-server-inference.jar"

start_server_trainingset() {
  local case="$1"
  local n_envs="$2"
  local train_models_path="$3"
  local test_models_path="$4"
  local out_path="$5"
  local port="${6:-8000}"

  # Store port globally for cleanup()
  PORT="$port"

  if [[ -z "$out_path" ]]; then
      out_path="$instance"
  else
      out_path="$instance-$out_path"
  fi


	echo "Run on port $PORT @ start_server"
  # Start server in background, capture PID
  cd "$PROJECT_DIR" || exit
  $RUN_SERVER_SET_CMD \
    --case "$case" \
    --n_envs "$n_envs" \
    --models_path "$train_models_path" \
    --test_models_path "$test_models_path" \
    --out_path "$out_path" \
    --port "$PORT" \
    > server_output_trainingset.log 2>&1 &

  JAVA_PID=$!

  echo "Waiting for server to be ready on port $PORT..."
  while ! nc -z localhost "$PORT"; do
      echo "Try connecting..."
      sleep 1
  done

  echo "Server is ready on port $PORT."
}

start_server() {
  local case="$1"
  local n_steps="$2"
  local n_envs="$3"
  local instance="${4:-3}"
  local out_path="$5"
  local port="${6:-8000}"

  # Store port globally for cleanup()
  PORT="$port"

  if [[ -z "$out_path" ]]; then
      out_path="$instance"
  else
      out_path="$instance-$out_path"
  fi


	echo "Run on port $PORT @ start_server"
  # Start server in background, capture PID
  cd "$PROJECT_DIR" || exit
  $RUN_SERVER_SET_INFERENCE_CMD \
    --case "$case" \
    --n_steps "$n_steps" \
    --n_envs "$n_envs" \
    --instance "$instance" \
    --out_path "$out_path" \
    --port "$PORT" \
    > server_output.log 2>&1 &

  JAVA_PID=$!

  echo "Waiting for server to be ready on port $PORT..."
  while ! nc -z localhost "$PORT"; do
      echo "Try connecting..."
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

