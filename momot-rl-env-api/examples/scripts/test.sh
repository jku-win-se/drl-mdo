#!/bin/bash

# Global variable for the server process ID
JAVA_PID=""

start_server() {
  local case="$1"
  local n_envs="$2"
  local instance="${3:-3}"
  local out_path="${4:-20000}"
  
  # Start server in background, capture PID
  sudo $RUN_SERVER_CMD --case "$case" --n_envs "$n_envs" --instance "$instance" --out_path "$out_path" > /dev/null 2>&1 &
  JAVA_PID=$!

  echo "Waiting for server to be ready on port 8000..."
  while ! nc -z localhost 8000; do
      sleep 0.5
  done

  echo "Server is ready."
}

cleanup() {
  echo "Cleaning up..."

    pkill -P $JAVA_PID  # Kill child processes
    kill $JAVA_PID 2>/dev/null
    wait $JAVA_PID 2>/dev/null
  # Kill process on port 8000
  local PORT_PID
  PORT_PID=$(sudo lsof -ti :8000 -sTCP:LISTEN)
  if [ -n "$PORT_PID" ]; then
    echo "Killing process on port 8000 (PID: $PORT_PID)"
    sudo kill "$PORT_PID"
    sleep 3
    if ps -p "$PORT_PID" > /dev/null 2>&1; then
      echo "Force killing process $PORT_PID"
      sudo kill -9 "$PORT_PID"
    fi
  else
    echo "No process on port 8000"
  fi

  # Kill JAVA_PID if still alive
  if [ -n "$JAVA_PID" ] && ps -p "$JAVA_PID" > /dev/null 2>&1; then
    echo "Killing server process PID $JAVA_PID"
    sudo kill "$JAVA_PID"
    sleep 3
    if ps -p "$JAVA_PID" > /dev/null 2>&1; then
      echo "Force killing server process $JAVA_PID"
      sudo kill -9 "$JAVA_PID"
    fi
  else
    echo "No server process to kill"
  fi
  
      # Ensure port is free
    while nc -z localhost 8000; do
        sleep 0.5
    done

  JAVA_PID=""
}

trap cleanup EXIT INT TERM



