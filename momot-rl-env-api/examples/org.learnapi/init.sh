#!/usr/bin/env bash

initializer_function(){
    if [ -d "$SCRIPT_DIR/models" ]; then
        mkdir -p "$PROJECT_DIR/models"
        cp -r "$SCRIPT_DIR/models/." "$PROJECT_DIR/models/"
    fi
}