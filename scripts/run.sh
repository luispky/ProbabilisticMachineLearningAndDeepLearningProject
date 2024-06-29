#!/bin/bash 

# Parse options
while getopts "ab" opt; do
  case $opt in
    a)  
      python run_diffusion.py
      ;;
    b)
      python3 run_diffusion.py 
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done