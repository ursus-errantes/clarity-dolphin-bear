# Search for a flac file by file name in a directory and play the flac file
import os
import sys
import subprocess

def find_flac_file(filename: str, directory: str) -> str:
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return ""

def play_flac_file(filepath: str):
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", filepath])
    except Exception as e:
        print(f"Error playing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python find_and_play.py <filename> <directory>")
        sys.exit(1)

    filename = sys.argv[1]
    directory = sys.argv[2]
    filepath = find_flac_file(filename, directory)

    if filepath:
        print(f"Playing file: {filepath}")
        play_flac_file(filepath)
    else:
        print(f"File '{filename}' not found.")