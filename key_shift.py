import csv
import shutil
import subprocess
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

NOTE_TO_I = {
    "C":0, "C#":1, "Db":1,
    "D":2, "D#":3, "Eb":3,
    "E":4,
    "F":5,
    "F#":6, "Gb":6,
    "G":7,
    "G#":8, "Ab":8,
    "A":9,
    "A#":10, "Bb":10,
    "B":11,
}
I_TO_NOTE_FLAT = {0:"C",1:"Db",2:"D",3:"Eb",4:"E",5:"F",6:"Gb",7:"G",8:"Ab",9:"A",10:"Bb",11:"B"}

def parse_key(label: str):
    tonic, mode = label.strip().split()
    mode = mode.lower()
    if tonic not in NOTE_TO_I or mode not in ("major", "minor"):
        raise ValueError(f"Unrecognized key label: {label!r}")
    return tonic, mode

def transpose_key(label: str, semitones: int) -> str:
    tonic, mode = parse_key(label)
    new_i = (NOTE_TO_I[tonic] + semitones) % 12
    return f"{I_TO_NOTE_FLAT[new_i]} {mode}"

def sox_pitch(in_path: Path, out_path: Path, semitones: int):
    cents = semitones * 100
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["C:\Program Files (x86)\sox-14-4-2\sox.exe", str(in_path), str(out_path), "pitch", str(cents)]
    subprocess.run(cmd, check=True)

def build_key_map(key_root: Path) -> dict[str, str]:
    """
    Returns: { basename_without_ext : "Db major" }
    Example: ".../10089.LOFI.key" -> key "10089.LOFI"
    """
    key_map = {}
    key_files = list(key_root.rglob("*.key")) + list(key_root.rglob("*.KEY"))
    for kf in key_files:
        stem = kf.stem  # basename without ".key"
        label = kf.read_text(encoding="utf-8", errors="ignore").strip()
        key_map[stem] = label
    return key_map

def augment_dataset(audio_root: Path, key_root: Path, output_root: Path, semitone_shifts, copy_original=True):
    audio_root = audio_root.resolve()
    key_root = key_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    key_map = build_key_map(key_root)

    audio_files = [p for p in audio_root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

    matched = []
    missing = []
    for af in audio_files:
        if af.stem in key_map:
            matched.append(af)
        else:
            missing.append(af)

    print(f"[scan] audio_root:  {audio_root}")
    print(f"[scan] key_root:    {key_root}")
    print(f"[scan] output_root: {output_root}")
    print(f"[scan] audio files found: {len(audio_files)}")
    print(f"[scan] key files found:   {len(key_map)}")
    print(f"[scan] matched pairs:     {len(matched)}")
    print(f"[scan] missing labels:    {len(missing)}")
    if missing[:5]:
        print("[example missing]", [p.name for p in missing[:5]])
    if matched[:5]:
        print("[example matched]", [p.name for p in matched[:5]])

    rows = []

    for audio_path in matched:
        base_label = key_map[audio_path.stem]

        rel = audio_path.relative_to(audio_root)
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if copy_original:
            out_audio = out_dir / audio_path.name
            out_key = out_audio.with_suffix(".key")
            shutil.copy2(audio_path, out_audio)
            out_key.write_text(base_label + "\n", encoding="utf-8")
            rows.append((str(out_audio), base_label))

        for k in semitone_shifts:
            if k == 0:
                continue
            out_audio = out_dir / f"{audio_path.stem}_ps{('+' if k>0 else '')}{k}{audio_path.suffix}"
            out_key = out_audio.with_suffix(".key")

            new_label = transpose_key(base_label, k)
            sox_pitch(audio_path, out_audio, k)
            out_key.write_text(new_label + "\n", encoding="utf-8")
            rows.append((str(out_audio), new_label))

    out_csv = output_root / "augmented_index.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "key_label"])
        w.writerows(rows)

    print(f"[done] Wrote {len(rows)} entries to {out_csv}")

if __name__ == "__main__":
    # CHANGE THESE THREE:
    audio_dir  = Path(r"./giantsteps-key-dataset/audio")
    key_dir    = Path(r"./giantsteps-key-dataset/annotations/key")
    out_dir    = Path(r"./augmented-data")

    shifts = [-3, -2, -1, 1, 2, 3]
    augment_dataset(audio_dir, key_dir, out_dir, shifts, copy_original=True)