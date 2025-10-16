#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import shutil
from tqdm import tqdm

PATTERN = re.compile(r"^image_(\d+)_\d{4}\.png$")  # ex: image_123_0000.png

def main(images_root: Path, labels_root: Path, out_root: Path, link: bool, dry_run: bool):
    if not images_root.is_dir():
        raise SystemExit(f"[ERREUR] Dossier images introuvable: {images_root}")
    if not labels_root.is_dir():
        raise SystemExit(f"[ERREUR] Dossier labels introuvable: {labels_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Sous-dossiers numériques dans images_root (1, 2, 3, 10, 11, 100, ...)
    subdirs = sorted([d for d in images_root.iterdir() if d.is_dir() and d.name.isdigit()],
                     key=lambda p: int(p.name))

    missing, created, copied = 0, 0, 0

    for sub in tqdm(subdirs, desc="Traitement des sous-dossiers"):
        n_dir = sub.name  # ex: "10"
        target_dir = out_root / n_dir
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            created += 1

        # Fichiers image_X_0000.png à l’intérieur
        for f in sorted(sub.iterdir()):
            if not f.is_file():
                continue
            m = PATTERN.match(f.name)
            if not m:
                continue
            n = m.group(1)  # "X" dans image_X_0000.png
            label_file = labels_root / f"image_{int(n)}.png"  # normalise 001 -> 1

            if not label_file.exists():
                missing += 1
                print(f"[AVERTISSEMENT] Annotation manquante pour {f.name} -> {label_file}")
                continue

            dest = target_dir / label_file.name
            if dry_run:
                print(f"[DRY-RUN] {'ln -s' if link else 'cp'} {label_file} -> {dest}")
                copied += 1
                continue

            if link:
                # crée un lien symbolique si non existant
                if dest.exists():
                    dest.unlink()
                dest.symlink_to(label_file)
            else:
                shutil.copy2(label_file, dest)
            copied += 1

    print(f"\nRésumé :")
    print(f" - Sous-dossiers traités : {len(subdirs)}")
    print(f" - Fichiers placés       : {copied}")
    print(f" - Annotations manquantes: {missing}")
    if dry_run:
        print("Mode simulation (aucune écriture).")

if __name__ == "__main__":

    main(Path("/home/loai/Images/DataTest/Chronoroot/Compiled/TestSet/images"),
         Path("/home/loai/Images/DataTest/Chronoroot/Compiled/TestSet/labelsTr"),
         Path("/home/loai/Images/DataTest/Chronoroot/Compiled/TestSet/labels"),
         False, False)
