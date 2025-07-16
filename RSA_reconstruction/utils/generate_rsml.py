import os
import subprocess


def run_RST_pipeline(input_path, output_path, acq_times,
                     jar_path="/home/loai/Documents/code/RSMLExtraction/RootSystemTracker/target/rootsystemtracker-1.6.1-jar-with-dependencies.jar"):
    """
    Exécute le pipeline Java pour l’extraction RSML.

    Args:
        input_path (str): Chemin vers le dossier d'entrée.
        output_path (str): Chemin vers le dossier de sortie.
        acq_times (str): Temps d’acquisition, séparés par des virgules.
        jar_path (str): Chemin vers le JAR du pipeline Java.

    Returns:
        dict: Contient 'stdout', 'stderr', 'returncode', 'success'.
    """
    cmd = [
        "xvfb-run", "-a", "java", "-cp", jar_path,
        "io.github.rocsg.rootsystemtracker.PipelineActionsHandler",
        f"--input={input_path}",
        f"--output={output_path}",
        f"--acqTimes={acq_times}"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": (result.returncode == 0)
        }
        if result.returncode != 0:
            print("Erreur d'exécution Java !")
        else:
            print("Exécution Java OK !")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return output
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}


def generate_graph_with_java(input_path: str, output_dir: str, acq_times: list,
                             jar_path: str = "/home/loai/Documents/code/RSMLExtraction/RootSystemTracker/target/rootsystemtracker-1.6.1-jar-with-dependencies.jar",
                             expected_filename: str = "61_graph.rsml",
                             timeout: int = 120):
    """
    Exécute le pipeline Java pour reconstruire un graphe et retourne le chemin du fichier généré.
    Fonction prête à être utilisée en parallèle (multiprocessing, joblib...).

    Args:
        input_path (str): Chemin vers le dossier d'entrée.
        output_dir (str): Dossier de sortie.
        acq_times (list): Temps d’acquisition, séparés par des virgules.
        jar_path (str): Chemin vers le JAR.
        expected_filename (str): Nom du fichier à chercher dans output_dir.
        timeout (int): Temps max d’attente en secondes.

    Returns:
        str | None: Chemin complet du fichier généré, ou None si échec.
    """
    cmd = [
        "java", "-Djava.awt.headless=true", "-cp", jar_path,
        "io.github.rocsg.rootsystemtracker.PipelineActionsHandler",
        f"--input={input_path}",
        f"--output={output_dir}",
        f"--acqTimes={','.join(str(x) for x in acq_times)}"

    ]
    try:
        # On ne log pas stdout, juste les vraies erreurs
        result = subprocess.run(cmd, capture_output=False, text=False, timeout=timeout)
    except Exception as e:
        print(f"[ERREUR] Java failed for {input_path} → {e}")
        return None

    # Vérifie la présence du fichier attendu
    expected_path = os.path.join(output_dir, expected_filename)
    if os.path.exists(expected_path):
        return expected_path
    else:
        print(f"[ERREUR] Fichier attendu non trouvé : {expected_path}")
        return None


# Exemple d’utilisation avec multiprocessing :
if __name__ == "__main__":
    from multiprocessing import Pool

    jobs = [
        ("/home/loai/Images/DataTest/UC1_data/230629PN033/", "/home/loai/Documents/code/RSMLExtraction/Test/Output",
         [0.0, 13.6571, 19.6551, 25.6568, 31.6557, 37.6549, 43.6534, 49.6535, 55.6554, 61.6557, 67.6543, 73.6542,
          79.6554, 85.6568, 91.6541, 95.4295, 101.4302, 107.43, 113.4316, 119.4308, 125.4289, 131.4292, 137.4273,
          143.4294, 158.6042, 164.6, 170.5996, 176.5985, 182.599]),
        # etc.
    ]


    def job_wrapper(args):
        return generate_graph_with_java(*args)


    num_cpus = os.cpu_count() or 8  # Nombre de CPU disponibles
    # print(f"Nombre de CPU disponibles : {num_cpus}")

    with Pool(int(num_cpus * 3 / 4)) as pool:
        results = pool.map(job_wrapper, jobs)
    # print(results)
    # write results in temporary file ?

# Exemple d’utilisation :
if __name__ == "__main__0":
    input_path = "/home/loai/Images/DataTest/UC1_data/230629PN033/"
    output_path = "/home/loai/Documents/code/RSMLExtraction/Test/Output"
    acq_times = "0.0,13.6571,19.6551,25.6568,31.6557,37.6549,43.6534,49.6535,55.6554,61.6557,67.6543,73.6542,79.6554,85.6568,91.6541,95.4295,101.4302,107.43,113.4316,119.4308,125.4289,131.4292,137.4273,143.4294,158.6042,164.6,170.5996,176.5985,182.599"

    run_RST_pipeline(input_path, output_path, acq_times)
