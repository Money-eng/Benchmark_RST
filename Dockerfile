# Stage 1: build Java native binary with Native Image Community Edition
FROM ghcr.io/graalvm/native-image-community:24.0.1-ol8-20250415 AS java-builder

# Passer en root pour installer git et maven
USER root
RUN microdnf update -y \
    && microdnf install -y git maven \
    && microdnf clean all

# Copier et compiler le projet RootSystemTracker
WORKDIR /build
COPY RootSystemTracker /build/RootSystemTracker
WORKDIR /build/RootSystemTracker
RUN mvn clean package -DskipTests

# Générer l'exécutable natif à partir du JAR
RUN native-image -jar target/rootsystemtracker-*-jar-with-dependencies.jar \
    --no-server \
    -H:Name=rootsystemtracker

# Stage 2: Python environment avec micromamba
FROM mambaorg/micromamba:latest AS python-builder

# Passer en root pour installer les certificats
USER root
RUN apt-get update \
    && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ajouter le certificat Fortinet et mettre à jour le store
COPY Fortinet_CA_SSL.crt /usr/local/share/ca-certificates/Fortinet_CA_SSL.crt
RUN update-ca-certificates

# Utiliser bash pour micromamba
SHELL ["bash", "-lc"]

# Créer et activer un environnement Python 3.12
RUN micromamba create -y -n py312 python=3.12 -c conda-forge \
    && micromamba clean --all -f -y
ENV PATH=/opt/conda/envs/py312/bin:$PATH

# Copier et installer les dépendances Python
WORKDIR /app
COPY py312.yaml /app/
RUN micromamba install -y -n py312 -f py312.yaml \
    && micromamba clean --all -f -y

# Copier et installer le module RSML en mode editable
COPY external/rsml /build/rsml
WORKDIR /build/rsml
# Installer git et définir une version fallback pour setuptools-scm
USER root
RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*
# Bypass setuptools-scm missing git repository by fixant la version
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0

# Installer en editable
RUN micromamba run -n py312 pip install -e .

# Copier le code Python du projet
COPY RSA_deep_working /app/

# Copier l'exécutable natif Java construit précédemment
COPY --from=java-builder /build/RootSystemTracker/rootsystemtracker /usr/local/bin/rootsystemtracker
RUN chmod +x /usr/local/bin/rootsystemtracker

# Exposer le port TensorBoard (optionnel)
EXPOSE 8080

WORKDIR /app/RSA_deep_working

# Execute python script main.py
CMD ["python3", "/app/RSA_deep_working/Models/main.py"]
