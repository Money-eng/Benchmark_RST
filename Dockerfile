#############################################
# 1) Builder Java + native-image (GraalVM) #
#############################################
FROM ghcr.io/graalvm/graalvm-community:24.0.1-ol8-20250415 AS java-builder

# Installer native-image
RUN gu install native-image

# Copier et packager avec Maven (Shade pour jar "uber")
WORKDIR /app/RootSystemTracker
COPY RootSystemTracker/pom.xml .
COPY RootSystemTracker/src ./src

RUN mvn clean package -DskipTests \ 
 && native-image \
      -cp target/*-jar-with-dependencies.jar \
      -H:Name=rsml-tracker \
      io.github.rocsg.gui.RSMLNoGUI

##################################
# 2) Image finale Python + mamba #
##################################
FROM mambaorg/micromamba:latest AS python-builder

# 2a) Créer l'env Python via micromamba
#    - placez votre environment.yml à la racine du contexte Docker
COPY RSA_deep_working/environment.yml .
RUN micromamba create -n rsa-env -f environment.yml \
 && micromamba clean --all --yes

# 2b) Ajouter le binaire Java natif
COPY --from=java-builder /app/RootSystemTracker/rsml-tracker /usr/local/bin/rsml-tracker

# 2c) Copier le code Python
COPY RSA_deep_working ./RSA_deep_working
COPY CreateRSADataset ./CreateRSADataset

# 2d) Activer l'env mamba par défaut
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/rsa-env/bin:$PATH

# Entrypoint qui lance votre pipeline
WORKDIR /app
ENTRYPOINT ["python", "RSA_deep_working/main.py"]
CMD ["--config", "RSA_deep_working/config.yml"]


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