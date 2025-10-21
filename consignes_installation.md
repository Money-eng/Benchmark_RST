Créer un environnement virtuel pour Python
----------------------------------------

```bash
python -m venv env
```

Activer l'environnement virtuel
    ----------------------------

- Sur Windows :
```bash
env\Scripts\activate
```
- Sur macOS et Linux :
```bash
source env/bin/activate
```

Installer les dépendances requises
----------------------------    

```bash
pip3 install numpy pandas seaborn matplotlib jupyter ipywidgets tifffile scipy --no-cache
pip3 install git+https://github.com/openalea/mtg --no-cache
pip3 install git+https://github.com/openalea/rsml@hirros --no-cache
``` 

Lancer Jupyter Notebook
----------------------
```bash
jupyter notebook
```

Ouvrir le notebook
----------------------
- Dans l'interface Jupyter, naviguez jusqu'au répertoire contenant le notebook souhaité et cliquez dessus pour l'ouvrir.
- Assurez-vous que le kernel sélectionné est celui de l'environnement virtuel que vous avez créé : "env".

Exécuter les cellules du notebook
----------------------
- Cliquez sur chaque cellule du notebook et appuyez sur "Shift + Enter" pour exécuter le code.
- Suivez les instructions et observez les résultats affichés dans le notebook