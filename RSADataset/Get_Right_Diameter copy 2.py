import os
import tifffile
import numpy as np
import rsml
from rsml import hirros
import utils.CustomDumper as CD
from itertools import combinations

# Classes et fonctions pour la gestion des systèmes racinaires
class RootSystem:
    def __init__(self, folder_path: str, load_date_map: bool = False):
        """
        Initialise le système racinaire en chargeant le MTG issu du RSML,
        l'image stack (série temporelle d'images) et, optionnellement, le dateMap.
        
        :param folder_path: Chemin vers le dossier contenant les fichiers.
        :param load_date_map: Si True, charge également le dateMap (default: False).
        """
        self.folder_path = folder_path
        self.load_date_map = load_date_map

        # Attributs qui seront chargés
        self.image_stack = None
        self.date_map = None
        self.mtg = None
        self.obs_times = None
        self.geometry = None
        self.time_hours = None

        # Lancement du chargement des données
        self._load_data()

    def _load_data(self):
        # Chargement de l'image stack
        image_stack_file = os.path.join(self.folder_path, "22_registered_stack.tif")
        if os.path.exists(image_stack_file):
            self.image_stack = tifffile.imread(image_stack_file)
        else:
            raise FileNotFoundError(f"Fichier image stack non trouvé: {image_stack_file}")

        # Chargement optionnel du dateMap
        if self.load_date_map:
            date_map_file = os.path.join(self.folder_path, "40_date_map.tif")
            if os.path.exists(date_map_file):
                self.date_map = tifffile.imread(date_map_file)
            else:
                print(f"Avertissement: dateMap non trouvé dans {self.folder_path}")

        # Chargement du RSML et conversion en MTG
        expertized_rsml_file = os.path.join(self.folder_path, "61_graph_expertized.rsml")
        default_rsml_file = os.path.join(self.folder_path, "61_graph.rsml")
        if os.path.exists(expertized_rsml_file):
            self.mtg = rsml.rsml2mtg(expertized_rsml_file)
        elif os.path.exists(default_rsml_file):
            self.mtg = rsml.rsml2mtg(default_rsml_file)
        else:
            raise FileNotFoundError(f"Aucun fichier RSML trouvé dans {self.folder_path}")

        # Extraction des propriétés du MTG (temps d'observation, géométrie, temps par vertex)
        if self.mtg is not None:
            self.obs_times = hirros.times(self.mtg)
            self.geometry = self.mtg.property('geometry')
            self.time_hours = self.mtg.property('time_hours')
            
            # set metadata image['name'] of the mtg to the image name 
            meta_name = self.mtg.graph_properties().get('metadata', None).get('image', None).get('name', None)
            #print(self.mtg.graph_properties().get('metadata', None).get('image', None))
            #print(meta_name)
            #print(self.mtg.graph_properties())
            #print(self.mtg.properties())
            if meta_name is None:
                image_meta = self.mtg.graph_properties().get('metadata', None).get('image', None)
                image_meta['name'] = os.path.basename(image_stack_file)
                self.mtg.graph_properties().get('metadata', None)['image'] = image_meta
                #print(self.mtg.graph_properties().get('metadata', None).get('image', None))
            # if "functions" not in metadata in graph properties, copy time, time_hours into it
            if 'functions' not in self.mtg.graph_properties().get('metadata', None):
                self.mtg.graph_properties().get('metadata', None)['functions'] = {}
                self.mtg.graph_properties().get('metadata', None)['functions']['time'] = self.mtg.properties()['time']
                self.mtg.graph_properties().get('metadata', None)['functions']['time_hours'] = self.mtg.properties()['time_hours']
                #print(self.mtg.graph_properties().get('metadata', None).get('functions', None))
    
    def visualize(self, show_date_map: bool = False):
        """
        Lance une visualisation Napari affichant l'image stack en fond et 
        la superposition des données MTG (nœuds et arêtes) en fonction du temps.
        
        Si show_date_map est True et que self.date_map est chargé, 
        ouvre une deuxième fenêtre affichant le date_map (2D) et le graph correspondant 
        au dernier temps d'observation.
        """
        import napari
        import numpy as np

        # Fonctions locales pour le calcul de la couleur et la troncature d'une polyline
        def get_gradient_color(f,
                            start_color=np.array([1.0, 1.0, 0.9]),  # blanc-jaune clair
                            end_color=np.array([0.5, 0.0, 0.5])):   # violet
            color = (1 - f) * start_color + f * end_color
            return np.concatenate([color, [1.0]])

        def get_truncated_polyline(polyline, t_vector, current_time):
            """
            Tronque la polyline pour ne conserver que les points dont le temps est inférieur ou égal à current_time.
            Inverse x et y pour l'affichage.
            """
            polyline = np.array(polyline)
            polyline = polyline[:, ::-1]  # inversion x et y
            t_vector = np.array(t_vector)
            valid = t_vector <= current_time
            if not np.any(valid):
                return None
            last_idx = np.where(valid)[0][-1]
            return polyline[:last_idx + 1]

        # Vérification des données nécessaires
        if self.image_stack is None or self.mtg is None or self.obs_times is None:
            raise ValueError("Les données nécessaires (image, MTG, temps) ne sont pas chargées.")

        # Calcul des couleurs par vertex en fonction du temps de naissance
        birth_times = [times[0] for times in self.time_hours.values()]
        min_birth = min(birth_times)
        max_birth = max(birth_times) if max(birth_times) > min_birth else min_birth + 1e-6
        vertex_colors = {}
        for vid, times in self.time_hours.items():
            f = (times[0] - min_birth) / (max_birth - min_birth)
            vertex_colors[vid] = get_gradient_color(f)

        # Création du viewer principal avec l'image stack et le graph interactif
        viewer = napari.Viewer()
        current_time = self.obs_times[0]
        initial_edges = []
        initial_nodes = []
        initial_nodes_colors = []

        for vid, polyline in self.geometry.items():
            t_vector = self.time_hours[vid]
            truncated = get_truncated_polyline(polyline, t_vector, current_time)
            if truncated is not None and len(truncated) > 1:
                initial_edges.append(truncated)
                for pt in truncated:
                    initial_nodes.append(pt)
                    initial_nodes_colors.append(vertex_colors[vid])
        initial_nodes = np.array(initial_nodes)
        initial_nodes_colors = np.array(initial_nodes_colors)

        # Ajout de l'image stack en fond
        viewer.add_image(
            self.image_stack,
            name="Série temporelle",
            colormap="gray",
            contrast_limits=[self.image_stack.min(), self.image_stack.max()]
        )

        # Ajout des arêtes et des nœuds du MTG
        edges_layer = viewer.add_shapes(
            initial_edges,
            shape_type='path',
            edge_color=[vertex_colors[vid] for vid, poly in self.geometry.items() 
                        if get_truncated_polyline(poly, self.time_hours[vid], current_time) is not None],
            face_color='transparent',
            name='Edges MTG'
        )
        nodes_layer = viewer.add_points(
            initial_nodes,
            face_color=initial_nodes_colors,
            size=3,
            name='Nodes MTG'
        )

        # Mise à jour interactive des couches lors du changement de frame
        @viewer.dims.events.current_step.connect
        def update_layers(event):
            time_idx = event.value[0]
            current_time = self.obs_times[time_idx] if time_idx < len(self.obs_times) else self.obs_times[-1]

            new_edges = []
            new_edge_colors = []
            new_nodes = []
            new_nodes_colors = []

            for vid, polyline in self.geometry.items():
                t_vector = self.time_hours[vid]
                truncated = get_truncated_polyline(polyline, t_vector, current_time)
                if truncated is not None and len(truncated) > 1:
                    new_edges.append(truncated)
                    new_edge_colors.append(vertex_colors[vid])
                    for pt in truncated:
                        new_nodes.append(pt)
                        new_nodes_colors.append(vertex_colors[vid])
            nodes_array = np.array(new_nodes) if new_nodes else np.empty((0, 2))
            nodes_layer.data = nodes_array
            if new_nodes_colors:
                nodes_layer.face_color = np.array(new_nodes_colors)
            edges_layer.data = new_edges
            if new_edge_colors:
                edges_layer.edge_color = new_edge_colors

        # Si l'option show_date_map est activée, création d'une seconde fenêtre Napari
        if show_date_map:
            if self.date_map is None:
                print("Avertissement: date_map n'est pas chargé. La vue séparée ne peut pas être affichée.")
            else:
                viewer2 = napari.Viewer(title="Graph sur Date Map (dernier temps)")
                viewer2.add_image(
                    self.date_map,
                    name="Date Map",
                    colormap="gray",
                    contrast_limits=[self.date_map.min(), self.date_map.max()]
                )
                final_time = self.obs_times[-1]
                final_edges = []
                final_nodes = []
                final_nodes_colors = []
                final_edge_colors = []
                for vid, polyline in self.geometry.items():
                    t_vector = self.time_hours[vid]
                    truncated = get_truncated_polyline(polyline, t_vector, final_time)
                    if truncated is not None and len(truncated) > 1:
                        final_edges.append(truncated)
                        final_edge_colors.append(vertex_colors[vid])
                        for pt in truncated:
                            final_nodes.append(pt)
                            final_nodes_colors.append(vertex_colors[vid])
                final_nodes = np.array(final_nodes)
                final_nodes_colors = np.array(final_nodes_colors)
                viewer2.add_shapes(
                    final_edges,
                    shape_type='path',
                    edge_color=final_edge_colors,
                    face_color='transparent',
                    name='Edges MTG'
                )
                viewer2.add_points(
                    final_nodes,
                    face_color=final_nodes_colors,
                    size=3,
                    name='Nodes MTG'
                )

        napari.run()

    def save2folder(self, destination_folder: str):
            """
            Sauvegarde l'image stack et le fichier RSML d'origine dans le dossier destination_folder.
            Si le dossier n'existe pas, il sera créé.
            
            :param destination_folder: Chemin vers le dossier de destination.
            """
            if self.image_stack is None or self.mtg is None:
                raise ValueError("Les données nécessaires à la sauvegarde ne sont pas chargées.")

            # Création du dossier de destination s'il n'existe pas
            os.makedirs(destination_folder, exist_ok=True)

            # Sauvegarde de l'image stack
            image_filename = "22_registered_stack.tif"
            image_save_path = os.path.join(destination_folder, image_filename)
            tifffile.imwrite(image_save_path, self.image_stack)
            print(f"Image stack sauvegardée dans : {image_save_path}")

            # Sauvegarde du fichier RSML depuis le mtg
            rsml_filename = "61_graph.rsml"
            rsml_save_path = os.path.join(destination_folder, rsml_filename)
            mtg2rsml(self.mtg, rsml_save_path)
            print(f"Fichier RSML sauvegardé dans : {rsml_save_path}")

    def save_rsml(self, output_filename: str):
     
        import xml.etree.ElementTree as ET
        import xml.dom.minidom
        from datetime import datetime
        # Définition de la racine RSML avec le namespace requis
        ns = {"po": "http://www.plantontology.org/xml-dtd/po.dtd"}
        rsml_elem = ET.Element("rsml", attrib={"xmlns:po": ns["po"]})

        # Construction de la section metadata
        metadata_elem = ET.SubElement(rsml_elem, "metadata")
        ET.SubElement(metadata_elem, "version").text = "1.4"
        ET.SubElement(metadata_elem, "unit").text = "pixel(um)"
        ET.SubElement(metadata_elem, "size").text = "76.0"
        ET.SubElement(metadata_elem, "last-modified").text = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        ET.SubElement(metadata_elem, "software").text = "RootSystemTracker"
        ET.SubElement(metadata_elem, "user").text = "Unknown"

        # On utilise ici une propriété 'file-key' présente dans le MTG ou un défaut
        file_key = self.mtg.graph_properties().get("file-key", "default_file_key")
        ET.SubElement(metadata_elem, "file-key").text = str(file_key)

        # Observation-hours : on récupère la liste des temps (par exemple via self.obs_times)
        if self.obs_times is not None:
            obs_hours_str = ",".join(str(t) for t in self.obs_times)
        else:
            obs_hours_str = ""
        ET.SubElement(metadata_elem, "observation-hours").text = obs_hours_str

        # Section image (ici, on utilise le file_key comme label par défaut)
        image_elem = ET.SubElement(metadata_elem, "image")
        image_label = self.mtg.graph_properties().get("image-label", file_key)
        ET.SubElement(image_elem, "label").text = str(image_label)
        ET.SubElement(image_elem, "sha256").text = self.mtg.graph_properties().get("image-sha256", "Nothing there")

        # Construction de la section scene
        scene_elem = ET.SubElement(rsml_elem, "scene")
        plant_elem = ET.SubElement(scene_elem, "plant", attrib={"ID": "1", "label": ""})

        # Fonction récursive pour ajouter les branches (racines) à l'élément parent
        def add_root_branch(parent_xml_elem, vertex_id, id_str):
            """
            Ajoute une balise <root> pour le vertex donné, avec ses données de géométrie
            et, le cas échéant, ses branches enfants.
            """
            # Crée l'élément root avec l'ID et un label vide
            root_elem = ET.SubElement(parent_xml_elem, "root", attrib={"ID": id_str, "label": ""})
            # Ajout de la géométrie (polyline) si présente
            geometry_data = self.geometry.get(vertex_id, None)
            if geometry_data is not None:
                geometry_elem = ET.SubElement(root_elem, "geometry")
                polyline_elem = ET.SubElement(geometry_elem, "polyline")
                # On parcourt la liste des points de la polyline.
                # Ici, on suppose que chaque point est une séquence (x, y)
                for pt in geometry_data:
                    ET.SubElement(polyline_elem, "point", attrib={
                        "coord_t": "0.0",    # valeur par défaut, à ajuster si vous disposez d'informations temporelles
                        "coord_th": "0.0",
                        "coord_x": str(pt[0]),
                        "coord_y": str(pt[1]),
                        "diameter": "0.0",
                        "vx": "0.0",
                        "vy": "0.0"
                    })
            # Récupération des branches enfants (si la propriété 'children' existe dans le MTG)
            children = self.mtg.property('children').get(vertex_id, []) if self.mtg.property('children') is not None else []
            child_count = 1
            for child_id in children:
                new_id_str = id_str + "." + str(child_count)
                add_root_branch(root_elem, child_id, new_id_str)
                child_count += 1

        # Identification du (ou des) vertex racine(s) du MTG.
        # Ici on suppose que self.mtg.roots() existe (sinon on utilise les clés de self.geometry)
        if hasattr(self.mtg, "roots"):
            main_roots = self.mtg.roots()
        else:
            main_roots = list(self.geometry.keys())
        if main_roots:
            # On prend le premier comme racine principale du plant
            add_root_branch(plant_elem, main_roots[0], "1.1")

        # Conversion de l'arbre XML en une chaîne avec une mise en forme "pretty print"
        rough_string = ET.tostring(rsml_elem, encoding="utf-8")
        reparsed = xml.dom.minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="    ", encoding="UTF-8")
        with open(output_filename, "wb") as f:
            f.write(pretty_xml)
        print(f"RSML sauvegardé dans : {output_filename}")

    def estimate_diameter_from_date_map(self):
        """
        Estime le diamètre des racines à partir du dateMap.
        Hypothèses : 
            - Tout ce qui n'est pas de la racine est noir.
            - Les racines forment des objets connexes.
            - Chaque segment de la géométrie (entre 2 nœuds) est contenu dans la zone racinaire.
            - Il existe toujours un moment où la racine pousse librement (sans croisement).
        Méthode d'estimation :
            - Détecter les arêtes qui se croisent.
            - Pour les segments non croisés, pour chaque segment :
                * Calculer la direction de la racine (vecteur unitaire) et sa normale.
                * À partir du milieu du segment, sonder dans la direction normale (dans les 2 sens)
                  pour compter le nombre de pixels appartenant à la racine.
        """
        # Correction : utiliser des combinaisons pour parcourir toutes les paires d'arêtes
        crossing_edges = find_crossing_edges(self.mtg)
        print("Arêtes qui se croisent :", crossing_edges)
        intensity_dict = self.group_nodes_by_intensity()
        print("Groupes de nœuds par intensité :", intensity_dict)

    def group_nodes_by_intensity(self):
        """
        Parcourt tous les nœuds (points) de la géométrie et, pour chacun,
        récupère l'intensité du pixel correspondant dans le date_map.
        
        Retourne un dictionnaire dont :
            - les clés sont les intensités (valeurs du date_map)
            - les valeurs sont des listes de nœuds ayant exactement cette intensité.
              Chaque nœud est représenté par un tuple (vertex_id, (x, y)).
        """
        if self.date_map is None:
            raise ValueError("date_map n'est pas chargé.")
        
        intensity_dict = {}
        # On parcourt chaque polyline associée à un vertex du MTG.
        for vid, polyline in self.geometry.items():
            for pt in polyline:
                # On considère que pt est (x, y) dans le système de coordonnées du système racinaire.
                # Pour récupérer l'intensité dans date_map, on convertit les coordonnées en indices de pixels.
                col = int(round(pt[0]))
                row = int(round(pt[1]))
                # Vérifier que l'indice est dans les bornes de l'image date_map.
                if row < 0 or row >= self.date_map.shape[0] or col < 0 or col >= self.date_map.shape[1]:
                    continue
                intensity = self.date_map[row, col]
                if intensity not in intensity_dict:
                    intensity_dict[intensity] = []
                # corresponding time_hours of the vertex
                time_hours = self.time_hours[vid][0]
                intensity_dict[intensity].append((vid, (pt[0], pt[1]), time_hours))
        return intensity_dict


def find_crossing_edges(mtg):
    """
    Trouve les arêtes qui se croisent dans le graphe MTG.
    
    :param mtg: MTG du système racinaire.
    :return: Liste des couples d'identifiants d'arêtes qui se croisent.
    """
    def do_edges_cross(p1, p2, q1, q2):
        """
        Vérifie si deux segments (p1-p2 et q1-q2) se coupent.
        """
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
    
    edge_coords = {}
    for vid, polyline in mtg.property("geometry").items():
        for i in range(len(polyline) - 1):
            edge_coords[(vid, i)] = (polyline[i], polyline[i + 1])
    
    crossing_edges = []
    # Utilisation de combinations pour éviter les erreurs d'itération
    for key1, key2 in combinations(edge_coords.keys(), 2):
        (vid1, i) = key1
        (vid2, j) = key2
        if vid1 == vid2:
            continue  # ne pas comparer des segments du même vertex
        p1, p2 = edge_coords[key1]
        q1, q2 = edge_coords[key2]
        if do_edges_cross(p1, p2, q1, q2):
            crossing_edges.append((key1, key2))
    return crossing_edges

# Re-write de mtg2rsml pour éviter l'erreur d'écriture
def mtg2rsml(g, rsml_file):
    """
    Write **continuous** mtg `g` in `rsml_file`
    :See also: `Dumper`, `rsml.continuous`
    """
    dump = CD.Dumper()
    s = dump.dump(g)
    if isinstance(rsml_file, str):
        with open(rsml_file, 'wb') as f:
            f.write(s)
    else:
        rsml_file.write(s) 


# Exemple d'utilisation
if __name__ == "__main__":
    folder = "/home/loai/Images/DataTest/230629PN021/"
    # On choisit de charger le dateMap en passant load_date_map=True
    root_system = RootSystem(folder, load_date_map=True)

    # Affichage de quelques informations chargées
    print("Image stack shape:", root_system.image_stack.shape)
    print("Nombre de temps d'observation:", len(root_system.obs_times) if root_system.obs_times else "N/A")
    print("Date map chargée:", root_system.date_map is not None)
    print("MTG chargé:", root_system.mtg is not None)
    print("Géométrie chargée:", root_system.geometry is not None)
    print("Temps par vertex chargé:", root_system.time_hours is not None)
    
    root_system.estimate_diameter_from_date_map()
    # Lancement de la visualisation interactive
    root_system.visualize(show_date_map=True)
    
    # Sauvegarde des données dans un dossier de destination
    dest_folder = "/home/loai/Images/DataTest/230629PN021_copy"
    root_system.save2folder(dest_folder)
    #import matplotlib.pyplot as plt
    #rsml.plot2d(root_system.mtg)
    #plt.show()
    
    
    