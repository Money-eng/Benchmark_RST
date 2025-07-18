import os
import tifffile
import numpy as np
import rsml
from rsml import hirros
from openalea.mtg import MTG
import utils.CustomDumper as CD
from itertools import combinations

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
        self.time = None

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
            mask_t1_file = os.path.join(self.folder_path, "31_mask_at_t1.tif")
            if os.path.exists(date_map_file):
                self.date_map = tifffile.imread(date_map_file)
            if os.path.exists(mask_t1_file):
                self.mask_t1 = tifffile.imread(mask_t1_file)
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
            self.time = self.mtg.property('time')

            # Mise à jour des metadata de l'image
            metadata = self.mtg.graph_properties().get('metadata', {})
            image_meta = metadata.get('image', {})
            if image_meta.get('name', None) is None:
                image_meta['name'] = os.path.basename(image_stack_file)
                metadata['image'] = image_meta

            # Initialisation de la section "functions" des metadata si absente
            if 'functions' not in metadata:
                metadata['functions'] = {}
                metadata['functions']['time'] = self.mtg.properties()['time']
                metadata['functions']['time_hours'] = self.mtg.properties()['time_hours']
                
                # Ajout de la fonction "diameter" si elle n'existe pas déjà
                if 'diameter' not in metadata['functions']:
                    if self.load_date_map and self.date_map is not None:
                        # Calcul du diamètre à partir du date_map
                        import right_Diameter as grd
                        diameter = grd.project_root_system_on_diameter_map(self)
                        #diameter2 = grd.project_root_system_on_diameter_maps(self)
                        #metadata['functions']['diameter'] = diameter2
                        #self.mtg.add_property('diameter')
                        #self.mtg.properties()['diameter'] = diameter2
                        metadata['functions']['diameter'] = diameter
                        self.mtg.add_property('diameter')
                        self.mtg.properties()['diameter'] = diameter
                    else:
                        print("Le date_map n'est pas chargé, impossible de calculer le diamètre.")
            # Mise à jour des metadata dans le mtg
            self.mtg.graph_properties()['metadata'] = metadata
        # to remove unwanted date_map values
        self.correct_date_map()
        
        self.assert_len_consistency()  # Vérification de la cohérence des longueurs des propriétés
    
    # on calcule le diamètre avant de corriger la date_map ? on va supposer que ça passe :
    def correct_date_map(self):
        date_map = self.date_map
        mtg = self.mtg
        # get the connected components of the date_map
        from skimage.measure import label
        binary_date_map = np.zeros_like(date_map)
        binary_date_map[date_map > 0] = 1
        labeled_date_map = label(binary_date_map)
        cc_of_mtg = set()
        # for each vertex in the mtg, get the connected component of the date_map associated with the vertex position and add it to the set
        for vid in mtg.vertices_iter():
            if mtg.scale(vid) == 0 or mtg.scale(vid) == 1:
                continue
            # get the position of the vertex in the date_map
            positions = mtg.property('geometry')[vid]
            # get the connected component of the date_map associated with the vertex position
            for pos in positions:
                cc = labeled_date_map[int(pos[1]), int(pos[0])]
                cc_of_mtg.add(cc)
        cc_date_map = np.unique(labeled_date_map)
        # if a connected component of the date_map is not in the set, we delete it from the original date_map image
        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #ax[0].imshow(date_map)
        #ax[0].set_title("Date Map")
        #ax[1].imshow(labeled_date_map)
        # for each connected component of the date_map, write the number of the connected component in the image at one if its pixel position
        #for cc in cc_date_map:
         #   pixel_found = np.where(labeled_date_map == cc)
          #  if len(pixel_found[0]) > 0:
           #     x = pixel_found[1][0]
            #    y = pixel_found[0][0]
            #    ax[1].text(x, y, str(cc), color='red', fontsize=8)
        #ax[1].set_title("Labeled Date Map with CC")
        #plt.show()
        for cc in cc_date_map:
            if cc not in cc_of_mtg and cc != 0:
                date_map[labeled_date_map == cc] = 0
        #plt.imshow(date_map)
        #plt.show()

    def visualize(self, show_date_map: bool = False, show_diameter_overlay: bool = False):
        """
        Lance une visualisation Napari affichant l'image stack en fond et 
        la superposition des données MTG (nœuds et arêtes) en fonction du temps.
        
        Si show_date_map est True et que self.date_map est chargé, 
        ouvre une deuxième fenêtre affichant le date_map (2D) et le graph correspondant 
        au dernier temps d'observation.
        
        Si show_diameter_overlay est True et que le diamètre est disponible dans les metadata,
        affiche un trait transparent le long de la portion de polyline visible avec une épaisseur 
        correspondant au diamètre estimé.
        """
        import napari
        import numpy as np

        def get_gradient_color(f,
                            start_color=np.array([1.0, 1.0, 0.9]),  # blanc-jaune clair
                            end_color=np.array([0.5, 0.0, 0.5])):   # violet
            color = (1 - f) * start_color + f * end_color
            return np.concatenate([color, [1.0]])

        def get_truncated_polyline(polyline, t_vector, current_time):
            """
            Tronque la polyline pour ne conserver que les points dont le temps est inférieur
            ou égal à current_time. Inverse x et y pour l'affichage.
            """
            polyline = np.array(polyline)
            polyline = polyline[:, ::-1]  # inversion x et y
            t_vector = np.array(t_vector)
            valid = t_vector <= current_time
            if not np.any(valid):
                return None
            last_idx = np.where(valid)[0][-1]
            return polyline[:last_idx + 1]

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

        viewer = napari.Viewer()
        current_time = self.obs_times[0]
        initial_edges = []
        initial_nodes = []
        initial_nodes_colors = []

        # Calque pour les labels de vertex (numéros)
        initial_labels_positions = []
        initial_labels_texts = []

        for vid, polyline in self.geometry.items():
            t_vector = self.time_hours[vid]
            truncated = get_truncated_polyline(polyline, t_vector, current_time)
            if truncated is not None and len(truncated) > 1:
                initial_edges.append(truncated)
                for pt in truncated:
                    initial_nodes.append(pt)
                    initial_nodes_colors.append(vertex_colors[vid])
                initial_labels_positions.append(truncated[-1])
                initial_labels_texts.append(str(vid))
                    
        initial_nodes = np.array(initial_nodes)
        initial_nodes_colors = np.array(initial_nodes_colors)

        # Ajout de l'image stack
        viewer.add_image(
            self.image_stack,
            name="Série temporelle",
            colormap="gray",
            contrast_limits=[self.image_stack.min(), self.image_stack.max()]
        )

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
        labels_layer = None
        if initial_labels_positions:
            labels_layer = viewer.add_points(
                np.array(initial_labels_positions),
                face_color="transparent",
                size=0,
                text=initial_labels_texts,
                name="Vertex Labels"
            )
            labels_layer.text_size = 8
            labels_layer.text_color = "white"

        # ===== Adaptation pour le second diamètre =====
        # On récupère le dictionnaire des diamètres calculé par project_root_system_on_diameter_maps
        # (pour chaque vertex, une liste de diamètres en fonction du temps).
        diameter_layer = None
        if show_diameter_overlay:
            metadata = self.mtg.graph_properties().get('metadata', {})
            functions = metadata.get('functions', {})
            diameter_dict = None # functions.get('diameter', None)
            if diameter_dict is None:
                print("Diamètre non disponible dans les metadata.")
            else:
                diameter_shapes = []
                edge_widths = []
                edge_colors = []
                # Pour chaque vertex, on utilise la portion visible de la polyline
                current_idx = 0  # indice temporel initial
                for vid, polyline in self.geometry.items():
                    if vid in diameter_dict:
                        truncated = get_truncated_polyline(polyline, self.time_hours[vid], current_time)
                        if truncated is None or len(truncated) < 2:
                            continue
                        diam_list = diameter_dict[vid]
                        if current_idx < len(diam_list):
                            diameter_value = diam_list[current_idx]
                        else:
                            diameter_value = diam_list[-1]
                        diameter_shapes.append(truncated)
                        edge_widths.append(diameter_value)
                        edge_colors.append((0, 0, 1, 0.5))
                if diameter_shapes:
                    diameter_layer = viewer.add_shapes(
                        diameter_shapes,
                        shape_type='path',
                        edge_color=edge_colors,
                        edge_width=edge_widths,
                        face_color='transparent',
                        name='Diameter Overlay'
                    )
        # =================================================

        @viewer.dims.events.current_step.connect
        def update_layers(event):
            time_idx = event.value[0]
            current_time = self.obs_times[time_idx] if time_idx < len(self.obs_times) else self.obs_times[-1]

            new_edges = []
            new_edge_colors = []
            new_nodes = []
            new_nodes_colors = []
            new_labels_positions = []
            new_labels_texts = []

            for vid, polyline in self.geometry.items():
                t_vector = self.time_hours[vid]
                truncated = get_truncated_polyline(polyline, t_vector, current_time)
                if truncated is not None and len(truncated) > 1:
                    new_edges.append(truncated)
                    new_edge_colors.append(vertex_colors[vid])
                    for pt in truncated:
                        new_nodes.append(pt)
                        new_nodes_colors.append(vertex_colors[vid])
                    new_labels_positions.append(truncated[-1])
                    new_labels_texts.append(str(vid))

            nodes_array = np.array(new_nodes) if new_nodes else np.empty((0, 2))
            nodes_layer.data = nodes_array
            if new_nodes_colors:
                nodes_layer.face_color = np.array(new_nodes_colors)
            edges_layer.data = new_edges
            if new_edge_colors:
                edges_layer.edge_color = new_edge_colors

            if labels_layer is not None:
                labels_layer.data = np.array(new_labels_positions) if new_labels_positions else np.empty((0, 2))
                labels_layer.text = new_labels_texts

            # ===== Mise à jour du calque du diamètre (second méthode) =====
            if show_diameter_overlay and diameter_layer is not None:
                new_diameter_shapes = []
                new_edge_widths = []
                new_edge_colors = []
                # On considère ici que l'indice temporel de la vue correspond à l'indice dans les listes de diamètres.
                current_idx = time_idx  
                for vid, polyline in self.geometry.items():
                    if vid in diameter_dict:
                        truncated = get_truncated_polyline(polyline, self.time_hours[vid], current_time)
                        if truncated is None or len(truncated) < 2:
                            continue
                        diam_list = diameter_dict[vid]
                        if current_idx < len(diam_list):
                            diameter_value = diam_list[current_idx]
                        else:
                            diameter_value = diam_list[-1]
                        new_diameter_shapes.append(truncated)
                        new_edge_widths.append(diameter_value)
                        new_edge_colors.append((0, 0, 1, 0.5))
                if new_diameter_shapes:
                    diameter_layer.data = new_diameter_shapes
                    diameter_layer.edge_width = new_edge_widths
                    diameter_layer.edge_color = new_edge_colors
            # =================================================

        if show_date_map:
            if self.date_map is None:
                print("Avertissement: date_map non chargé. La vue séparée ne peut pas être affichée.")
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

    def save2folder(self, destination_folder: str, save_date_map: bool = False):
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
            
            if save_date_map and self.date_map is not None:
                date_map_filename = "40_date_map.tif"
                date_map_save_path = os.path.join(destination_folder, date_map_filename)
                mask_at_t1 = "31_mask_at_t1.tif"
                tifffile.imwrite(date_map_save_path, self.date_map)
                mask_save_path = os.path.join(destination_folder, mask_at_t1)
                tifffile.imwrite(mask_save_path, self.mask_t1)  # Sauvegarde un masque binaire
                print(f"Date map sauvegardée dans : {date_map_save_path}")
            
            # copy /home/loai/Images/DataTest/UC1/InfoSerieRootSystemTracker.csv to destination_folder
            import shutil
            info_file_src = "/home/loai/Images/DataTest/UC1/InfoSerieRootSystemTracker.csv"
            info_file_dst = os.path.join(destination_folder, "InfoSerieRootSystemTracker.csv")
            try:
                shutil.copy(info_file_src, info_file_dst)
                print(f"Fichier InfoSerieRootSystemTracker.csv copié dans : {info_file_dst}")
            except Exception as e:
                print(f"Erreur lors de la copie du fichier InfoSerieRootSystemTracker.csv : {e}")

    def assert_len_consistency(self):   
        mtg = self.mtg
        # for all nodes in mtg geometry, assert it has the same length as the time, time_hours and diameter properties
        for vid in mtg.vertices_iter():
            if mtg.scale(vid) != mtg.max_scale():
                continue
            geometry = mtg.property('geometry').get(vid, [])
            time = mtg.property('time').get(vid, [])
            time_hours = mtg.property('time_hours').get(vid, [])
            diameter = mtg.property('diameter').get(vid, [])
            
            if len(geometry) != len(time) or len(time) != len(time_hours) or len(time_hours) != len(diameter):
                raise ValueError(f"Longueur incohérente pour le vertex {vid}: "
                                f"geometry={len(geometry)}, time={len(time)}, "
                                f"time_hours={len(time_hours)}, diameter={len(diameter)}")

# not used
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

# Exemple d'utilisation principal
if __name__ == "__main__":
    folder = "/home/loai/Documents/code/RSMLExtraction/230629PN014"
    root_system = RootSystem(folder, load_date_map=True)

    print("Image stack shape:", root_system.image_stack.shape)
    print("Nombre de temps d'observation:", len(root_system.obs_times) if root_system.obs_times else "N/A")
    print("Date map chargée:", root_system.date_map is not None)
    print("MTG chargé:", root_system.mtg is not None)
    print("Géométrie chargée:", root_system.geometry is not None)
    print("Temps par vertex chargé:", root_system.time_hours is not None)
    # Lancement de la visualisation interactive (ici, on n'affiche que l'image et le MTG)
    root_system.visualize(show_date_map=True, show_diameter_overlay=True)


if __name__ == "__main__0":
    # Chemin de base contenant tous les dossiers à traiter
    source_base = '/home/loai/Images/DataTest/UC1'
    # Dossier de destination où sera recréée l'arborescence et sauvegardé chaque dataset
    dest_base = '/home/loai/Images/DataTest/UC1_data' # Modifiez ce chemin selon votre besoin

    # Assurez-vous que le dossier de destination existe
    if not os.path.exists(dest_base):
        os.makedirs(dest_base)

    # Récupérer la liste des sous-dossiers dans source_base
    subdirs = [d for d in os.listdir(source_base) if os.path.isdir(os.path.join(source_base, d))]

    for subdir in subdirs:
        source_folder = os.path.join(source_base, subdir)
        dest_folder = os.path.join(dest_base, subdir)
        print(f"Traitement de {source_folder} vers {dest_folder}")
        
        try:
            # Charger le dataset avec le date_map activé pour estimer le diamètre
            rs = RootSystem(source_folder, load_date_map=True)
            # Sauvegarder le dataset dans le dossier de destination
            rs.save2folder(dest_folder, save_date_map=True)
            print(f"Dataset pour {subdir} sauvegardé avec succès.")
        except Exception as e:
            print(f"Erreur lors du traitement de {source_folder} : {e}")