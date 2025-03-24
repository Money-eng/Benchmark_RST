import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import tifffile
import numpy as np
import networkx as nx
import unittest

########################################
# Définition des classes RSML (parsing)
########################################

class RSMLMetadata:
    def __init__(self, version: str, unit: str, size: Optional[float] = None,
                 last_modified: Optional[str] = None, software: Optional[str] = None,
                 user: Optional[str] = None, file_key: Optional[str] = None, **kwargs):
        self.version = version
        self.unit = unit
        self.size = size
        self.last_modified = last_modified
        self.software = software
        self.user = user
        self.file_key = file_key
        # D'autres métadonnées éventuelles
        self.extra = kwargs

    def __repr__(self):
        return f"RSMLMetadata(version={self.version}, unit={self.unit})"

class RSMLPoint:
    def __init__(self, coord_t: float, coord_th: float, coord_x: float, coord_y: float,
                 diameter: float = 0.0, vx: float = 0.0, vy: float = 0.0):
        self.coord_t = coord_t
        self.coord_th = coord_th
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.diameter = diameter
        self.vx = vx
        self.vy = vy

    def __repr__(self):
        return (f"RSMLPoint(x={self.coord_x}, y={self.coord_y}, "
                f"diameter={self.diameter if self.diameter else 0}, "
                f"t={self.coord_t if self.coord_t else 0}, "
                f"th={self.coord_th if self.coord_th else 0}, "
                f"vx={self.vx if self.vx else 0}, vy={self.vy if self.vy else 0})")

class RSMLGeometry:
    def __init__(self, points: List[RSMLPoint]):
        self.points = points

    def __repr__(self):
        return f"RSMLGeometry({len(self.points)} points)"

class RSMLFunction:
    def __init__(self, name: str, samples: List[float]):
        self.name = name
        self.samples = samples

    def __repr__(self):
        return f"RSMLFunction(name={self.name}, samples={self.samples})"

class RSMLRoot:
    def __init__(self, node_id: str, label: str = "", 
                 geometry: Optional[RSMLGeometry] = None,
                 functions: Optional[Dict[str, RSMLFunction]] = None,
                 properties: Optional[Dict[str, float]] = None,
                 annotations: Optional[Dict[str, any]] = None):
        self.node_id = node_id
        self.label = label
        self.geometry = geometry
        self.functions = functions if functions is not None else {}
        self.properties = properties if properties is not None else {}
        self.annotations = annotations if annotations is not None else {}
        self.children: List[RSMLRoot] = []  # branches secondaires, tertiaires, etc.

    def add_child(self, child: 'RSMLRoot'):
        self.children.append(child)

    def __repr__(self):
        return f"RSMLRoot(id={self.node_id}, label={self.label}, children={len(self.children)})"

class RSMLPlant:
    def __init__(self, plant_id: str, label: str = "", roots: Optional[List[RSMLRoot]] = None):
        self.plant_id = plant_id
        self.label = label
        # Une plante peut contenir plusieurs racines primaires
        self.roots = roots if roots is not None else []

    def __repr__(self):
        return f"RSMLPlant(id={self.plant_id}, label={self.label}, roots={len(self.roots)})"

class RSMLDocument:
    def __init__(self, metadata: RSMLMetadata, plants: List[RSMLPlant]):
        self.metadata = metadata
        self.plants = plants

    def __repr__(self):
        return f"RSMLDocument(version={self.metadata.version}, plants={len(self.plants)})"

########################################
# Fonctions de parsing RSML
########################################

def parse_metadata(elem: ET.Element) -> RSMLMetadata:
    md = {}
    for child in elem:
        md[child.tag] = child.text.strip() if child.text else ""
    version = md.get('version', '')
    unit = md.get('unit', '')
    size = float(md.get('size', '0')) if 'size' in md and md.get('size') else None
    last_modified = md.get('last-modified', None)
    software = md.get('software', None)
    user = md.get('user', None)
    file_key = md.get('file-key', None)
    extra = {k: v for k, v in md.items() if k not in ['version', 'unit', 'size', 'last-modified', 'software', 'user', 'file-key']}
    return RSMLMetadata(version, unit, size, last_modified, software, user, file_key, **extra)

def parse_geometry(geo_elem: ET.Element, funcs_elem: Optional[ET.Element]) -> RSMLGeometry:
    points = []
    polyline_elem = geo_elem.find('polyline')
    if polyline_elem is None:
        return RSMLGeometry(points)
    if funcs_elem is None:
        for point_elem in polyline_elem.findall('point'):
            coord_t = float(point_elem.attrib.get('coord_t', point_elem.attrib.get('t', 0)))
            coord_th = float(point_elem.attrib.get('coord_th', point_elem.attrib.get('th', 0)))
            coord_x = float(point_elem.attrib.get('coord_x', point_elem.attrib.get('x', 0)))
            coord_y = float(point_elem.attrib.get('coord_y', point_elem.attrib.get('y', 0)))
            diameter = float(point_elem.attrib.get('diameter', 0))
            vx = float(point_elem.attrib.get('vx', 0))
            vy = float(point_elem.attrib.get('vy', 0))
            points.append(RSMLPoint(coord_t, coord_th, coord_x, coord_y, diameter, vx, vy))
    else:
        # Utilisation éventuelle de fonctions pour récupérer les features
        diameter_func = funcs_elem.get('diameter', RSMLFunction('diameter', []))
        orientation_func = funcs_elem.get('orientation', RSMLFunction('orientation', []))
        for point_elem in polyline_elem.findall('point'):
            coord_t = float(point_elem.attrib.get('coord_t', point_elem.attrib.get('t', 0)))
            coord_th = float(point_elem.attrib.get('coord_th', point_elem.attrib.get('th', 0)))
            coord_x = float(point_elem.attrib.get('coord_x', point_elem.attrib.get('x', 0)))
            coord_y = float(point_elem.attrib.get('coord_y', point_elem.attrib.get('y', 0)))
            diameter = diameter_func.samples[int(coord_t)] if diameter_func.samples else 0
            vx = orientation_func.samples[int(coord_t)] if orientation_func.samples else 0
            vx = 0
            vy = 0
            points.append(RSMLPoint(coord_t, coord_th, coord_x, coord_y, diameter, vx, vy))
    return RSMLGeometry(points)

def parse_functions(funcs_elem: ET.Element) -> Dict[str, RSMLFunction]:
    functions = {}
    for func_elem in funcs_elem.findall('function'):
        func_name = func_elem.attrib.get('name', '')
        samples = []
        for sample in func_elem.findall('sample'):
            try:
                samples.append(float(sample.text))
            except (TypeError, ValueError):
                pass
        if func_name:
            functions[func_name] = RSMLFunction(func_name, samples)
    return functions

def parse_properties(prop_elem: ET.Element) -> Dict[str, any]:
    properties = {}
    for child in prop_elem:
        try:
            properties[child.tag] = float(child.text)
        except (TypeError, ValueError):
            properties[child.tag] = child.text
    return properties

def parse_annotations(ann_elem: ET.Element) -> Dict[str, any]:
    annotations = {}
    for child in ann_elem:
        key = child.attrib.get('name', child.tag)
        annotations[key] = child.text
    return annotations

def parse_node(elem: ET.Element) -> RSMLRoot:
    node_id = elem.attrib.get('ID', '')
    label = elem.attrib.get('label', '')
    geometry = None
    functions = {}
    properties = {}
    annotations = {}
    
    funcs_elem = elem.find('functions')
    if funcs_elem is not None:
        functions = parse_functions(funcs_elem)
        
    prop_elem = elem.find('properties')
    if prop_elem is not None:
        properties = parse_properties(prop_elem)
        
    ann_elem = elem.find('annotations')
    if ann_elem is not None:
        annotations = parse_annotations(ann_elem)
        
    geo_elem = elem.find('geometry')
    if geo_elem is not None:
        geometry = parse_geometry(geo_elem, functions)
        
    node = RSMLRoot(node_id, label, geometry, functions, properties, annotations)
    
    for child in elem.findall('root'):
        child_node = parse_node(child)
        node.add_child(child_node)
    
    return node

plant_counter = 1
def parse_plant(elem: ET.Element) -> RSMLPlant:
    global plant_counter
    plant_id = elem.attrib.get('ID', '')
    label = elem.attrib.get('label', '')
    
    if not plant_id:
        plant_id = elem.attrib.get('id', f"Plant{plant_counter}")
        plant_counter += 1
    
    roots = [parse_node(root_elem) for root_elem in elem.findall('root')]
    return RSMLPlant(plant_id, label, roots)

def parse_rsml(file_path: str) -> RSMLDocument:
    tree = ET.parse(file_path)
    root_elem = tree.getroot()
    
    metadata_elem = root_elem.find('metadata')
    metadata = parse_metadata(metadata_elem) if metadata_elem is not None else None
    
    scene_elem = root_elem.find('scene')
    plants = []
    if scene_elem is not None:
        for plant_elem in scene_elem.findall('plant'):
            plants.append(parse_plant(plant_elem))
    
    return RSMLDocument(metadata, plants)

########################################
# Conversion RSML vers NetworkX Temporak
########################################


import networkx_temporal as nxt

def build_temporal_graph_nxt(rsml_doc: RSMLDocument) -> nxt.TemporalDiGraph:
    """
    Construit un graphe temporel à partir du document RSML.
    Chaque RSMLRoot sera décomposé en nœuds temporels, un pour chaque point (instant).
    """
    G = nxt.TemporalDiGraph()
    
    for plant in rsml_doc.plants:
        for root in plant.roots:
            # Si la géométrie est présente, on suppose qu'elle contient une séquence de points
            if root.geometry and root.geometry.points:
                # Pour chaque point, on crée un nœud identifié par une clé combinant l'ID du root et l'instant (coord_t)
                prev_node_id = None
                for pt in root.geometry.points:
                    node_id = f"{root.node_id}_{pt.coord_t}"
                    # On ajoute le nœud avec ses attributs spatio-temporels
                    G.add_node(node_id, time=pt.coord_t, x=pt.coord_x, y=pt.coord_y)
                    # Optionnel : on peut ajouter des arêtes temporelles reliant le même root dans le temps
                    if prev_node_id is not None:
                        # Par exemple, l'arête peut contenir des informations sur la distance ou la variation
                        G.add_edge(prev_node_id, node_id, weight=(((pt.coord_x - G.nodes[prev_node_id]['x'])**2 + 
                                                                     (pt.coord_y - G.nodes[prev_node_id]['y'])**2)**0.5))
                    prev_node_id = node_id
            # Pour les branches secondaires, on peut construire récursivement le graphe,
            # éventuellement en distinguant les liens spatiaux de la hiérarchie structurelle.
            # Ici, on peut appeler une fonction récursive si nécessaire.
    return G

# Exemple d'utilisation :
# rsml_doc = parse_rsml("votre_fichier.rsml")
# temporal_graph = build_temporal_graph_nxt(rsml_doc)
# Vous pourrez alors utiliser les fonctionnalités de networkx_temporal pour analyser l'évolution temporelle.
if __name__ == "__main__":
    # Exemple d'utilisation
    rsml_doc = parse_rsml("/home/loai/Images/DataTest/230629PN016/61_graph_expertized.rsml")
    temporal_graph = build_temporal_graph_nxt(rsml_doc)
    print(temporal_graph)