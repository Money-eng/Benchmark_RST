import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from apls import APLSMetric

# ==========================================
# 1. Remplacement de sknw (Version stable)
# ==========================================
def skeleton_to_graph(skeleton):
    """
    Convertit un squelette binaire en graphe NetworkX sans utiliser sknw/numba.
    Cette méthode est plus robuste aux versions de Python.
    """
    # Récupérer les coordonnées des pixels du squelette (y, x)
    pixels = np.column_stack(np.where(skeleton))
    
    # Créer un graphe où chaque pixel est un nœud
    G = nx.Graph()
    for r, c in pixels:
        G.add_node((r, c)) # On utilise des tuples (r, c) comme IDs
        
    # Connecter les voisins (8-connectivité)
    # Pour chaque pixel, on regarde ses voisins et on ajoute une arête
    for r, c in pixels:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in G.nodes:
                    # Poids = distance euclidienne (1 ou 1.414)
                    dist = np.sqrt(dr**2 + dc**2)
                    G.add_edge((r, c), (nr, nc), length=dist)
    
    # Simplification du graphe : 
    # On supprime les nœuds de degré 2 (les pixels au milieu d'une ligne)
    # pour ne garder que les intersections et les extrémités.
    
    # On copie pour itérer tranquillement
    nodes_to_remove = [n for n in G.nodes if G.degree(n) == 2]
    
    for n in nodes_to_remove:
        # Récupérer les deux voisins
        neighbors = list(G.neighbors(n))
        u, v = neighbors[0], neighbors[1]
        
        # Calculer la nouvelle longueur (somme des deux segments)
        w1 = G[u][n]['length']
        w2 = G[n][v]['length']
        
        # Ajouter l'arête directe u-v
        # Attention : si l'arête existe déjà (cas d'un petit cycle), on prend le min ou on ajoute
        if G.has_edge(u, v):
            G[u][v]['length'] += w1 + w2 # Cas rare, simplification additive
        else:
            G.add_edge(u, v, length=w1 + w2)
            
        # Supprimer le nœud intermédiaire
        G.remove_node(n)
        
    # Mise en forme finale pour APLS (ajout x, y explicites)
    # Les nœuds sont actuellement des tuples (row, col)
    for node in G.nodes:
        r, c = node
        G.nodes[node]['y'] = r
        G.nodes[node]['x'] = c
        
    # Renommer les nœuds en entiers pour être propre (optionnel mais conseillé)
    G = nx.convert_node_labels_to_integers(G, label_attribute='pos_tuple')
    
    # S'assurer que les attributs x,y sont conservés après renommage
    for n, data in G.nodes(data=True):
        if 'pos_tuple' in data:
            r, c = data['pos_tuple']
            G.nodes[n]['y'] = r
            G.nodes[n]['x'] = c
            
    return G

def mask_to_graph(binary_mask):
    # 1. Squelettisation
    skeleton = skeletonize(binary_mask.astype(bool))
    
    # 2. Conversion avec notre nouvelle fonction (plus de sknw)
    G = skeleton_to_graph(skeleton)
    
    return G

# ==========================================
# NOUVELLE FONCTION : Augmentation du graphe
# ==========================================
def inject_midpoints(G, interval_pixels):
    """
    Injecte des nœuds intermédiaires le long des arêtes trop longues.
    C'est CRUCIAL pour l'APLS sur des segmentations discontinues.
    """
    G_aug = G.copy()
    edges_to_remove = []
    edges_to_add = []
    
    # On itère sur toutes les arêtes existantes
    for u, v, data in G.edges(data=True):
        length = data.get('length', 0)
        
        # Si l'arête est plus longue que l'intervalle, on la découpe
        if length > interval_pixels:
            edges_to_remove.append((u, v))
            
            # Récupérer positions
            p_u = np.array([G.nodes[u]['y'], G.nodes[u]['x']])
            p_v = np.array([G.nodes[v]['y'], G.nodes[v]['x']])
            
            # Combien de segments ?
            num_segments = int(np.ceil(length / interval_pixels))
            
            # Création des points intermédiaires
            prev_node = u
            for i in range(1, num_segments):
                # Interpolation linéaire (t varie de 0 à 1)
                t = i / num_segments
                new_pos = p_u + t * (p_v - p_u)
                
                # Créer un ID unique pour le nouveau nœud (tuple float)
                new_node_id = (new_pos[0], new_pos[1])
                
                # Ajouter le nœud avec ses attributs
                G_aug.add_node(new_node_id, y=new_pos[0], x=new_pos[1])
                
                # Calculer la distance du petit segment
                seg_len = np.linalg.norm(new_pos - np.array([G_aug.nodes[prev_node]['y'], G_aug.nodes[prev_node]['x']]))
                
                # Ajouter l'arête
                edges_to_add.append((prev_node, new_node_id, seg_len))
                prev_node = new_node_id
            
            # Connecter le dernier point intermédiaire au nœud final v
            last_seg_len = np.linalg.norm(p_v - np.array([G_aug.nodes[prev_node]['y'], G_aug.nodes[prev_node]['x']]))
            edges_to_add.append((prev_node, v, last_seg_len))
            
    # Appliquer les modifications
    G_aug.remove_edges_from(edges_to_remove)
    for u, v, l in edges_to_add:
        G_aug.add_edge(u, v, length=l)
        
    return G_aug

# ==========================================
# TEST CORRIGÉ
# ==========================================

# 1. Création des Graphes (comme avant)
# mask_gt (ligne continue) / mask_pred (ligne coupée et décalée)
height, width = 100, 100
mask_gt = np.zeros((height, width), dtype=int)
mask_pred = np.zeros((height, width), dtype=int)

mask_gt[10:90, 50] = 1 
mask_pred[10:45, 52] = 1 
mask_pred[55:90, 52] = 1 

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.title("Ground Truth Mask")
plt.imshow(mask_gt, cmap='gray')
plt.subplot(1,2,2)
plt.title("Predicted Mask")
plt.imshow(mask_pred, cmap='gray')
plt.show()

# Conversion de base
G_gt_raw = mask_to_graph(mask_gt)
G_pred_raw = mask_to_graph(mask_pred)

print(f"Brut -> GT Noeuds: {len(G_gt_raw.nodes())}, Pred Noeuds: {len(G_pred_raw.nodes())}")

# 2. APPLICATION DU FIX : Injection de nœuds
# On injecte un nœud tous les 10 pixels (ajuster selon échelle, ex: 50m)
interval = 10.0 
G_gt_aug = inject_midpoints(G_gt_raw, interval)
G_pred_aug = inject_midpoints(G_pred_raw, interval)

plt.subplot(1,2,1)
pos_gt = {n: (data['x'], data['y']) for n, data in G_gt_aug.nodes(data=True)}
nx.draw(G_gt_aug, pos=pos_gt, node_size=10, node_color='blue', edge_color='lightblue')
plt.title("Graphe GT Augmenté") 
plt.subplot(1,2,2)
pos_pred = {n: (data['x'], data['y']) for n, data in G_pred_aug.nodes(data=True)}
nx.draw(G_pred_aug, pos=pos_pred, node_size=10, node_color='red', edge_color='pink')
plt.title("Graphe Pred Augmenté") 
plt.show()

print(f"Augmenté -> GT Noeuds: {len(G_gt_aug.nodes())}, Pred Noeuds: {len(G_pred_aug.nodes())}")

# 3. Calcul APLS sur les graphes augmentés
# snap_buffer doit être suffisant pour attraper le décalage (ici 2px, buffer=5px c'est bon)
apls_tool = APLSMetric(G_gt_aug, G_pred_aug, snap_buffer_meters=5.0) 
score = apls_tool.compute()

print(f"Score APLS final : {score:.4f}")