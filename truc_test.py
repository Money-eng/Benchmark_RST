from openalea.rsml import rsml2mtg

path = "/home/loai/Documents/code/RSMLExtraction/test.rsml"
path2 = "/home/loai/Documents/code/RSMLExtraction/testo.rsml" 

mtg = rsml2mtg(path)


print(mtg)
print(mtg.vertices())
print("vertex min scale:", min(mtg.scales()))
print("vertex max scale:", max(mtg.scales()))
print(("vertices at scale 0 :", list(mtg.vertices(scale=0))))
print(("vertices at scale 1 :", list(mtg.vertices(scale=1))))
print(("vertices at scale 2 :", list(mtg.vertices(scale=2))))

# get vertex from testo at scale 1 
mtg2 = rsml2mtg(path2)
print("vertex min scale:", min(mtg2.scales()))
print("vertex max scale:", max(mtg2.scales()))
print(("vertices at scale 0 :", list(mtg2.vertices(scale=0))))
print(("vertices at scale 1 :", list(mtg2.vertices(scale=1))))
print(("vertices at scale 2 :", list(mtg2.vertices(scale=2))))
vertex_plant = list(mtg2.vertices(scale=1))[0]

mtg.insert_sibling(vtx_id1 = list(mtg2.vertices(scale=1))[0], vtx_id2 = vertex_plant)
print("After adding child from mtg2 to mtg1")