import pandas as pd

from os.path import join
import os
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from unimol_tools import UniMolRepr
import numpy as np
from tqdm import tqdm

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

clf = UniMolRepr(data_type='molecule', 
                 remove_hs=True,
                 model_name='unimolv2', # avaliable: unimolv1, unimolv2
                 model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                 )

UNIPROT_df = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "enzyme_data", "Uniprot_df_with_seq_identities.pkl"))
df_UID_MID = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "enzyme_substrate_data", "df_UID_MID.pkl"))
mol_IDs = list(set(df_UID_MID["molecule ID"]))

df_unimol = pd.DataFrame(data = {"substrate ID" : mol_IDs})
df_unimol["unimol"] = ""

UNIPROT_df = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "enzyme_data", "Uniprot_df_with_seq_identities.pkl"))
df_UID_MID = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "enzyme_substrate_data", "df_UID_MID.pkl"))

ChEBI_IDs = list(set(df_UID_MID["molecule ID"]))
f = open(join(CURRENT_DIR, ".." ,"data", "substrate_data", "ChEBI_IDs.txt"),"w")
for ID in list(set(ChEBI_IDs)):
    if ID[:2] == "CH":
        f.write(str(ID.replace("CHEBI", "ChEBI")) + "\n")
f.close()

df_chebi_to_inchi = pd.read_csv(join(CURRENT_DIR, ".." ,"data", "substrate_data", "chebiID_to_inchi.tsv"), sep = "\t")
df_chebi_to_inchi.head()

mol_folder = "./mol-files"

failed = 0

for ind in tqdm(df_unimol.index):
    met_ID = df_unimol["substrate ID"][ind]
    is_CHEBI_ID = (met_ID[0:5] == "CHEBI")
    
    
    if is_CHEBI_ID:
        try:
            ID = int(met_ID.split(" ")[0].split(":")[-1])
            Inchi = list(df_chebi_to_inchi["Inchi"].loc[df_chebi_to_inchi["ChEBI"] == float(ID)])[0]
            if not pd.isnull(Inchi):
                mol = Chem.inchi.MolFromInchi(Inchi)
        except IndexError:
            mol = None
        
    else:
        try:
            mol = Chem.MolFromMolFile(mol_folder +  "/mol-files/" + met_ID + '.mol')
        except OSError:
            None
            
    if mol is not None:
        try:
            mol = Chem.RemoveHs(mol, implicitOnly=False)
            h_smarts = Chem.MolFromSmarts("[H]")
            # 删除所有匹配到的子结构
            mol = Chem.DeleteSubstructs(mol, h_smarts)
            Chem.SanitizeMol(mol)  
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol)
            # 获取原子坐标
            conformer = mol.GetConformer()
            positions = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
            positions = np.array([[pos.x, pos.y, pos.z] for pos in positions])
            inputs = {"atoms": [atoms], "coordinates": [positions]}
            unimol_repr = clf.get_repr(inputs, return_atomic_reprs=True)
            df_unimol["unimol"][ind] = unimol_repr['cls_repr']
        except:
            print(f"error :{ind}")
            failed += 1
        
        #input()

print(f"{failed} failed")
df_unimol = df_unimol.loc[df_unimol["unimol"] !=""]

df_unimol.to_pickle(join(CURRENT_DIR, ".." ,"data", "substrate_data", "df_unimol.pkl"))