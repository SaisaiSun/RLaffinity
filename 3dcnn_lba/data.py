import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#import dotenv as de
import numpy as np
import pandas as pd

from lba.datasets import LMDBDataset
from lba.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid
from torch.utils.data import DataLoader
import scipy.io

#de.load_dotenv(de.find_dotenv(usecwd=True))


class CNN3D_TransformLBA(object):
    def __init__(self, random_seed=None, **kwargs):
        self.random_seed = random_seed
        self.grid_config =  dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                #'H': 0,
                'C': 0,
                'O': 1,
                'N': 2,
                'P': 3,
                'F': 4,
                'Cl': 5,
                'Br': 6,
                'I': 7,
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 20.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms_pocket, atoms_ligand):
        # Use center of ligand as subgrid center
        elements = atoms_pocket['element'].values
        #print(elements.shape)
        ligand_pos = atoms_ligand[['x', 'y', 'z']].astype(np.float32)
        ligand_center = get_center(ligand_pos)
        # Generate random rotation matrix
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        # Transform protein/ligand into voxel grids and rotate
        grid = get_grid(pd.concat([atoms_pocket, atoms_ligand]),
                        ligand_center, config=self.grid_config, rot_mat=rot_mat)
        # Last dimension is atom channel, so we need to move it to the front
        # per pytroch style
        grid = np.moveaxis(grid, -1, 0)
        return grid

    def __call__(self, item):
        # Transform protein/ligand into voxel grids.
        # Apply random rotation matrix.
        transformed1 = {
            'feature': self._voxelize(item['atoms_pocket'], item['atoms_ligand']),
            #'label': item['scores']['neglog_aff'],
            'id': item['id']
        }
        transformed2 = {
            'feature': self._voxelize(item['atoms_pocket'], item['atoms_ligand']),
            #'label': item['scores']['neglog_aff'],
            'id': item['id']
        }
        return transformed1['feature'],transformed2['feature']


if __name__=="__main__":
    dataset_path = '/Users/saisaisun/Downloads/RNA-affinity-pretrain/data_for_atom3d/data' #os.path.join(os.environ['LBA_DATA'], 'all')
    dataset1 = LMDBDataset(dataset_path, transform=CNN3D_TransformLBA(radius=13.0))
    #dataset2 = LMDBDataset(dataset_path, transform=CNN3D_TransformLBA(radius=13.0))
    dataloader1 = DataLoader(dataset1, batch_size=2, shuffle=False)
    #dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=False)

    feature1 = []
    feature2 = []
    for item1,item2 in dataloader1:
        #print(item1['id'])
        #print('feature shape:', item['feature'].shape)
        #print('label:', item['label'])
        feature1 = item1.numpy()
        feature2 = item2.numpy()
        #print(feature.ravel()[np.flatnonzero(feature)])
        print("数组和数组比较:{}".format(np.array_equal(feature1, feature2)))
        #scipy.io.savemat('/Users/saisaisun/Downloads/RNA-affinity-pretrain/RNA-affinity-pdb_data/data/all/data.mat',{'feature':feature})
        break

    # for item2 in dataloader2:
    #     print(item2['id'])
    #     feature2 = item2['feature'].numpy()
    #     break
    # print("数组和数组比较:{}".format(np.array_equal(feature1, feature2)))
