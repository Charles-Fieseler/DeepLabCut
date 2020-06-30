import numpy as np
import open3d as o3
import transformations as trans
from probreg import bcpd
from probreg import callbacks
import copy
import os

# My imports
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import glob
import re

##
## Helper functions
##

def prepare_source_and_target_nonrigid_3d(source_filename,
                                          target_filename,
                                          voxel_size=5.0):
    """
    Import data from FIJI-created .csv files using pandas and normalize

    Output: 2 open3d PointCloud objects
    """
    source = o3.geometry.PointCloud()
    target = o3.geometry.PointCloud()

    # Read a dataframe, not just a text file
    df1 = pd.read_csv(target_filename)
    df2 = pd.read_csv(source_filename)
    target_np = df1[['XM', 'YM', 'ZM']].to_numpy()
    source_np = df2[['XM', 'YM', 'ZM']].to_numpy()
    # Test: normalize each column
    target_np = target_np / target_np.max(axis=0)
    source_np = source_np / source_np.max(axis=0)

    source.points = o3.utility.Vector3dVector(source_np)
    target.points = o3.utility.Vector3dVector(target_np)
    source = source.voxel_down_sample(voxel_size=voxel_size)
    target = target.voxel_down_sample(voxel_size=voxel_size)
    print(source)
    print(target)
    return source, target

def correspondence_from_transform(tf_param, source, target):
    """
    From the learned registration and the source and target distributions,
    calculate the closest neighbor to determine correspondence

    Uses sklearn.neighbors
    """
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)

    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)

    target_loc, result_loc = cv(target), cv(result)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target_loc)
    distances, indices = nbrs.kneighbors(result_loc)
    # print("Fitting {} transformed points to {} target points".format(len(result_loc), len(target_loc)))

    return indices

def save_indices(indices, fname=None):
    """ Saves indices (csv) using a standard format """

    df = pd.DataFrame({'indices':np.ndarray.flatten(indices)})
    # df = pd.DataFrame.from_records({'indices':indices})
    if fname is None:
        fname = 'test_bcpd_indices.csv'
    df.to_csv(fname, header=False)

    return fname

def save_indices_DLC(indices, centroid_fnames,
                     fname=None,
                     scorer='Charlie'):

    num_neurons, num_tracked_pairs = np.shape(indices)

    # Original centroid statistics have the XYZ locations
    centroid_dfs = []
    centroid_nums = []
    for fname in centroid_fnames:
        # Get the number in the filename
        this_num = re.findall(r'\d+', fname)
        centroid_nums.append(int(this_num[0]))
        centroid_dfs.append(pd.read_csv(fname))
    print(centroid_nums)
    # print(type(centroid_nums[0]))
    dataFrame = None
    coords = np.empty((len(centroid_dfs),3,))

    output_path='.'
    # Get list of images
    #  Instead of looking at all image files, only parse the processed centroids
    # imlist=[]
    # imlist.extend([fn for fn in glob.glob(os.path.join(output_path,'*.csv')) if ('Statistics' in fn)])

    if len(centroid_fnames)==0:
        print("No images found; aborting")
        return
    else:
        print("{} images found".format(len(centroid_fnames)), centroid_fnames)

    # index = np.sort(imlist)
    # print(index)
    print('Working on folder: {}'.format(os.path.split(str(output_path))[-1]))
    print("Note: this does not have the exact DLC format, but is specific to Linux")
    # Note: only works for single-digit indexed images

    # Define output for DLC on cluster
    subfoldername = 'test_100frames.ome'# TODO
    # TODO: hardcode linux filesep
    # Use the numbers from the centroid files
    print(centroid_nums)
    relativeimagenames=['/'.join(('labeled-data',subfoldername,'img{}.tif'.format(i))) for i in centroid_nums]
    print(relativeimagenames)

    # Build correctly DLC-formatted dataframe
    for i_neuron in range(num_neurons):
        bodypart = 'neuron{}'.format(i_neuron)

        # Get the index of THIS neuron in each file
        target_ind = [indices[i_neuron, i_target] for i_target in range(num_tracked_pairs)]
        ind_in_files = [i_neuron]
        ind_in_files.extend(target_ind)
        # ind_in_files = [i_neuron, indices[i_neuron][0]]
        # print("Tracked neuron from {} (source) to {} (target)".format(ind_in_files[0], ind_in_files[1]))

        # Get xyz coordinates for one neuron, for all files
        # print(centroid_dfs[0]['X'])
        for i2, df in enumerate(centroid_dfs):
            i3 = ind_in_files[i2] # The neuron index for this file
            coords[i2,:] = np.array([df['X'][i3], df['Y'][i3], df['Z'][i3]])
            # print("Coordinates for neuron {}, file {}".format(i, i2), coords[i2,:])

        index = pd.MultiIndex.from_product([[scorer], [bodypart],
                                            ['x', 'y', 'z']],
                                            names=['scorer', 'bodyparts', 'coords'])

        frame = pd.DataFrame(coords, columns = index, index = relativeimagenames)
        dataFrame = pd.concat([dataFrame, frame],axis=1)

    dataFrame.to_csv(os.path.join(output_path,"CollectedData_" + scorer + ".csv"))
    dataFrame.to_hdf(os.path.join(output_path,"CollectedData_" + scorer + '.h5'),'df_with_missing',format='table', mode='w')

##
## Initialize and align
##

if __name__ == "__main__":
    # Visualization
    to_plot = False

    # Get list of processed frames in folder
    output_path='.'
    # target_fnames=[]
    target_fnames = [fn for fn in glob.glob(os.path.join(output_path,'*.csv')) if ('Statistics' in fn)]

    # Choose a source frame; the rest are targets
    source_index = 3
    source_fname = target_fnames.pop(source_index)
    print("Source filename: ", source_fname)
    print("Target fnames: ", target_fnames)

    # Loop over the pairs
    all_indices = None
    for t_fname in target_fnames:
        # All the same source
        source, target = prepare_source_and_target_nonrigid_3d(source_fname,
                                                               t_fname,
                                                               0.005)
        if to_plot:
            cbs = [callbacks.Open3dVisualizerCallback(source, target)]
        else:
            cbs = []
        # Do BCPD
        print("Note: the value of w may need to be tweaked")
        tf_param = bcpd.registration_bcpd(source, target, w=1e-12,
                                          #gamma=10.0, #lmd=0.2, #k = 1e2,
                                          maxiter=100,
                                          callbacks=cbs)
        # Compute correspondence
        indices = correspondence_from_transform(tf_param, source, target)
        # save_indices(indices)
        if all_indices is None:
            all_indices = indices
        else:
            # print(all_indices.shape)
            # print(indices.shape)
            np.hstack((all_indices, indices))


    ## directly output in DLC format
    all_fnames = [source_fname].extend(target_fnames)
    print(all_fnames)
    save_indices_DLC(indices, all_fnames)
