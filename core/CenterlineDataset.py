import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm
import os
import tensorflow as tf
import random

def read_vtk(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def centerline_to_pd(centerline):
    # print(centerline.GetPointData())
    centerline_df = dict()

    for i in range(centerline.GetPointData().GetNumberOfArrays()):
        if centerline.GetPointData().GetArray(i).GetNumberOfComponents() == 1:
            centerline_df[centerline.GetPointData().GetArrayName(i)] = vtk_to_numpy(centerline.GetPointData().GetArray(i))
        else:
            centerline_df[centerline.GetPointData().GetArrayName(i)] = np.linalg.norm(vtk_to_numpy(centerline.GetPointData().GetArray(i)),axis=1)

    centerline_df = pd.DataFrame.from_dict(centerline_df)
    return centerline_df

def load_centerline_data(path, abscissas_array_name="Abscissas", centerlineids_array_name="CenterlineIds",features=[]):
    # load vtk data
    centerline = read_vtk(path)

    # extract the longest centerline
    longest_centerlineid = 0
    longest_centerline_length = 0

    thresholdFilter = vtk.vtkThreshold()
    thresholdFilter.SetInputData(centerline)
    thresholdFilter.SetInputArrayToProcess(0, 0, 0,"vtkDataObject::FIELD_ASSOCIATION_CELLS",centerlineids_array_name)
    for i in range(int(centerline.GetCellData().GetArray(centerlineids_array_name).GetRange()[1])):
        thresholdFilter.ThresholdBetween(i,i)
        thresholdFilter.Update()
        centerline_extracted = thresholdFilter.GetOutput()
        length = centerline_extracted.GetPointData().GetArray(abscissas_array_name).GetRange()[1] - \
            centerline_extracted.GetPointData().GetArray(abscissas_array_name).GetRange()[0]

        if length > longest_centerline_length:
            longest_centerlineid = i
            longest_centerline_length =  length

    thresholdFilter.ThresholdBetween(longest_centerlineid,longest_centerlineid)
    thresholdFilter.Update()
    centerline_main = thresholdFilter.GetOutput()

    centerline_main_pd = centerline_to_pd(centerline_main)
    centerline_main_selected = dict()
    centerline_main_selected[abscissas_array_name] = centerline_main_pd[abscissas_array_name]
    for feature in features:
        centerline_main_selected[feature] = centerline_main_pd[feature]

    # abscissas normalization
    abscissas_0 = centerline_main_selected[abscissas_array_name].iloc[0]
    centerline_main_selected[abscissas_array_name] = centerline_main_selected[abscissas_array_name].apply(lambda x: x-abscissas_0)

    centerline_main_interp = dict()
    centerline_main_interp[abscissas_array_name] = np.arange(centerline_main_selected[abscissas_array_name].iloc[0],centerline_main_selected[abscissas_array_name].iloc[-1], 0.5)
    for feature in features:
        if feature == abscissas_array_name:
            continue

        f = interpolate.interp1d(centerline_main_selected[abscissas_array_name],centerline_main_selected[feature])
        centerline_main_interp[feature] = f(centerline_main_interp[abscissas_array_name])

    centerline_main_interp = pd.DataFrame.from_dict(centerline_main_interp)
    # centerline_main_interp = centerline_main_interp.agg({feature: lambda x : ','.join(x.astype(str)) for feature in selected_features})
    centerline_main_interp = centerline_main_interp.agg({feature: lambda x : list(x) for feature in features})

    return centerline_main_interp


class CenterlineDataset(object):
    """
    load centerline-label pair for training, testing and inference.
    Args:
        data_dir (string): Path to data directory.
        centerline_filename (string): Filename of centerline vtp data.
        label_filename (string): Filename of target label data.
        train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.
    """
    
    def __init__(self,
        data_dir="",
        centerline_filename="centerline.vtp",
        label_filename="label.csv",
        case_column_name="case",
        class_names=[],
        centerline_features=[],
        absciass_array_name = "Abscissas",
        centerlineids_array_name = "CenterlineIds"
        ):
        super().__init__()

        # init membership variables
        self.data_dir = data_dir
        self.centerline_filename = centerline_filename
        self.label_filename = label_filename
        self.case_column_name = case_column_name
        self.class_names = class_names
        self.centerline_features = centerline_features
        self.abscissas_array_name = absciass_array_name
        self.centerlineids_array_name = centerlineids_array_name

    def get_dataset(self):
        # read labels from csv file
        self.label_df = pd.read_csv(self.label_filename, converters={self.case_column_name: lambda x: str(x)})

        data_list = os.listdir(self.data_dir)
        random.shuffle(data_list)

        self.dataset = tf.data.Dataset.from_tensor_slices(data_list)
        self.dataset = self.dataset.map(lambda case: tuple(tf.py_function(
            func=self.input_parser, inp=[case], Tout=[tf.float32, tf.int32, tf.int32])),
            num_parallel_calls=2
        )
        self.data_size = len(data_list)

        return self.dataset

    def input_parser(self,case):
        case = case.numpy().decode("utf-8")

        data = load_centerline_data(
            os.path.join(self.data_dir,case,self.centerline_filename),
            abscissas_array_name=self.abscissas_array_name,
            centerlineids_array_name=self.centerlineids_array_name,
            features=self.centerline_features
            )

        seq_len = len(data[0])

        # get the associate label
        label = self.label_df.loc[self.label_df[self.case_column_name]==case].iloc[0][self.class_names].values
        label = label.astype(np.int32)

        return data, seq_len, label