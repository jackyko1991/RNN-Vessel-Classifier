from typing import Sequence
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from scipy import interpolate
from sgt import SGT
import os
from tqdm import tqdm
import tensorflow as tf

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

def load_single_data(path, selected_features=[]):
    # load vtk data
    centerline = read_vtk(path)

    # print(centerline)

    # extract the longest centerline
    longest_centerlineid = 0
    longest_centerline_length = 0

    thresholdFilter = vtk.vtkThreshold()
    thresholdFilter.SetInputData(centerline)
    thresholdFilter.SetInputArrayToProcess(0, 0, 0,"vtkDataObject::FIELD_ASSOCIATION_CELLS","CenterlineIds_average")
    for i in range(int(centerline.GetCellData().GetArray("CenterlineIds_average").GetRange()[1])):
        thresholdFilter.ThresholdBetween(i,i)
        thresholdFilter.Update()
        centerline_extracted = thresholdFilter.GetOutput()
        length = centerline_extracted.GetPointData().GetArray("Abscissas_average").GetRange()[1] - \
            centerline_extracted.GetPointData().GetArray("Abscissas_average").GetRange()[0]

        if length > longest_centerline_length:
            longest_centerlineid = i
            longest_centerline_length =  length

    thresholdFilter.ThresholdBetween(longest_centerlineid,longest_centerlineid)
    thresholdFilter.Update()
    centerline_main = thresholdFilter.GetOutput()

    centerline_main_pd = centerline_to_pd(centerline_main)
    centerline_main_selected = centerline_main_pd[selected_features]

    # abscissas normalization
    abscissas_0 = centerline_main_selected["Abscissas_average"].iloc[0]
    centerline_main_selected["Abscissas_average"] = centerline_main_selected["Abscissas_average"].apply(lambda x: x-abscissas_0)

    centerline_main_interp = dict()
    centerline_main_interp["Abscissas_average"] = np.arange(centerline_main_selected["Abscissas_average"].iloc[0],centerline_main_selected["Abscissas_average"].iloc[-1], 0.5)
    for feature in selected_features:
        if feature == "Abscissas_average":
            continue

        f = interpolate.interp1d(centerline_main_selected["Abscissas_average"],centerline_main_selected[feature])
        centerline_main_interp[feature] = f(centerline_main_interp["Abscissas_average"])

    centerline_main_interp = pd.DataFrame.from_dict(centerline_main_interp)
    # centerline_main_interp = centerline_main_interp.agg({feature: lambda x : ','.join(x.astype(str)) for feature in selected_features})
    centerline_main_interp = centerline_main_interp.agg({feature: lambda x : list(x) for feature in selected_features})

    return centerline_main_interp

def input_parser(file_path,results):
    file_path = file_path.numpy().decode("utf-8")
    print(file_path,results)

    selected_features = [
        "Abscissas_average",
        "Curvature_average",
        "Radius_average",
        "Torsion_average",
        "p(mmHg)_average",
        "U_average",
        "vorticity_average",
        "wallShearStress_average",
    ]
    single_data = load_single_data(file_path,selected_features=selected_features)
    seq_len = len(single_data[0])

    return single_data, seq_len, results

# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
    dtype=tf.float32,
    shape=shape,
    initializer=initer)

def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
        dtype=tf.float32,
        initializer=initial)

def main():
    data_dir = "./data"

    selected_features = [
        "Abscissas_average",
        "Curvature_average",
        "Radius_average",
        "Torsion_average",
        "p(mmHg)_average",
        "U_average",
        "vorticity_average",
        "wallShearStress_average",
        ]

    data_list = []
    for data_name in tqdm(os.listdir(data_dir)):
        data_list.append(os.path.join(data_dir,data_name))

    # for data_name in tqdm(os.listdir(data_dir)):
    #     single_data = load_single_data(os.path.join(data_dir,data_name),selected_features=selected_features)
    #     single_data["filename"] = data_name
    #     data_list.append(single_data)

    # data = pd.DataFrame(data_list)
    # col = data.pop("filename")
    # data.insert(0, col.name, col)
    # print(data)

    tf_dataset = tf.data.Dataset.from_tensor_slices((data_list,[0,1]))
    tf_dataset = tf_dataset.map(lambda file_path, results: tuple(tf.py_function(
        func=input_parser, inp=[file_path,results], Tout=[tf.float32,tf.int32,tf.int32])),
        num_parallel_calls=2)

    tf_dataset = tf_dataset.padded_batch(2, padded_shapes=([None,None],[],[]),drop_remainder=False)
    iterator = tf_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # network definition
    sequence_placeholder = tf.placeholder(tf.float32, [None,None,8])
    output_placeholder = tf.placeholder(tf.int32, [None])
    sequence_length_placeholder = tf.placeholder(tf.int32,[None])
    outputs_op, state_op = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.LSTMCell(25),
        sequence_placeholder,
        dtype=tf.float32,
        sequence_length=sequence_length_placeholder
    )

    # put time axis to first rank and select last frame
    last_output_op = state_op.h
    dense = tf.layers.dense(last_output_op,units=10,activation=None)
    logits_op = tf.layers.dense(dense,units=1,activation=None)

    sigmoid_op = tf.sigmoid(logits_op)
    loss_op = tf.losses.log_loss(labels=output_placeholder,predictions=sigmoid_op[:,0])

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5)
    train_op = optimizer.minimize(
        loss=loss_op,
        global_step=tf.train.get_or_create_global_step()
        )

    # training cycle
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        sess.run(iterator.initializer)
        series, seq_len, label = sess.run(next_element)
        # swap axes 1 and 2 to fit conventional rnn input
        series = np.swapaxes(series, axis1=1,axis2=2)

        print(series[0])
        print("series:{}, seq_len:{},  label: {}".format(series.shape,seq_len.shape,label.shape))

        for i in range(1000):
            o, s,last_output, sig, loss,_ = sess.run([outputs_op,state_op, last_output_op,sigmoid_op, loss_op, train_op], feed_dict={sequence_placeholder: series, output_placeholder: label, sequence_length_placeholder: seq_len})
            # print("o1")
            # print(o1)
            # print(np.shape(o1))
            # print("s1")
            # print(s1)
            # print(np.shape(s1))
            print("last output: {}".format(last_output))
            print("sigmoid: {}".format(sig))
            print("loss: {:.4f}".format(loss))
    
if __name__=="__main__":
    main()