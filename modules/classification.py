# encoding: utf-8
import numpy as np
from sklearn import svm
import os
import load_npy
from itertools import izip
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
import copy
import cPickle
import sys

def discriminate_dataset(working_directory):

    """Returns a list with all full paths for all the files that shall be processed within a working directory."""
    all_feature_files = []
    for dirpath, dirname, file_names in os.walk(working_directory):
        dirpath = dirpath.replace('\\', '/')
        for a_file in file_names:
            if a_file.endswith('_features.npy') or a_file.endswith('_test.npy'):
                all_feature_files.append(os.path.join(dirpath, a_file))
    return all_feature_files

def load_data(working_directory, feature_file, keep_features, exclude_circumscribed = True, class_balance = True):

    """Load and return as numpy array the selected calculated features from the specified feature file. If no positive labels
    are found in an image this is possibly due to registration errors and nothing is returned.

    :param string working_directory:
    :param string feature_file: full path to feature file
    :param string keep_features: full path to a text file containing all features (in lines) that should be loaded
    :param bool exclude_circumscribed: If set to True, pixels where circumscribed ROIs do not correspond to actual label values are disregarded. This only has an effect to ROI based calculation of C2 and C5 features. Default value is True.
    :param class_balance: If set to True, majority class random undersampling is performed and a balanced dataset is returned. Default value is True.

    :returns: numpy array with feature vectors for all the pixels

    :returns: numpy array with the labels corresponding to the vectors in the first returned array
    """
    print('loading data from '+ working_directory)
    labels = load_npy.load_file(working_directory, feature_file, ['ROI_GS'], return_array=True)
    if np.sum(labels)==0:
        print("no positive labels in " + feature_file)
        return np.array([]), np.array([], dtype=np.uint8)
    os.chdir(working_directory)
    all_features_to_keep = []
    with open(keep_features, 'r') as features:
        for line in features:
            all_features_to_keep.append(line.rstrip())
    dataset = load_npy.load_file(working_directory, feature_file, all_features_to_keep, return_array=True)
    # doing undersampling here to save ram
    if exclude_circumscribed:
        circumscribed_labels = load_npy.load_file(working_directory, feature_file, ['ortho_ROI_GS'], return_array=True)
        d1 = []
        l1 = []
        for i in range(1, labels.shape[0]):
            if labels[i, 0] == circumscribed_labels[i, 0]:
                d1.append(dataset[i, :])
                l1.append(labels[i, 0])
        d1 = np.array(d1)
        l1 = np.array(l1, dtype=np.uint8)
        dataset = d1
        labels = l1
    if class_balance:
        rus = RandomUnderSampler()
        dataset, labels = rus.fit_sample(dataset, labels)
    return dataset, labels

def load_dataset(working_directory, keep_features, multi_case = False):

    """Returns dataset with all selected features for all feature files within a working directory. (The entire dataset,
     where data points refer to vectors that correspond to pixels for evey image.)

    :param string working_directory:
    :param string keep_features: full path to a text file containing all features (in lines) that should be loaded

    :return:
    """
    dataset_files = discriminate_dataset(working_directory)
    patient_directories = []
    for a_file in dataset_files:
        a_directory = a_file.replace('\\', '/')
        a_directory = a_directory.split('/')
        patient_directories.append('/'.join(a_directory[:-1]))
    for a_file, a_directory in izip(dataset_files, patient_directories):
        data_temp, labels_temp = load_data(a_directory, a_file, keep_features)
        try:
            #MERK
            data = np.append(data, data_temp, axis=0)
            labels = np.append(labels, labels_temp, axis=0)
        except:
            #MERK
            if data_temp.size != 0:
                data = data_temp
                labels = labels_temp
    if 'data' in locals():
        if not multi_case:
            return data, labels
        else:
            return data, labels, True
    else: 
        if multi_case:
            return [], [], False
        else:
            return [], []


def get_all_features(working_directory, features_switch):

    """Save a keep features text file inside the working directory. This contains in each line a feature name
    that should be loaded. This file is based on only one patient directory within the working directory.
    Therefore, it is assumed that the same features have been calculated for all patients, otherwise an exception will
    be raised when trying to load features that have not been calculated.

    :param string working_directory:
    :param list features_switch: Which of the features to retrieve (from families C1, C2, C3, C4, C5).It is important to feed the features switch in order (eg never give f3, f5,f1 but f1, f3, f5)

    :returns: string which is full path to keep features text file
    """
    # make a keep_features file with all reported features. It is a single file, so all patient folders must be consistent
    if type(working_directory) == list:
        working_directory = working_directory[0]
    for i in range(len(features_switch)):
        features_switch[i] = features_switch[i].replace('f','C') + '.txt'

    all_features_file_path = os.path.join(working_directory, 'all_features.txt')
    if not os.path.isfile(all_features_file_path):
        #Haralick 14 is buggy!
        exclude_lines = ['dummy\n', 'ROI_GS\n', 'ortho_ROI_GS\n', 'Haralick_range_14\n', 'Haralick_mean_14\n']
        feature_files = []
        names_found = False
        for dirpath, dirname, file_names in os.walk(working_directory):
            dirpath = dirpath.replace('\\', '/')
            for a_file in file_names:
                pick_feature_family = False
                for a_feature_family in features_switch:
                    if a_file.endswith(a_feature_family):
                        pick_feature_family = True
                        break
                if a_file.startswith('features_names') and pick_feature_family:
                    names_found = True
                    feature_files.append(os.path.join(dirpath, a_file))
            if names_found:
                break
        os.chdir(working_directory)
        with open('all_features.txt', 'w') as outfile:
            for fname in feature_files:
                with open(fname) as infile:
                    for line in infile:
                        if line not in exclude_lines:
                            outfile.write(line)
    return all_features_file_path

def reduction_analysis(working_directory, case_name = 'undefined', features_switch = ['f1', 'f2', 'f3', 'f4', 'f5'], multi_dir = False):

    """Perform PCA analysis and save the eigenvalues in descending order in text files. This analysis is done both on
    the original data and on data that has been scaled with an outlier robust scaler, which selects values within the
    25 and 75th percentile of all values. (see documentation of sklearn.preprocessing.RobustScaler)

    :param string working_directory:
    :param string file_name: name of the output text file
    :param list features_switch: Which of the features to retrieve (from families C1, C2, C3, C4, C5).It is important to feed the features switch in order (eg never give f3, f5,f1 but f1, f3, f5). Default value is ['f1', 'f2', 'f3', 'f4', 'f5'].
    """
    keep_features = get_all_features(working_directory, features_switch)
    if multi_dir:
        data, labels = multi_directory_data(working_directory, keep_features)
        working_directory = working_directory[0]
    else:
        data, labels = load_dataset(working_directory, keep_features)
    pca_unscaled = PCA()
    pca_unscaled.fit(data)
    l1 = pca_unscaled.explained_variance_
    scaler = RobustScaler()
    data_sc = scaler.fit_transform(data, labels)
    pca_scaled = PCA()
    pca_scaled.fit(data_sc)
    l2 = pca_scaled.explained_variance_
    # for feature selection do on scaled data as
    rfe_5 = SelectKBest(k='all')
    rfe_5.fit(data, labels)
    l3_unscaled = rfe_5.scores_
    rfe_5.fit(data_sc, labels)
    l3_scaled = rfe_5.scores_

    ######################
    os.chdir(working_directory)
    file_name_ending = '_reduction_analysis_results.txt'
    file_name = case_name + file_name_ending
    with open(file_name, 'w') as outstream:
        outstream.write('PCA explained variance for not scaled data \n')
        for a_value in l1:
            outstream.write(str(a_value) + '\n')
        outstream.write("------------------------")
        outstream.write('PCA explained variance for scaled data (Robust scaler, data over the 25th and 75th perc \n')
        for a_value in l2:
            outstream.write(str(a_value) + '\n')
    #################################################################
    # play = PCA(n_components=15)
    # d2 = play.fit_transform(data_sc)
    # svc = svm.SVC()
    # scores = cross_val_score(svc, d2, labels, cv=5)
    ################################################################
    #return



def classify_svm(working_directory, keep_features, no_xval = 5, no_features = 15, case_name = 'undefined', multi_dir = False):

    """Scale the data with RobustScaler (see documentation of sklearn.preprocessing.RobustScaler) as SVM is sensitive to the data scales.
    Then apply PCA on the new dataset. Train and evaluate a svm model with cross validation. Save the results of each
    cross validation fold in a text file within the working directory.

    :param string working_directory:
    :param string keep_features: full path to a text fi   le containing all features (in lines) that should be loaded
    :param string file_name: name of the file where the results are saved
    :param int no_xval: number of cross validation folds. Default is 5.
    :param int no_features: number of features that are kept during PCA. Default is 15. (Usually many more features are present).

    :return: list of cross validation folds accuracies
    """
    if multi_dir:
        data, labels = multi_directory_data(working_directory, keep_features)
        working_directory = working_directory[0]
    else:
        data, labels = load_dataset(working_directory, keep_features)
    scaler = RobustScaler()
    dataset_sc = scaler.fit_transform(data, labels)
    #     svm_classify.fit(X_train, y_train)
    play = PCA(n_components=no_features)
    dataset_sc_2 = play.fit_transform(dataset_sc)
    svm_classify = svm.SVC(cache_size=2000)
    scores = cross_val_score(svm_classify, dataset_sc_2, labels, cv=no_xval)
    file_name_ending = '_PCA_xval_svm.txt'
    file_name = case_name + file_name_ending
    os.chdir(working_directory)
    with open(file_name, 'w') as outstream:
        for a_score in scores:
            outstream.write(str(a_score) + '\n')
    return scores

def classify_RF(working_directory, keep_features, no_xval = 5, no_trees = 60, case_name = 'undefined_', multi_dir = False):

    """Train and evaluate a random forest model with cross validation. Save the results of each
    cross validation fold in a text file within the working directory.

    :param string working_directory:
    :param string keep_features: full path to a text file containing all features (in lines) that should be loaded
    :param string file_name: name of the file where the results are saved
    :param int no_xval: number of cross validation folds. Default is 5.
    :param int no_trees: number of trees of the random forest. Default is 60.

    :return: list of cross validation folds accuracies
    """
    if multi_dir:
        data, labels = multi_directory_data(working_directory, keep_features)
        working_directory = working_directory[0]
    else:
        data, labels = load_dataset(working_directory, keep_features)
    clf = RandomForestClassifier(max_features=None)
    scores = cross_val_score(clf, data, labels, cv=no_xval)
    os.chdir(working_directory)
    #file_name = 'xval_on_RF_' +str(no_trees)+ '.txt'
    file_name_ending = '_xval_on_RF.txt'
    file_name = case_name + file_name_ending
    with open(file_name, 'a') as outstream:
        outstream.write('For number of trees ' + str(no_trees) + '\n')
        for a_score in scores:
            outstream.write(str(a_score) + '\n')
        outstream.write('---------------------')
    return scores

def RF_trees_number_optimization(working_directory, keep_features, no_xval = 5, case_name = 'undefined_', multi_dir = False):

    """call classify RF recursively for several numbers of trees of the random forest"""
    for trees_no in range(10, 80, 10):
        classify_RF(working_directory, keep_features, no_xval=no_xval, no_trees=trees_no, case_name=case_name, multi_dir=multi_dir)

def classify_svm_independent(training_directory, test_directory, keep_features, no_features = 15, case_name = 'undefined', multi_dir = False):

    """Train a SVM model and evaluate on an independent dataset.

    Scale the data with RobustScaler (see documentation of sklearn.preprocessing.RobustScaler) as SVM is sensitive to the data scales.
    Then apply PCA on the new dataset. Train and evaluate a svm model. Save the resulting accuracy in a text file within the
    working directory.

    :param string/list training_directory: root directory or directories of the training set
    :param string test_directory: root directory of the test set
    :param string keep_features: full path to a text file containing all features (in lines) that should be loaded
    :param string case_name: name of the file where the results are saved. It is not the full name, as the file will
    always end with _PCA_independent_svm.txt. This is the initial part of the file name, that should be indicative
    of the case (e.g. peripheral_zone_all_features_).
    :param bool multi_dir: Must be true if training_directory is a list of directories. This means that data under multiple
    roots can be used as a training set and a model on multi-clinic data can be trained. Default is False.
    :param int no_features: number of features that are kept during PCA. Default is 15. (Usually many more features are present).

    :return: classification accuracy
    """
    if multi_dir:
        X_train, y_train = multi_directory_data(training_directory, keep_features)
        fusion_degree = len(training_directory)
        training_directory = training_directory[0]
    else:
        X_train, y_train = load_dataset(training_directory, keep_features)
    X_test, y_test = load_dataset(test_directory, keep_features)
    if not list(X_test):
        os.chdir(training_directory)
        os.chdir('../')
        print('no labels found in ' + test_directory)
        file_name_ending = '_PCA_independent_svm.txt'
        file_name = case_name + file_name_ending
        if not os.path.isdir('./results'):
            os.mkdir('./results')
        os.chdir('./results')
        with open(file_name, 'a') as outstream:
            outstream.write('no labels found in ' + test_directory)
        return
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.fit_transform(X_test, y_test)
    #     svm_classify.fit(X_train, y_train)
    play = PCA(n_components=no_features)
    play.fit(X_train)
    X_train = play.transform(X_train)
    X_test = play.transform(X_test)
    svm_classify = svm.SVC(cache_size=2000)
    clf = svm_classify.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    file_name_ending = '_PCA_independent_svm.txt'
    file_name = case_name + file_name_ending
    os.chdir(training_directory)
    os.chdir('../')
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    os.chdir('./results')
    if not multi_dir:
        with open(file_name, 'w') as outstream:
            outstream.write(str(accuracy) + '\n')
    else:
        with open(file_name, 'a') as outstream:
            outstream.write(str(accuracy) + ' for fusion degree ' + str(fusion_degree) + '\n')
    return accuracy

def classify_RF_independent(training_directory, test_directory, keep_features, no_trees = 10, case_name = 'undefined_', multi_dir = False):

    """Train a Random Forest model and evaluate on an independent dataset.  Save the resulting accuracy in a text
    file within the working directory.

    :param string/list training_directory: root directory or directories of the training set
    :param string test_directory: root directory of the test set
    :param string keep_features: full path to a text file containing all features (in lines) that should be loaded
    :param string case_name: name of the file where the results are saved. It is not the full name, as the file will always end with _independent_test_RF.txt. This is the initial part of the file name, that should be indicative of the case (e.g. peripheral_zone_all_features_).
    :param int no_trees: Number of trees used by the random forest. Default is 70.
    :param bool multi_dir: Must be true if training_directory is a list of directories. This means that data under multiple roots can be used as a training set and a model on multi-clinic data can be trained. Default is False.

    :return: classification accuracy
    """
    if multi_dir:
        X_train, y_train = multi_directory_data(training_directory, keep_features)
        fusion_degree = len(training_directory)
        training_directory = training_directory[0]
    else:
        X_train, y_train = load_dataset(training_directory, keep_features)
    X_test, y_test = load_dataset(test_directory, keep_features)
    if not list(X_test):
        os.chdir(training_directory)
        os.chdir('../')
        print('no labels found in ' + test_directory)
        file_name_ending = '_independent_test_RF.txt'
        file_name = case_name + file_name_ending
        if not os.path.isdir('./results'):
            os.mkdir('./results')
        os.chdir('./results')
        with open(file_name, 'a') as outstream:
            outstream.write('no labels found in ' + test_directory)
        return
    clf = RandomForestClassifier(max_features=None)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    importance = clf.feature_importances_
    os.chdir(training_directory)
    os.chdir('../')
    #file_name = 'xval_on_RF_' +str(no_trees)+ '.txt'
    file_name_ending = '_independent_test_RF.txt'
    file_name = case_name + file_name_ending
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    os.chdir('./results')
    all_features_to_keep = []
    with open(keep_features, 'r') as features:
        for line in features:
            all_features_to_keep.append(line.rstrip())
    if not multi_dir:
        with open(file_name, 'a') as outstream:
            outstream.write(str(accuracy) + '\n' + '------------' + '\n')
            for i in range(len(importance)):
                outstream.write(str(importance[i]) + '&'+ all_features_to_keep[i] + '\n')
    else:
        with open(file_name, 'a') as outstream:
            outstream.write(str(accuracy) + ' for fusion degree ' + str(fusion_degree) + '\n'+ '------------' + '\n')
            for i in range(len(importance)):
                outstream.write(str(importance[i]) + '\n')

def RF_trees_opt(train_directory, test_directory, keep_features, no_trees = 60, case_name = 'undefined_'):

    for trees_no in range(10, 80, 10):
        classify_RF_independent(train_directory, test_directory, keep_features, no_trees, case_name)

def multi_directory_data(dir_list, keep_features):

    """Load and return data as an array of vectors and the corresponding labels from multiple directory roots.
    These arrays and labels can be pipelined to any sklearn model.

    :param list dir_list: list of full root directories of datasets (the corresponding working directories, where all feature files are saved. Not the actual raw dataset directories.)
    :param string keep_features: full path to a text file containing all features (in lines) that should be loaded

    :returns: numpy array with feature vectors for all the pixels of all feature files under dir_list entries

    :returns: numpy array with the labels corresponding to the vectors in the above array
    """
    for a_directory in dir_list:
        temp_data, temp_labels, has_data = load_dataset(a_directory, keep_features, multi_case = True)
        try:
            if has_data:
                data = np.append(data, temp_data, axis = 0)
                labels = np.append(labels, temp_labels, axis = 0)
        except:
            if has_data:
                data = temp_data
                labels = temp_labels
    return data, labels

def dataset_fusion(dataset1, dataset2, patient,file_name='undefined_'):

    dataset2_directories = []
    for a_dir, _, _ in os.walk(dataset2):
        found_feat_file = False
        for a_file in os.listdir(a_dir):
            if a_file.endswith('features.npy'):
                found_feat_file = True
                break
        if found_feat_file and (a_dir != patient):
            dataset2_directories.append(a_dir)

    keep_file = get_all_features(dataset1, ['f1','f2','f3','f4','f5'])
    #classify_svm_independent(dataset1, patient, keep_file, case_name=file_name)
    classify_RF_independent(dataset1, patient, keep_file, case_name=file_name)
    for i in range(1, len(dataset2_directories)):
        train_set = [dataset1]+dataset2_directories[:i]
        #classify_svm_independent(train_set, patient, keep_file, case_name='fuse_data_TCIA_train_k', multi_dir=True)
        classify_RF_independent(train_set, patient, keep_file, case_name=file_name, multi_dir=True)

def independent_patient_xval(working_directory, keep_features, case_name):

    """Leave one out cross validation. Training set is the total of pixels of all -1 patients,
    test set is set of pixels of 1 patient."""

    os.chdir(working_directory)
    all_patients = discriminate_dataset(working_directory)
    patients_directories = []
    #strip all patient files, crappy design
    for a_patient in all_patients:
        stripped_list = a_patient.split('/')
        a_patient_directory = ''
        for kk in stripped_list[:-1]:
            a_patient_directory += kk + '/'
        patients_directories.append(a_patient_directory)
    all_patients = patients_directories
    for i in range(len(all_patients)):
        patients_copy = copy.deepcopy(all_patients)
        test_set = patients_copy.pop(i)
        train_set = patients_copy
        classify_RF_independent(train_set, test_set, keep_features=keep_features, case_name=case_name, multi_dir=True)
        classify_svm_independent(train_set, test_set, keep_features=keep_features, case_name=case_name, multi_dir=True)

def save_models(working_directory, keep_features, case_name):

    """save a model according to case name"""
    X_train, y_train = load_dataset(working_directory, keep_features)
    os.chdir(working_directory)
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    os.chdir('./models')
    clf_RF = RandomForestClassifier(max_features=None)
    clf_RF.fit(X_train, y_train)
    # clf_RF_name = case_name + '_trained_RF.pkl'
    # save model trained on all data for evaluation on the other dataset
    with open('RF.pkl', 'wb') as fid:
        cPickle.dump(clf_RF, fid)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    play = PCA(n_components=15)
    play.fit(X_train)
    X_train = play.transform(X_train)
    svm_classify = svm.SVC(cache_size=2000)
    clf = svm_classify.fit(X_train, y_train)
    #clf_svm_name = case_name + '_trained_svm.pkl'
    with open('svm.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)
    with open('pca_trf.pkl', 'wb') as fid:
        cPickle.dump(play, fid)
    with open('scaler.pkl', 'wb') as fid:
        cPickle.dump(scaler, fid)

def independent_val(model, data, keep_features, case_name):

    os.chdir(model)
    os.chdir('./models')
    with open('pca_trf.pkl', 'rb') as fid:
        play = cPickle.load(fid)
    with open('svm.pkl', 'rb') as fid:
        svm_classify = cPickle.load(fid)
    with open('RF.pkl', 'rb') as fid:
        clf_RF = cPickle.load(fid)
    with open('scaler.pkl', 'rb') as fid:
        scaler = cPickle.load(fid)
    os.chdir(data)
    X_test, y_test = load_dataset(data, keep_features)
    os.chdir('../')
    accuracy = clf_RF.score(X_test, y_test)
    importance = clf_RF.feature_importances_
    # file_name = 'xval_on_RF_' +str(no_trees)+ '.txt'
    file_name_ending = '_independent_RF.txt'
    file_name = case_name + file_name_ending
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    os.chdir('./results')
    all_features_to_keep = []
    with open(keep_features, 'r') as features:
        for line in features:
            all_features_to_keep.append(line.rstrip())
    with open(file_name, 'a') as outstream:
        outstream.write(str(accuracy) + '\n' + '-----------------' + '\n')
        for i in range(len(importance)):
            outstream.write(str(importance[i]) + '&'+ all_features_to_keep[i] + '\n')
    X_test = scaler.fit_transform(X_test, y_test)
    X_test = play.transform(X_test)
    accuracy = svm_classify.score(X_test, y_test)
    file_name_ending = '_independent_svm.txt'
    file_name = case_name + file_name_ending
    with open(file_name, 'a') as outstream:
        outstream.write(str(accuracy) + '\n')


def case_split_xval(input_string):

    """used to be main, but with change of cluster can be submitted as a single job"""

    if input_string == 1:  # TCIA all features whole prostate
        wd = '/root/work/wd_TCIA_all_features_whole'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'v2_TCIA_all_features_whole_prostate'
    elif input_string == 2: # TCIA all features whole prostate sliding window
        wd = '/root/work/wd_TCIA_all_features_whole_sliding'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'v2_TCIA_all_features_whole_prostate_sliding'
    elif input_string == 3: #TCIA all features pz
        wd = '/root/work/wd_TCIA_all_features_pz'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'v2_TCIA_all_features_pz'
    elif input_string == 4:# TCIA all features pz sliding window
        wd = '/root/work/wd_TCIA_all_features_pz_sliding'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'v2_TCIA_all_features_pz_sliding'
    elif input_string == 5: # TCIA C2 C5 whole prostate
        wd = '/root/work/wd_TCIA_all_features_whole'
        feat_switch = ['f2','f5']
        case_name = 'v2_TCIA_C2_C5_whole_prostate'
    elif input_string == 6: # TCIA C2 C5 whole prostate sliding
        wd = '/root/work/wd_TCIA_all_features_whole_sliding'
        # wd = '/media/pavlos/Elements/playground/kk'
        feat_switch = ['f2', 'f5']
        case_name = 'v2_TCIA_C2_C5_whole_prostate_sliding'
    elif input_string == 7: # TCIA C2 C5 pz
        wd = '/root/work/wd_TCIA_all_features_pz'
        feat_switch = ['f2', 'f5']
        case_name = 'v2_TCIA_C2_C5_pz'
    elif input_string == 8: #TCIA C2 C5 pz sliding
        wd = '/root/work/wd_TCIA_all_features_pz_sliding'
        feat_switch = ['f2', 'f5']
        case_name = 'v2_TCIA_C2_C5_pz_sliding'
    elif input_string == 9: # PCMM all features whole prostate
        wd = '/root/work/wd_PCMM_all_features_whole'
        #wd = '/media/pavlos/Elements/playground/kkk'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'v2_PCMM_all_features_whole'
    elif input_string == 10: #PCMM all features whole prostate sliding
        wd = '/root/work/wd_PCMM_all_features_whole_sliding'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'v2_PCMM_all_features_whole_sliding'
    elif input_string == 11: #PCMM all features PZ
        wd = '/root/work/wd_PCMM_all_features_pz'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'v2_PCMM_all_features_pz'
    elif input_string == 12: #PCMM all features pz sliding
        wd = '/root/work/wd_PCMM_all_features_pz_sliding'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'v2_PCMM_all_features_pz_sliding'
    elif input_string == 13: #PCMM C2 C5 whole prostate
        wd = '/root/work/wd_PCMM_all_features_whole'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'v2_PCMM_C2_C5_features_whole'
    elif input_string == 14: # PCMM C2 C5 whole prostate sliding
        wd = '/root/work/wd_PCMM_all_features_whole_sliding'
        feat_switch = ['f2', 'f5']
        case_name = 'v2_PCMM_C2_C5_whole_sliding'
    elif input_string == 15: #PCMM C2 C5 pz
        wd = '/root/work/wd_PCMM_all_features_pz'
        feat_switch = ['f2','f5']
        case_name = 'v2_PCMM_C2_C5_pz'
    elif input_string == 16: #PCMM C2 C5 pz sliding
        wd = '/root/work/wd_PCMM_all_features_pz_sliding'
        feat_switch = ['f2', 'f5']
        case_name = 'v2_PCMM_C2_C5_pz_sliding'
    else:
        print('nobot')
        return

    keep_features = get_all_features(wd, feat_switch)
    independent_patient_xval(wd, keep_features, case_name)

def case_split_indie(input_string):


    if input_string == 17: # TCIA train PCMM test all features whole
        model = '/root/work/wd_TCIA_all_features_whole'
        data = '/root/work/wd_PCMM_all_features_whole'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'ind_TCIA_PCMM_all_features_whole_prostate'
    elif input_string == 18: ## TCIA train PCMM test all features whole prostate sliding window
        model = '/root/work/wd_TCIA_all_features_whole_sliding'
        data = '/root/work/wd_PCMM_all_features_whole_sliding'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'ind_TCIA_PCMM_all_features_whole_sliding'
    elif input_string == 19: #TCIA train PCMM test all features pz
        model = '/root/work/wd_TCIA_all_features_pz'
        data = '/root/work/wd_PCMM_all_features_pz'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'ind_TCIA_PCMM_all_features_pz'
    elif input_string == 20: #TCIA train PCMM test all features pz sliding
        model = '/root/work/wd_TCIA_all_features_pz_sliding'
        data = '/root/work/wd_PCMM_all_features_pz_sliding'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'ind_TCIA_PCMM_all_features_pz_sliding'
    elif input_string == 21: #TCIA train PCMM test C2 C5 whole
        model = '/root/work/wd_TCIA_all_features_whole'
        data = '/root/work/wd_PCMM_all_features_whole'
        feat_switch = ['f2', 'f5']
        case_name = 'ind_TCIA_PCMM_C2_C5_whole'
    elif input_string == 22: #TCIA train PCMM test C2 C5 whole sliding
        model = '/root/work/wd_TCIA_all_features_whole_sliding'
        data = '/root/work/wd_PCMM_all_features_whole_sliding'
        # model = '/media/pavlos/Elements/playground/kkk'
        # data = '/media/pavlos/Elements/playground/kz'
        feat_switch = ['f2', 'f5']
        case_name = 'ind_TCIA_PCMM_C2_C5_whole_sliding'
    elif input_string == 23: #TCIA train PCMM test C2 C5 pz
        model = '/root/work/wd_TCIA_all_features_pz'
        data = '/root/work/wd_PCMM_all_features_pz'
        feat_switch = ['f2', 'f5']
        case_name = 'ind_TCIA_PCMM_C2_C5_pz'
    elif input_string == 24: #TCIA train PCMM test C2 C5 pz sliding
        model = '/root/work/wd_TCIA_all_features_pz_sliding'
        data = '/root/work/wd_PCMM_all_features_pz_sliding'
        feat_switch = ['f2', 'f5']
        case_name = 'ind_TCIA_PCMM_C2_C5_pz_sliding'
    elif input_string == 25: # PCMM train TCIA test all features whole
        model = '/root/work/wd_PCMM_all_features_whole'
        data = '/root/work/wd_TCIA_all_features_whole'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'ind_PCMM_TCIA_all_features_whole_prostate'
    elif input_string == 26: ## PCMM train TCIA test all features whole prostate sliding window
        model = '/root/work/wd_PCMM_all_features_whole_sliding'
        data = '/root/work/wd_TCIA_all_features_whole_sliding'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'ind_PCMM_TCIA_all_features_whole_sliding'
    elif input_string == 27: #PCMM train TCIA test all features pz
        model = '/root/work/wd_PCMM_all_features_pz'
        data = '/root/work/wd_TCIA_all_features_pz'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'ind_PCMM_TCIA_all_features_pz'
    elif input_string == 28: #PCMM train TCIA test all features pz sliding
        model = '/root/work/wd_PCMM_all_features_pz_sliding'
        data = '/root/work/wd_TCIA_all_features_pz_sliding'
        feat_switch = ['f1', 'f2', 'f3', 'f4', 'f5']
        case_name = 'ind_PCMM_TCIA_all_features_pz_sliding'
    elif input_string == 29: #PCMM train TCIA test C2 C5 whole
        model = '/root/work/wd_PCMM_all_features_whole'
        data = '/root/work/wd_TCIA_all_features_whole'
        feat_switch = ['f2', 'f5']
        case_name = 'ind_PCMM_TCIA_C2_C5_whole'
    elif input_string == 30: #PCMM train TCIA test C2 C5 whole sliding
        model = '/root/work/wd_PCMM_all_features_whole_sliding'
        data = '/root/work/wd_TCIA_all_features_whole_sliding'
        feat_switch = ['f2', 'f5']
        case_name = 'ind_PCMM_TCIA_C2_C5_whole_sliding'
    elif input_string == 31: #PCMM train TCIA test C2 C5 pz
        model = '/root/work/wd_PCMM_all_features_pz'
        data = '/root/work/wd_TCIA_all_features_pz'
        feat_switch = ['f2', 'f5']
        case_name = 'ind_PCMM_TCIA_C2_C5_pz'
    elif input_string == 32: #PCMM train TCIA test C2 C5 pz sliding
        model = '/root/work/wd_PCMM_all_features_pz_sliding'
        data = '/root/work/wd_TCIA_all_features_pz_sliding'
        feat_switch = ['f2', 'f5']
        case_name = 'ind_PCMM_TCIA_C2_C5_pz_sliding'

    else:
        print('nobot')
        return

    keep_features = get_all_features(data, feat_switch)
    save_models(model, keep_features, case_name = case_name)
    independent_val(model, data, keep_features, case_name)


if __name__ == "__main__":

    input_string = int(sys.argv[1])
    if input_string == 1:
        for i in range(1,5):
            case_split_xval(i)
    elif input_string == 2:
        for i in range(5,9):
            case_split_xval(i)
    elif input_string == 3:
        for i in range(9, 13):
            case_split_xval(i)
    elif input_string == 4:
        for i in range(13, 17):
            case_split_xval(i)
    elif input_string == 5:
        for i in range(17, 21):
            case_split_indie(i)
    elif input_string == 6:
        for i in range(21, 25):
            case_split_indie(i)
    elif input_string == 7:
        for i in range(25, 29):
            case_split_indie(i)
    elif input_string == 8:
        for i in range(29, 33):
            case_split_indie(i)
    else:
        print('invalid argument')

#case_split_xval(9)

#case_split_indie(22)



#classify_svm('C:/Users/157136/Desktop/pground/wd','C:/Users/157136/Desktop/pground/keep_features_game.txt')
#get_all_features('C:/Users/157136/Desktop/pground/wdd')
#reduction_analysis('/media/pavlos/Elements/TCIA/wd2/aaa0051')
#RF_trees_number__optimization('F:/pground/wd/PC-3120951400', 'F:/wd_04/all_features.txt')
#reduction_analysis('/media/pavlos/Elements/playground/wd_sliding')
#get_all_features('/media/pavlos/Elements/TCIA/wd2', features_switch=['f5'])
#classify_RF('/media/pavlos/Elements/TCIA/wd2/aaa0051', '/media/pavlos/Elements/TCIA/wd2/aaa0053', keep_features='/media/pavlos/Elements/TCIA/wd2/all_features.txt', no_features=1)
#get_all_features(['/media/pavlos/Elements/TCIA/wd2/aaa0051',  '/media/pavlos/Elements/TCIA/wd2/aaa0053'],features_switch=['f5'])


