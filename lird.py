#!/usr/bin/pyhton

import csv
import time
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import multiclass
from sklearn import cross_validation
from sklearn import metrics
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from math import sqrt
import argparse
import exceptions


# Stores the parsed command line args
args = None

# use labelEncoder.transform() and .inverse_transform() to convert from
# letters to numbers and vice versa.
# The label encoder will be instantiated during data parsing.
labelEncoder = None

# This label will be applied by the RadiusNeighborsClassifier to detect outliers
# in the data set.
OUTLIER_LABEL = '?'

# A list of all supported SVM kernels and classifiers
SVM_KERNELS = ['linear-svc', 'linear-ovr', 'linear', 'poly', 'rbf', 'sigmoid']

def _parse_csv(filename, typeCast=int):
    '''
    Parse data from CSV and return (y,X) with y a array of labels and X array
    of observation vectors. X entries are casted using typeCast.
    Global variable labelEncoder will be instantiated
    '''
    data = csv.reader(open(filename))
    X = []
    y = []
    for row in data:
        y.append(row[0])
        X.append([typeCast(r) for r in row[1:len(row)]])

    # sklearn classifiers expect classes to be integers.
    # use preprocessing to encode the labels accordingly
    global labelEncoder
    labelEncoder = preprocessing.LabelEncoder()
    # Add an extra outlier label '?' to the list of labels so that methods like
    # RadiusNeighborsClassifier that detect outliers can give them a
    # indistinguishable label.
    labelEncoder.fit(y + [OUTLIER_LABEL]);
    y = labelEncoder.transform(y)

    return (y, X)


def __print_confusion_matrix(cm, labels):
    '''
    Print the confusion matrix in a nice way
    This will leave out 0 entries and label the columns and rows
    '''
    print ' ',
    for label in labels:
        print '   %s' % label,

    for i in range(cm.size):
        if i % len(labels) == 0:
            print ''
            print labels[i / len(labels)],
        if not cm.item(i) == 0:
            print '%4d' % cm.item(i),
        else:
            print 4 * ' ',


def _evaluate_calssifier(clf, trainingSet, validationSet):
    '''
    Trains the given classifier clf with the training set and compares the
    predicted results with the validation set.
    Returns the trained classifier
    '''
    (tLabels, tVectors) = trainingSet
    (vLabels, vVectors) = validationSet

    t = time.time()
    clf.fit(tVectors, tLabels)
    trainingTime = time.time() - t

    t = time.time()
    pLabels = clf.predict(vVectors)
    validationTime = time.time() - t

    score = clf.score(vVectors, vLabels)

    if (args.verbose > 1):
        print '        Training %d elements in %d Seconds, Prediction in %d ' \
              'Seconds' % (len(tVectors), trainingTime, validationTime)

    if (args.verbose > 2):
        errCount = sum([1 for (p, v) in zip(pLabels, vLabels) if p != v])
        print '        %s errors out of %s validation vectors' % (errCount,
                                                                  len(
                                                                      vVectors))
        label_names = labelEncoder.classes_
        print '        Classification Report:'
        print metrics.classification_report(vLabels, pLabels,
                                            target_names=label_names)
        print '        Confusion Matrix:'
        cm = metrics.confusion_matrix(vLabels, pLabels,
                                      labelEncoder.transform(label_names))
        __print_confusion_matrix(cm, label_names)
    elif (args.verbose > 1):
        print '        %.2f%% Accuracy' % (score * 100)

    return (score, trainingTime, validationTime)


def _evaluate_classifiers(classifiers, datasets):
    '''
    classifiers is expected to be a dictionary of classifiers and sets a dict
    of (training,validation) data sets, each made up of (Labels, Vectors)
    '''
    quality = {}
    for setKey in datasets.keys():
        if (args.verbose > 1):
            print 'Using Dataset %s:' % setKey
        (trainingSet, validationSet) = datasets[setKey]
        quality[setKey] = {}
        for clfKey in classifiers.keys():
            if (args.verbose > 1):
                print '    %s:' % clfKey
            res = _evaluate_calssifier(classifiers[clfKey],
                                       trainingSet,
                                       validationSet)
            quality[setKey][clfKey] = res
            if (args.verbose > 1):
                print ' '
    return quality


def __get_as_int_or_float(num):
    # TODO Document
    try:
        if int(num) > 1:
            return int(num)
    except exceptions.ValueError:
        return float(num)


def _prepare_data_set(trainingSetSize=16000, test_size=None, type=int,
                      scale=False, featureSelection=None):
    '''
    Load training and validation data in desired format
    '''

    train_size = __get_as_int_or_float(trainingSetSize)
    if not test_size is None:
        test_size = __get_as_int_or_float(test_size)

    (labels, vectors) = _parse_csv('res/letter-recognition.data', type)

    if (scale):
        vectors = preprocessing.scale(np.array(vectors))

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        vectors, labels, test_size=test_size, train_size=train_size)
    return ((y_train, x_train), (y_test, x_test))


def _prepare_classifiers(cmd_class=['all']):
    classifiers = {}
    select = args.select_features
    has_all = 'all' in cmd_class
    if has_all or 'svm' in cmd_class:
        # see http://scikit-learn.org/stable/modules/svm.html#classification
        kernels = args.svm_kernels
        if 'all' in kernels:
            kernels = SVM_KERNELS
        for k in kernels:
            if k == 'linear-ovr':
                classifiers[
                    'SVC kernel=linear OvR'] = multiclass.OneVsRestClassifier(
                    svm.SVC(kernel='linear'))
            elif k == 'linear-svc':
                classifiers['Linear SVC'] = svm.LinearSVC()
            else:
                g = 0.0
                if k == 'sigmoid':
                    # TODO: Document this magic number
                    # Maximum dot product of the vectors in our data set
                    g = 1.0 / 962.0
                classifiers['SVC kernel=%s' % k] = svm.SVC(kernel=k, gamma=g)
    if has_all or 'tree' in cmd_class:
        # see http://scikit-learn.org/stable/modules/tree.html
        if select == 'off' or select == 'both':
            classifiers['Default Decision Trees'] = tree.DecisionTreeClassifier()
        if select == 'on' or select == 'both':
            classifiers['Default Decision Trees Feature selection Pipeline'] = \
            Pipeline([('selection', ensemble.ExtraTreesClassifier()),
                             ('classification',tree.DecisionTreeClassifier())])

        for maxf in frange(0.85, 0.90, 0.024):
            for c in ['entropy', 'gini']:
                clf = tree.DecisionTreeClassifier(max_features=maxf, criterion=c)

                if select == 'on' or select == 'both':
                    pipe = Pipeline([('selection', ensemble
                    .ExtraTreesClassifier()),
                                     ('classification', clf)])
                    classifiers['Max Features %.2f with %s Decision Trees ' \
                                'Feature selection Pipeline' % (
                                    maxf, c)] = pipe
                if select == 'off' or select == 'both':
                    classifiers['Max Features %.2f with %s Decision Trees' %
                                (maxf, c)] = clf

    if has_all or 'random' in cmd_class:
        for i in range(10):
            classifiers['Random Tree #%d' % i] = tree \
                .ExtraTreeClassifier()

    if has_all or 'ensemble' in cmd_class:
        min_trees = args.min_trees
        max_trees = args.max_trees
        step = 0
        divisor = 10

        while step < 1:
            step = int((max_trees - min_trees)/divisor)
            divisor -= 1

        for trees in range(min_trees, max_trees+1, step):
            clf1 = ensemble.RandomForestClassifier(bootstrap=False,
                                                   n_estimators=trees)
            clf2 = ensemble.ExtraTreesClassifier(bootstrap=False,
                                                 n_estimators=trees)

            if select == 'on' or select == 'both':
                pipe1 = Pipeline([('selection', ensemble
                .ExtraTreesClassifier()),
                                  ('classification', clf1)])
                pipe2 = Pipeline([('selection', ensemble
                .ExtraTreesClassifier()),
                                  ('classification', clf2)])
                classifiers['%d Random Forest ' \
                            'Feature selection Pipeline' % trees] = pipe1
                classifiers['%d Extra Random Trees ' \
                            'Feature selection Pipeline' % trees] = pipe2

            if select == 'off' or select == 'both':
                classifiers['%d Random Forest' % trees] = clf1
                classifiers['%d Extra Random Trees' % trees] = clf2

    if has_all or 'kNN' in cmd_class or 'rNN' in cmd_class:
        # see http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py
        selected_weights = args.NN_weights
	if 'all' in args.NN_weights:
            selected_weights = ['uniform', 'distance']
	for weight in selected_weights:
            if 'kNN' in cmd_class:
                show_knn_warnings = args.verbose > 3
                for k in args.kNN_neighbors:
                    classifiers['kNN %s k=%d' % (weight, int(k))] = \
                        neighbors.KNeighborsClassifier(k, weights=weight,
                            warn_on_equidistant=show_knn_warnings)
            if 'rNN' in cmd_class:
                # XXX: Buggy scikit does not handle the distance weight
                #
                # The following error message is being thrown by scikit when
                # the distance weight is being used. This could not be fixed so
                # it is being ignored instead.
                #
                # Traceback (most recent call last):
                #   File "lird.py", line 329, in <module>
                #     main()
                #   File "lird.py", line 316, in main
                #     quality = _evaluate_classifiers(classifiers, datasets)
                #   File "lird.py", line 130, in _evaluate_classifiers
                #     validationSet)
                #   File "lird.py", line 90, in _evaluate_calssifier
                #     pLabels = clf.predict(vVectors)
                #   File "/usr/lib/pymodules/python2.7/sklearn/neighbors/classification.py", line 307, in predict
                #     for (pl, w) in zip(pred_labels[inliers], weights)],
                #   File "/usr/lib/pymodules/python2.7/sklearn/utils/extmath.py", line 305, in weighted_mode
                #     w = np.zeros(a.shape, dtype=w.dtype) + w
                # ValueError: operands could not be broadcast together with shapes (47) (194)
                if weight == 'distance':
                    continue

                # Assign the outlier class to outliers.
                outlier_class = labelEncoder.transform([OUTLIER_LABEL])
                for r in args.rNN_radius:
                    classifiers['rNN %s r=%f' % (weight, float(r))] = \
                        neighbors.RadiusNeighborsClassifier(radius=r,
                            weights=weight, outlier_label=outlier_class)

    # Disabled because of horrible performance
    # classifiers['NuSVC'] = svm.NuSVC()
    return classifiers


def _prepare_data_sets(train_size, test_size, sets):
    '''
    Loads and names all desired datasets into a dirctionary
    TODO proper cross validation see
    http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    '''
    datasets = {}
    if 'all' in sets or 'orig' in sets:
        datasets['Out of the Box Integer Data'] = \
            _prepare_data_set(train_size, test_size)
    if 'all' in sets or 'scaled' in sets:
        datasets['Normalizerd Float Data'] = \
            _prepare_data_set(train_size, test_size, float, True)

    return datasets


def _write_dot_file(dtClf, filename='dtClf.dot'):
    '''
    Creates dot file from Decision Tree classifier
    Use graphviz dot tool to create decision tree via
    $ dot -Tpdf out/dtCls.dot -o out/dtCls.pdf
    for more info see http://graphviz.org/
    '''
    try:
        import pydot

        f = open('out/' + filename, 'w')
        tree.export_graphviz(dtClf, out_file=f)
        print 'Written %s file' % ('out/' + filename)
        return
    except ImportError:
        return


def _parse_args():
    '''
    Parse the command line arguments to select test at runtime
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('classifiers', nargs='*', default='all',
        choices=['all', 'svm', 'kNN', 'rNN', 'tree', 'random', 'ensemble'])

    group_common = parser.add_argument_group('Common Options')
    group_common.add_argument('-v', '--verbose', type=int, action='store',
        default=1, choices=range(1, 5),
        help='Verbose Level, Default = %(default)d \
        (Levels: 1=Classifier Score, 2=Timings, \
        3=Confusion Matrix and Result Report, 4=Internal scikit messages)')
    group_common.add_argument('--train-size', default=16000, action='store',
        help='Amount of data used for training. Can be either an int \
        representing the absolute number of samples or a float between 0 and \
        1.0 representing the ratio of train samples. Default %(default)s')
    group_common.add_argument('--test-size', default=None, action='store',
        help='Amount of data used for testing, if not specified the rest of \
        the data set will be used. Like the train-size this can either be a \
        int or a float.')
    group_common.add_argument('--data', action='store', default=['all'],
        choices=['all', 'orig', 'scaled'],
        help='Select the preprocessing of the data. Default %(default)s')

    tree_group = parser.add_argument_group('Decision Tree and Ensemble Methods')
    tree_group.add_argument('--min-trees', action='store',default=10,type=int,
        help='Minimum number of trees used in ensemble methods. \
        Default %(default)d')
    tree_group.add_argument('--max-trees', action='store',default=50,type=int,
        help='Maximum number of trees used in ensemble methods. \
        Default %(default)d')
    tree_group.add_argument('--select-features', action='store', default='on',
        choices=['on', 'off', 'both'],
        help='Dis/enable feature selection before training. \
        Default %(default)s')

    svm_group = parser.add_argument_group('Support Vector Machines')
    svm_group.add_argument('--svm-kernels', action='store', default=['all'],
        choices=SVM_KERNELS, nargs='*',
        help='Select the kernels that should be trained for the SVM. \
        Default: %(default)s')

    nn_group = parser.add_argument_group('Nearest Neighbors')
    nn_group.add_argument('--NN-weights', action='store', default=['all'],
        choices=['all', 'uniform', 'distance'], nargs='*',
        help='The weighting method to be used by the Nearest Neighbors. \
        Default: %(default)s')
    nn_group.add_argument('--kNN-neighbors', action='store', default=[5],
        nargs='*', type=int,
        help='How many Neighbors should be used for classification. If more \
        than one number is given multiple runs will be done. \
        Default %(default)s')
    nn_group.add_argument('--rNN-radius', action='store', default=[3.5], \
        nargs='*', type=float,
        help='What radius should be used for classification. If more \
        than one number is given multiple runs will be done. \
        Default %(default)s')

    global args
    args = parser.parse_args()


def frange(start, stop, step):
    '''
    helper function to get a range generator for float values.
    '''
    while start < stop:
        yield start
        start += step


def main():
    secs = time.time()
    _parse_args()
    datasets = _prepare_data_sets(args.train_size, args.test_size, args.data)
    classifiers = _prepare_classifiers(args.classifiers)
    quality = _evaluate_classifiers(classifiers, datasets)

    # if 'all' in args.classifiers or 'tree' in args.classifiers:
    # _write_dot_file(classifiers['Decision Tree'])

    #rank classifiers by score and print highscore list
    for setKey in quality.keys():
        print 'Score on Dataset: %s' % setKey
        for clf, (score, trainingTime, validationTime) in sorted(quality[setKey].iteritems(),
                key=lambda (k, v): v[0]):
            print "%.2f%% in %d + %d secs: %s" % \
                (100 * score, trainingTime, validationTime, clf)
        print ''

    secs = time.time() - secs;
    print 'Total Time: %d seconds' % secs

    return


if __name__ == "__main__":
    main()
