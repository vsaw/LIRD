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
#from sklearn.externals.six import StringIO
import argparse
import exceptions


# Stores the parsed command line args
args = None


def _parse_csv(filename, typeCast=int):
    '''
    Parse data from CSV and return (y,X) with y a array of labels and X array
    of observation vectors. X entries are casted using typeCast
    '''
    data = csv.reader(open(filename))
    X = []
    y = []
    for row in data:
        # sklearn classifiers expect classes to be integers.
        #
        # The classes in the set however are stored as characters so they are
        # converted to their ASCII ordinal on load. Otherwise some classifiers
        # (e.g. DecisionTreeClassifier) fail to run or produce false results.
        y.append(ord(row[0]))
        X.append([typeCast(r) for r in row[1:len(row)]])
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
    # print '    Training set of size %s, Validation set of size %s' % (len(tVectors), len(vVectors))
    t = time.time()
    clf.fit(tVectors, tLabels)
    elapsed = time.time() - t
    print '        Training %d elements in %d Seconds' % (len(tVectors), elapsed)
    t = time.time()
    pLabels = clf.predict(vVectors)
    elapsed = time.time() - t
    print '        Prediction in %d Seconds' % elapsed
    errCount = sum([1 for (p, v) in zip(pLabels, vLabels) if p != v])
    print '        %f%%: %s errors out of %s validation vectors' % \
        (100. * float(errCount) / len(vVectors), errCount, len(vVectors))
    if args.verbose[0] > 2:
        label_names = map(lambda x: chr(x), range(ord('A'), ord('Z') + 1))
        print '        Classification Report:'
        print metrics.classification_report(vLabels, pLabels,
                                            target_names=label_names)
        print '        Confusion Matrix:'
        cm = metrics.confusion_matrix(vLabels, pLabels,
                                      range(ord('A'), ord('Z') + 1))
        __print_confusion_matrix(cm, label_names)


def _evaluate_classifiers(classifiers, datasets):
    '''
    classifiers is expected to be a dictionary of classifiers and sets a dict
    of (training,validation) data sets, each made up of (Labels, Vectors)
    '''
    for setKey in datasets.keys():
        print 'Using Dataset %s :' % setKey
        (trainingSet, validationSet) = datasets[setKey]
        for clfKey in classifiers.keys():
            print '    Training %s Classifier:' % clfKey
            _evaluate_calssifier(classifiers[clfKey], trainingSet,
                                 validationSet)
            print ' '


def __get_as_int_or_float(num):
    # TODO Document
    try:
        if int(num) > 1:
            return int(num)
    except exceptions.ValueError:
        return float(num)


def _prepare_data_set(trainingSetSize=16000, test_size=None, type=int, scale=False):
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
    has_all = 'all' in cmd_class
    if has_all or 'svm' in cmd_class:
        # see http://scikit-learn.org/stable/modules/svm.html#classification
        classifiers['Support Vector'] = svm.SVC()
        classifiers['Support Vector with kernel=linear'] = svm.SVC(kernel='linear')
        classifiers['Support Vector with kernel=linear OvR'] = multiclass.OneVsRestClassifier(svm.SVC(kernel='linear'))
        classifiers['Support Vector with kernel=poly'] = svm.SVC(kernel='poly')
        classifiers['Support Vector with kernel=sigmoid'] = svm.SVC(kernel='sigmoid')
        classifiers['Linear SVC'] = svm.LinearSVC()
    if has_all or 'tree' in cmd_class:
        # see http://scikit-learn.org/stable/modules/tree.html
        classifiers['Decision Tree'] = tree.DecisionTreeClassifier()
    if has_all or 'kNN' in cmd_class:
        # see http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py
        classifiers['K Nearest Naighbors'] = neighbors.KNeighborsClassifier(15,
                                                                            weights='uniform')
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
    if 'both' in sets or 'orig' in sets:
        datasets['Out of the Box Integer Data'] = _prepare_data_set(train_size,
                                                                    test_size)
    if 'both' in sets or 'scaled' in sets:
        datasets['Normalizerd Float Data'] = _prepare_data_set(train_size,
                                                               test_size,
                                                               float, True)
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

        f = open('out/'+filename, 'w')
        tree.export_graphviz(dtClf, out_file=f)
        print 'Written %s file' % ('out/'+filename)
        return
    except ImportError:
        return


def _parse_args():
    '''
    Parse the command line arguments to select test at runtime
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', type=int, nargs=1, default=1,
                        help='verbose level, default = 1', choices=range(1,4))
    parser.add_argument('--train-size', default=16000, action='store',
                        help='amount of data used for training. Can be either \
                              an int representing the absolute number of \
                              samples or a float between 0 and 1.0 \
                              representing the ratio of train samples')
    parser.add_argument('--test-size', default=None, action='store',
                        help='amount of data used for testing, if not \
                              specified the rest of the data set will be used \
                              Like the train-size this can either be a int or \
                              a float')
    parser.add_argument('--data', action='store', default=['both'],
                        choices=['both', 'orig', 'scaled'])
    parser.add_argument('classifiers', nargs='*', default='all',
                        choices=['all', 'svm', 'kNN', 'tree'])
    global args
    args = parser.parse_args()
    pass


def main():
    _parse_args()
    classifiers = _prepare_classifiers(args.classifiers)
    datasets = _prepare_data_sets(args.train_size, args.test_size, args.data)
    _evaluate_classifiers(classifiers, datasets)
    if 'all' in args.classifiers or 'tree' in args.classifiers:
        _write_dot_file(classifiers['Decision Tree'])
    return


if __name__ == "__main__":
    main()
