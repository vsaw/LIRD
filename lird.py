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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
import argparse
import exceptions


# Stores the parsed command line args
args = None

# use labelEncoder.transform() and .inverse_transform() to convert from
# letters to numbers and vice versa.
# The label encoder will be instantiated during data parsing.
labelEncoder = None

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
    labelEncoder.fit(y);
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

    if (args.verbose[0] > 2):
        print '        Training %d elements in %d Seconds, Prediction in %d ' \
              'Seconds' % (len(tVectors), trainingTime, validationTime)
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
    elif (args.verbose[0] > 1):
        print '        %.2f%% Accuracy' % (score * 100)

    return score


def _evaluate_classifiers(classifiers, datasets):
    '''
    classifiers is expected to be a dictionary of classifiers and sets a dict
    of (training,validation) data sets, each made up of (Labels, Vectors)
    '''
    quality = {}
    for setKey in datasets.keys():
        if (args.verbose[0] > 1):
            print 'Using Dataset %s :' % setKey
        (trainingSet, validationSet) = datasets[setKey]
        for clfKey in classifiers.keys():
            if (args.verbose[0] > 1):
                print '    %s :' % clfKey
            score = _evaluate_calssifier(classifiers[clfKey], trainingSet,
                                         validationSet)
            quality[clfKey] = score
            if (args.verbose[0] > 1):
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
    has_all = 'all' in cmd_class
    if has_all or 'svm' in cmd_class:
        # see http://scikit-learn.org/stable/modules/svm.html#classification
        classifiers['Support Vector'] = svm.SVC()
        classifiers['Support Vector with kernel=linear'] = svm.SVC(
            kernel='linear')
        classifiers[
            'Support Vector with kernel=linear OvR'] = multiclass.OneVsRestClassifier(
            svm.SVC(kernel='linear'))
        classifiers['Support Vector with kernel=poly'] = svm.SVC(kernel='poly')
        classifiers['Support Vector with kernel=sigmoid'] = svm.SVC(
            kernel='sigmoid')
        classifiers['Linear SVC'] = svm.LinearSVC()

    if has_all or 'tree' in cmd_class:
        # see http://scikit-learn.org/stable/modules/tree.html
        classifiers['Default Decision Trees'] = tree.DecisionTreeClassifier()
        classifiers['Default Decision Trees Feature selection Pipeline'] = \
            pipe = Pipeline([('selection', ExtraTreesClassifier()),
                        ('classification', tree.DecisionTreeClassifier())])

        for f in frange(0.85, 0.90, 0.024):
            for c in ['entropy', 'gini']:
                clf = tree.DecisionTreeClassifier(max_features=f, criterion=c)
                pipe = Pipeline([('selection', ExtraTreesClassifier()),
                        ('classification', clf)])

                classifiers['Max Features %.2f with %s Decision Trees' % (f, c)
                ] = clf

                classifiers['Max Features %.2f with %s Decision Trees ' \
                            'Feature selection Pipeline' % (f, c)
                ] = pipe


    if has_all or 'kNN' in cmd_class:
        # see http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py
        classifiers['K Nearest Naighbors'] = neighbors.KNeighborsClassifier(
            15, weights='uniform')
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
    parser.add_argument('-v', '--verbose', type=int, nargs=1, default=1,
                        help='verbose level, default = 1', choices=range(1, 4))
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
    parser.add_argument('--data', action='store', default=['all'],
                        choices=['all', 'orig', 'scaled'])
    parser.add_argument('classifiers', nargs='*', default='all',
                        choices=['all', 'svm', 'kNN', 'tree'])
    global args
    args = parser.parse_args()
    pass


def frange(start, stop, step):
    '''
    helper function to get a range generator for float values.
    '''
    while start < stop:
        yield start
        start += step


def main():
    _parse_args()
    classifiers = _prepare_classifiers(args.classifiers)
    datasets = _prepare_data_sets(args.train_size, args.test_size, args.data)
    quality = _evaluate_classifiers(classifiers, datasets)

    # if 'all' in args.classifiers or 'tree' in args.classifiers:
    # _write_dot_file(classifiers['Decision Tree'])

    #rank classifiers by score and print highscore list
    for clf, score in sorted(quality.iteritems(), key=lambda (k, v): (v, k)):
        print "%.2f %%: %s" % (100 * score, clf)

    return


if __name__ == "__main__":
    main()
