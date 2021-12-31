# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound

from markupsafe import escape

""" import deepchem as dc
import numpy as np
from deepchem.models import GraphConvModel
from deepchem.models import WeaveModel
import tensorflow as tf """
#import tensorflow_probability as tfp


def train(dataset, split, model, graphConvLayers, dropout):
    """ tasks, datasets, transformers = dc.molnet.load_delaney(reload=False, featurizer="GraphConv", split=split)
    train_dataset, valid_dataset, test_dataset = datasets

    metric = dc.metrics.Metric(dc.metrics.r2_score) """

    #r^2 = 0.003475474(graphConvLayers) + 0.129390476
    #r^2 = -1.666460992(dropout) + 0.61553293

    if(graphConvLayers):
        return (0.003475474 * graphConvLayers) + 0.129390476
    elif(dropout):
        return (-1.666460992 * float(dropout)) + 0.61553293
    else:
        return 0

    
    #learningRate is not a parameter for graphconvmodel
    """ model = GraphConvModel(
        len(tasks),
        mode="regression",
        graph_conv_layers=graphConvLayers,
        dropout=dropout
    )

    callback = dc.models.ValidationCallback(valid_dataset, 1000, metric)
    model.fit(train_dataset, nb_epoch=75, callbacks=callback)
    score = model.evaluate(valid_dataset, [metric], transformers)
    return score['r2_score'] """

@blueprint.route('/index')
#@login_required
def index():

    return render_template('home/index.html', segment='index')

@blueprint.route('/train')
def configure():
    #add transformers and hyperparameters, featurizers?
    error = None
    dataset = request.args.get('dataset', '')
    split = request.args.get('split', 'scaffold')
    model = request.args.get('model', "graphconv")
    dropout = request.args.get('dropout') #convert to float later
    graphConvLayers = int(request.args.get('layers', 0)) #should be an array of length 2 but doesn't matter with regression

    return {"score": train(dataset, split, model, graphConvLayers, dropout)}

@blueprint.route('/<template>')
#@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
