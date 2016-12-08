import json
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.conf import settings as djangoSettings
import os

from .models import Document
from .forms import DocumentForm

import datetime

import json


import sys
sys.path.append('..')
import blackbox as bb

x = ""
def index(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
            global x
            x = "media/images/" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            return HttpResponseRedirect('/fonts/result')
    else:
        form = DocumentForm()
    documents = Document.objects.all()
    return render(request, 'fonts/index.html', {'documents': documents, 'form': form},)


def result(request):
    # data = json.loads(bb.blackbox(x))
    data = bb.blackbox(x)
    data.replace('_',' ')
    return render(request, 'fonts/result.html', {'thing': data}) # replace thing not in quotations with json object
