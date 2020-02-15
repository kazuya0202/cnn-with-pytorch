# -*- Coding: utf-8 -*-
# Packages
import copy
import chainer
import chainer.functions as F
# MyPackages
import GlobalVariable as gv
from GradCAM import GuidedRelu as gr


class BaseBackprop(object):
    def __init__(self, model):
        self.model = model
        self.size = gv.G.Width
        self.xp = model.xp
    # End_Constructor

    def backward(self, x, label, layer):
        with chainer.using_config('train', False):
            acts = self.model(self.xp.asarray(x), layers=[layer, gv.G.TargetLayer])

        acts[gv.G.TargetLayer].grad = self.xp.zeros_like(acts[gv.G.TargetLayer].data)
        if label == -1:
            acts[gv.G.TargetLayer].grad[:, acts[gv.G.TargetLayer].data.argmax()] = 1
        else:
            acts[gv.G.TargetLayer].grad[:, label] = 1

        self.model.cleargrads()
        acts[gv.G.TargetLayer].backward(retain_grad=True)

        return acts
    # End_Method
# End_Class


class GradCAM(BaseBackprop):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
    # End_Constructor

    def generate(self, x, label, layer):
        acts = self.backward(x, label, layer)
        weights = self.xp.mean(acts[layer].grad, axis=(2, 3))
        gcam = self.xp.tensordot(weights[0], acts[layer].data[0], axes=(0, 0))
        # print("GradCAM_Generate : {}".format(gcam))
        gcam = self.xp.maximum(gcam, 0)
        return chainer.cuda.to_cpu(gcam)
    # End_Method
# End_Class


class GuidedBackprop(BaseBackprop):
    def __init__(self, model):
        super(GuidedBackprop, self).__init__(copy.deepcopy(model))
        _replace_relu(self.model)
    # End_Constructor

    def generate(self, x, label, layer):
        acts = self.backward(x, label, layer)
        gbp = chainer.cuda.to_cpu(acts['input'].grad[0])
        gbp = gbp.transpose(1, 2, 0)

        return gbp
    # End_Method
# End_Class


def _replace_relu(chain):
    for key, funcs in chain.functions.items():
        for i in range(len(funcs)):
            if hasattr(funcs[i], 'functions'):
                _replace_relu(funcs[i])
            elif funcs[i] is F.relu:
                funcs[i] = gr.GuidedReLU()
            # End_IfElse
        # End_For
    # End_For
# End_Func
