# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import copy
import gc
import logging
import math
import pprint

import torch
import tqdm

from utils import quant_utils, utils
from train_utils.quant_linear import QuantizeLinear


class GPTQ:
    def __init__(self, layer):
        self.layer = layer # merely point to the model's layer (not cloning)
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns))
        self.nsamples = 0
    
    def _reinit(self):
        self.H = torch.zeros((self.columns, self.columns))
        self.nsamples = 0
        gc.collect()
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H = self.H.to(self.dev)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
    ):
        W = self.layer.weight.data.to(self.dev)
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H.to(self.dev)
        self._reinit()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        # Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            # Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                # Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            # Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        # self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
        #     self.layer.weight.data.dtype
        # )

        self.layer.qweight = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        self.layer.qweight_activated = True

        if torch.any(torch.isnan(self.layer.qweight.data)):
            logging.warning("NaN in qweights")

            pprint.pprint(
                self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in qweights")


@torch.no_grad()
def gptq_fwrd(model, dev, args):
    """
    From GPTQ repo
    """
    layers = model.model.layers
    torch.cuda.empty_cache()

    for i in tqdm.tqdm(range(len(layers)), desc="Inserting weight quantizer"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(
            layer, layers=[torch.nn.Linear, QuantizeLinear]
        )

        for name in subset:
            layer_weight_bits = args.w_bits
            layer_weight_sym = not (args.w_asym)
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8
                
            if (args.target_module is None) \
                or (name.find(args.target_module) > -1):
                gptq = GPTQ(subset[name])
                gptq.quantizer = quant_utils.WeightQuantizer()
                gptq.quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                )

                subset[name].layer_w_groupsize = args.w_groupsize
                subset[name].percdamp = args.percdamp
                subset[name].actorder = args.act_order
                subset[name].gptq = gptq
            
            else:
                quantizer = quant_utils.WeightQuantizer()
                quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=not (args.w_asym),
                    mse=args.w_clip,
                )
                subset[name].quantizer = quantizer

        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
