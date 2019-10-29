from abc import ABCMeta
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Gumbel
from torch.nn import Sequential


def set_temperature(m: torch.nn.Module, temp: torch.Tensor):
    if isinstance(m, MixedModule):
        m.gumbel_temperature.copy_(temp)


def get_named_arch_params(module: torch.nn.Module):
    named_parameters = {n: p for n, p in module.named_parameters() if n.endswith('gumble_arch_params')}
    return named_parameters


def get_named_model_params(module: torch.nn.Module):
    named_parameters = {n: p for n, p in module.named_parameters() if not n.endswith('gumble_arch_params')}
    return named_parameters


class SuperNetwork(torch.nn.Module, metaclass=ABCMeta):
    def set_temperature(self, temp: Union[torch.Tensor, float]):
        if isinstance(temp, float):
            temp = torch.tensor(temp)
        self.apply(lambda x: set_temperature(x, temp))

    def get_named_arch_params(self):
        return get_named_arch_params(self)

    def get_named_model_params(self):
        return get_named_model_params(self)

    def sample_genotype(self):
        genotype_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, MixedModule):
                assert name not in genotype_dict
                genotype_dict[name] = module.sample_genotype_index().item()
        return genotype_dict

    def get_arch_values(self):
        genotype_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, MixedModule):
                assert name not in genotype_dict
                genotype_dict[name] = module.gumble_arch_params.data.cpu().numpy().tolist()
        return genotype_dict


class MixedModule(torch.nn.Module):
    def __init__(self, modules: Union[Sequential[torch.nn.Module], Sequential[Tuple[str, torch.nn.Module]]]):
        super().__init__()
        self.ops = torch.nn.ModuleDict()
        for i, module in enumerate(modules):
            if isinstance(module, torch.nn.Module):
                name = str(i)
                mod = module
            else:
                name, mod = module
            assert isinstance(name, str)
            assert isinstance(mod, torch.nn.Module)
            self.ops[name] = mod

        self.register_buffer('ops_cost', torch.zeros(len(self.ops)))
        self.gumble_arch_params = torch.nn.Parameter(torch.ones(len(self.ops), 1))
        self.register_buffer('gumbel_temperature', torch.ones(1))

    def forward(self, x, *xs, weights: Optional[torch.Tensor] = None, gene: Optional[int] = None):
        assert (weights is None) or (gene is None)

        input_size = x.size()
        batch_size = input_size[0]
        if weights is None and gene is None:
            weights = gumbel_softmax_sample(self.gumble_arch_params.expand(-1, batch_size),
                                            temperature=self.gumbel_temperature, dim=0)

        if gene is not None:
            output = self.ops[str(gene)](x, *xs)
            flops = self.ops_cost[gene].expand(x.shape[0])
        else:
            output = sum(
                w.view(-1, *([1] * (len(input_size) - 1))) * op(x, *xs) for w, op in zip(weights, self.ops.values()))
            flops = torch.sum(weights * self.ops_cost.view(-1, 1), dim=0)

        return output, flops

    def use_flops(self, delete=False):
        for i, op in enumerate(self.ops):
            total_flop = 0
            for m in op.modules():
                if hasattr(m, 'total_ops'):
                    total_flop += m.total_ops.item()
                    if delete:
                        del m.total_ops

                if hasattr(m, 'total_params'):
                    if delete:
                        del m.total_params
            self.ops_cost[i].copy_(torch.tensor(total_flop / 1e6))
        print(self.ops_cost)

    def sample_genotype_index(self):
        with torch.no_grad():
            sampled_alphas = gumbel_softmax_sample(self.gumble_arch_params.squeeze(),
                                                   temperature=self.gumbel_temperature, dim=0)
            best_sampled_alphas = torch.argmax(sampled_alphas, dim=0)
            return best_sampled_alphas.detach()


class MixedSequential(torch.nn.Sequential):
    def forward(self, input):
        total_cost = 0
        for module in self._modules.values():
            input = module(input)
            if isinstance(module, MixedModule):
                input, cost = input
                total_cost += cost
        return input


gumbel_dist = Gumbel(0.0, 1.0)


def gumbel_softmax_sample(logits, temperature, dim=None, std=1.0):
    y = logits + gumbel_dist.sample(logits.shape).to(device=logits.device, dtype=logits.dtype)
    # y = logits + sample_gumbel(logits=logits, std=std)
    return F.softmax(y / temperature, dim=dim)
