import pdb
import torch
from utils import *
import argparse
from neuralop.models.moe import MOEOperator,ExpertFactory

if __name__ == '__main__':
    x: torch.Tensor = torch.randn(size=(1,1,1000,350))
    args: argparse.Namespace = build_argparser_and_parse()
    config, runtime_ctx = get_seismic_config(args=args)
    is_logger = runtime_ctx['is_logger']
    experts = ExpertFactory.create_expert_ensemble(
            expert_configs=config.expert_configs,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels
    )
    model = MOEOperator(
        experts=experts,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_channels=config.hidden_channels,
        top_k=config.top_k,
        noisy_gating=config.noisy_gating,
        fusion_type=config.fusion_type,
        router_hidden_dim=config.router_hidden_dim,
        is_logger=is_logger,
        router_type=config.router_type,
        s_processor_type = config.s_processor_type,
        w_processor_type = config.w_processor_type,
        beta = config.beta,
        is_specific = config.is_specific,
        is_classier = config.is_classier,
        batch_size=config.batch_size,
        resize_input_for_experts = True,
        v_type_num=config.v_type_num,
    )
    output = model(x)
