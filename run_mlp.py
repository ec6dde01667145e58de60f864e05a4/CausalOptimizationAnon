import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from causal_meta.bivariate.mlp import StructuralModel
from causal_meta.data.mlp import GroundTruthMLPModel

def main(args):
    torch.manual_seed(args.seed)
    ground_truth = GroundTruthMLPModel(args.num_categories,
        hidden_size=args.hidden_size, num_nodes=2)
    model = StructuralModel(ground_truth.num_categories,
        hidden_size=ground_truth.hidden_size)

    # Pretraining
    if args.pretrain:
        optimizer = torch.optim.Adam(model.modules_parameters(), lr=1e-3)
        for samples in tqdm(ground_truth.sample_iter(256, num_batches=args.pretrain),
                total=args.pretrain, desc='Pretraining', leave=False):
            optimizer.zero_grad()
            loss = model.adapt_modules(samples)
            loss.backward()
            optimizer.step()

    # Initialization
    meta_optimizer = torch.optim.Adam(model.structural_parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, 50, gamma=0.9)
    model.w.data.zero_()
    alphas = np.zeros((args.num_episodes,))

    with trange(args.num_episodes) as episodes:
        for episode in episodes:
            optimizer = torch.optim.SGD(model.modules_parameters(), lr=1e-2)
            with ground_truth.save_params():
                # Create a new transfer distribution with an intervention
                ground_truth.intervene()
                with model.save_params():
                    logl_A_B, logl_B_A = torch.tensor(0.), torch.tensor(0.)
                    for samples in ground_truth.sample_iter(1, num_batches=args.num_steps_adaptation):
                        inner_loss_A_B = model.model_A_B(samples)
                        inner_loss_B_A = model.model_B_A(samples)

                        with torch.no_grad():
                            logl_A_B += torch.sum(inner_loss_A_B)
                            logl_B_A += torch.sum(inner_loss_B_A)

                        optimizer.zero_grad()
                        inner_loss = -torch.mean(inner_loss_A_B + inner_loss_B_A)
                        inner_loss.backward()
                        optimizer.step()

                    meta_optimizer.zero_grad()
                    loss = -torch.mean(model.online_loglikelihood(logl_A_B, logl_B_A))
                    loss.backward()
                    meta_optimizer.step()

            alpha = torch.sigmoid(model.w).item()
            alphas[episode] = alpha
            episodes.set_postfix(alpha='{0:.4f}'.format(alpha))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MLP')

    parser.add_argument('--num-categories', type=int, default=10,
        help='Number of categories')
    parser.add_argument('--hidden-size', type=int, default=8,
        help='Size of the hidden layer')
    parser.add_argument('--pretrain', type=int, default=1000,
        help='Number of steps of pre-training')
    parser.add_argument('--num-episodes', type=int, default=1000,
        help='Number of meta-training episodes')
    parser.add_argument('--num-steps-adaptation', type=int, default=10,
        help='Number of steps of adaptation')
    parser.add_argument('--seed', type=int, default=3,
        help='Random seed')

    args = parser.parse_args()
    main(args)
