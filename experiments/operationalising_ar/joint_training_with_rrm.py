import copy

import numpy as np
import torch

from experiments.operationalising_ar.agents import AgentsPool
from experiments.operationalising_ar.algo import run_offline_training, pretrain_ar_policy_offline, \
    train_ar_response_model, run_rrm_with_ar, run_rrm_with_ce
from experiments.operationalising_ar.dm import TrainedAgentModel, ARSampler
from pysrc.models import ThreeLayerReLUNet, TrainableConstantNet


def h(input: torch.Tensor) -> torch.Tensor:
    """
    A quadratic outcome function.
    """
    n, k = input.shape

    A = torch.ones((k, k), dtype=input.dtype, device=input.device)  # (k, k)
    B = -torch.ones(k, dtype=input.dtype, device=input.device)  # (k,)

    # scaling
    A = A * (1 / k)
    B = B * (1 / k)

    # Quadratic term: sum(x_i^T A x_i) for each row x_i
    quadratic_term = torch.einsum('ij,jk,ik->i', input, A, input)  # Shape (n,)

    # Linear term: x @ B^T
    linear_term = input @ B  # Shape (n,)

    # (n,)
    y = quadratic_term + linear_term

    return y


# def run_standard_ce(ap: AgentsPool, g: UsefulModel, n_epochs: int):
#     # init
#     xb = torch.from_numpy(ap.xb)
#     # n, d = xb.shape
#
#     # before generating, deploy arbitrary `xr` (e.g., by letting xr = xb).
#     xr = xb.detach().clone().requires_grad_()
#     nmse, compliance_ratio = query_strategic_nmse(ap=ap, g=g, xr=xr, h=h)
#
#     print(f"Compliance ratio: {compliance_ratio}")
#     print(f"Initial nmse: {nmse}")
#
#     # generating standard counterfactual explanations.
#     with tqdm(total=n_epochs, desc="Generating standard CE", unit="epoch", ncols=100) as pbar:
#         gen_ce(xb=xb, g=g, n_epochs=n_epochs, progress_bar=pbar)
#
#     # after generating standard xr, deploy them.
#     nmse, compliance_ratio = query_strategic_nmse(ap=ap, g=g, xr=xr, h=h)
#
#     print(f"Compliance ratio: {compliance_ratio}")
#     print(f"Final nmse: {nmse}")
#
#     return


def run():
    # config
    torch.set_num_threads(44)  # TODO: better way to determine the optimal number of threads.
    torch.set_default_dtype(torch.float64)
    xr_frequency = 1

    n = 10000  # num agents per round
    m = 100  # num rounds
    d = 3  # num observed features

    # Set seeds for reproducibility
    seed_val = 11
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # generate data
    test_pool = AgentsPool(n=(m * n), d=d, h=h)

    # models
    g = ThreeLayerReLUNet(input_size=d, hidden_size1=4, hidden_size2=4, output_size=1)
    local_sigma = ThreeLayerReLUNet(input_size=d, hidden_size1=4, hidden_size2=4, output_size=d)
    global_sigma = TrainableConstantNet(initial_value=torch.from_numpy(AgentsPool(n=1, d=d).xb[0]))
    psi = TrainedAgentModel(
        probabilistic_model=(
            ThreeLayerReLUNet(input_size=(2 * d + 1),
                              hidden_size1=4, hidden_size2=4,
                              output_size=1,
                              is_classification=True)
        ),
        binary_threshold=0.5
    )

    # pretrain models, on offline data
    # This step is CRUCIAL, do not remove!
    pretraining_pool = AgentsPool(n=5000, d=d, h=h)

    print("================= Pretrain g & sigma =================")
    run_offline_training(ap=pretraining_pool, g=g, h=h, n_epochs=2000, stop_at=0.01)
    pretrain_ar_policy_offline(sigma=local_sigma,
                               xb=torch.from_numpy(pretraining_pool.xb),
                               n_epochs=200)

    # interact with agents to learn their reaction model.
    print("================= Learning agents' reaction model =================")
    train_ar_response_model(training_pool=AgentsPool(n=(m * n), d=d, h=h), test_pool=test_pool,
                            g=g, psi=psi,
                            pi=ARSampler(local_sigma=local_sigma, global_sigma=global_sigma),
                            n_epochs=500)

    # run experiments
    training_set = [AgentsPool(n=n, d=d, h=h) for _ in range(m)]

    print("================= Learning local AR map with RRM =================")
    g1 = copy.deepcopy(g)
    local_losses = run_rrm_with_ar(training_set=training_set, test_pool=test_pool,
                                   h=h, g=g1,
                                   psi=psi, sigma=local_sigma,
                                   n_epochs=10, xr_frequency=xr_frequency)

    print("================= Learning global AR map with RRM =================")
    g2 = copy.deepcopy(g)
    global_losses = run_rrm_with_ar(training_set=training_set, test_pool=test_pool,
                                    h=h, g=g2,
                                    psi=psi, sigma=global_sigma,
                                    n_epochs=10, xr_frequency=xr_frequency)

    print("================= Standard CE with RRM =================")
    g3 = copy.deepcopy(g)
    ce_losses = run_rrm_with_ce(training_set=training_set, test_pool=test_pool,
                                h=h, g=g3,
                                psi=psi,
                                n_rrm_epochs=10, n_ce_epochs=10, xr_frequency=xr_frequency)

    # save results
    np.save(f"local-ar-training-losses.npy", local_losses[0])
    np.save(f"local-ar-test-losses.npy", local_losses[1])

    np.save(f"global-ar-training-losses.npy", global_losses[0])
    np.save(f"global-ar-test-losses.npy", global_losses[1])

    np.save(f"CE-ar-training-losses.npy", ce_losses[0])
    np.save(f"CE-ar-test-losses.npy", ce_losses[1])

    return


if __name__ == '__main__':
    run()
