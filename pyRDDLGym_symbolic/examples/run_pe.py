"""
SIR domain:

python pyRDDLGym_symbolic/examples/run_pe.py --domain SIR --policy_fpath=./pyRDDLGym_symbolic/examples/files/SIR/policy/p1.json --max_iter=4 --save_graph

Racecar domain:

python pyRDDLGym_symbolic/examples/run_pe.py --domain Racecar --policy_fpath=./pyRDDLGym_symbolic/examples/files/Racecar/policy/policy.json --max_iter=5 --save_graph

The Racecar domain also has a square goal region domain apart from the default circular goal region. Uncomment the respective lines in the domain file to use the square goal region.

Reservoir domain:

python pyRDDLGym_symbolic/examples/run_pe.py --domain Reservoir --policy_fpath=./pyRDDLGym_symbolic/examples/files/Reservoir/policy/policy.json --max_iter=2 --save_graph


Note that changing 'resolution' to a smaller value will increase the accuracy of the plot, but will also increase the computation time.
"""

"""An example PE run."""

import argparse
import os

from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym.core.parser.reader import RDDLReader
from pyRDDLGym.core.parser.parser import RDDLParser

from pyRDDLGym_symbolic.core.model import RDDLModelXADD
from pyRDDLGym_symbolic.mdp.mdp_parser import MDPParser
from pyRDDLGym_symbolic.mdp.policy_parser import PolicyParser
from pyRDDLGym_symbolic.solver.pe import PolicyEvaluation

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

_DIR = 'pyRDDLGym_symbolic/examples/files/{domain}/'
_DOMAIN_PATH = _DIR + 'domain.rddl'
_INSTANCE_PATH = _DIR + 'instance{instance}.rddl'


def run_vi(args: argparse.Namespace):
    """Runs PE."""
    # Read and parse domain and instance
    domain = args.domain
    instance = args.instance
    domain_file = _DOMAIN_PATH.format(domain=domain)
    instance_file = _INSTANCE_PATH.format(domain=domain, instance=instance)
    reader = RDDLReader(
        domain_file,
        instance_file,
    )
    rddl_txt = reader.rddltxt
    parser = RDDLParser(None, False)
    parser.build()

    # Parse RDDL file
    rddl_ast = parser.parse(rddl_txt)

    # Ground domain
    grounder = RDDLGrounder(rddl_ast)
    model = grounder.ground()

    # XADD compilation
    xadd_model = RDDLModelXADD(model, reparam=False)
    xadd_model.compile()

    mdp_parser = MDPParser()
    mdp = mdp_parser.parse(
        xadd_model,
        xadd_model.discount,
        concurrency=rddl_ast.instance.max_nondef_actions,
        is_linear=args.is_linear,
        is_vi=False,
    )

    policy_parser = PolicyParser()
    policy = policy_parser.parse(
        mdp=mdp,
        policy_fname=args.policy_fpath,
        assert_concurrency=args.assert_concurrency,
        concurrency=mdp.max_allowed_actions,
    )

    pe_solver = PolicyEvaluation(
        policy=policy,
        mdp=mdp,
        max_iter=args.max_iter,
        enable_early_convergence=args.enable_early_convergence,
        perform_reduce_lp=args.reduce_lp,
    )
    res = pe_solver.solve()

    # print(f"{model.cpfs=}")
    # print(f"{mdp.context._id_to_node[22]}")

    # Export the solution to a file
    env_path = os.path.dirname(domain_file)
    sol_dir = os.path.join(env_path, 'sdp', 'pe')
    os.makedirs(sol_dir, exist_ok=True)
    for i in range(args.max_iter):
        sol_fpath = os.path.join(sol_dir, f'value_dd_iter_{i+1}.xadd')
        value_dd = res['value_dd'][i]
        mdp.context.export_xadd(value_dd, fname=sol_fpath)

        # Visualize the solution XADD
        if args.save_graph:
            # Below is a hack to enforce saving to the given dir
            graph_fpath = os.path.join(
                os.path.abspath(sol_dir), f'value_dd_iter_{i+1}.pdf')
            mdp.context.save_graph(value_dd, file_name=graph_fpath)
    print(f'Times per iterations: {res["time"]}')

    var_set = mdp.context.collect_vars(res["value_dd"][-1])
    print(f"{var_set=}")
    var_dict = {}
    for i in var_set:
        var_dict[f"{i}"] = i
    # print(f"{var_dict=}")

    #########################################################################################################################
    # Visualizations

    def plot_and_save(Z, x, y, xlabel, ylabel, title, filename, show=False):
        X, Y = np.meshgrid(x, y)
        fig, ax = plt.subplots()
        c = ax.pcolormesh(X, Y, Z, cmap="gray", shading="auto")
        fig.colorbar(c, ax=ax)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close()

    if domain == "SIR":

        resolution = 0.01
        # the smaller the resolution, the more accurate the plot, but the slower the computation

        infected = 1e-10
        beta_values = np.arange(0, 2, resolution)

        ## no vaccination (gamma vs beta)

        gamma_values = np.arange(0, 2, resolution)
        Z = np.zeros((len(gamma_values), len(beta_values)))

        bool_assign = {var_dict["g"]: False, var_dict["overwhelmed"]: False}
        cont_assign = {var_dict["infected"]: infected, var_dict["susceptible"]: 1 - infected, var_dict["vaccinated"]: 0}

        for i, gemma_val in enumerate(tqdm(gamma_values)):
            for j, beta_val in enumerate(beta_values):
                cont_assign[var_dict["beta"]] = beta_val
                cont_assign[var_dict["gamma"]] = gemma_val
                Z[i][j] = mdp.context.evaluate(res["value_dd"][-1], bool_assign=bool_assign, cont_assign=cont_assign)

        Z = Z.T

        plot_and_save(Z, gamma_values, beta_values, "Recovery Rate", "Infection Rate", f"Value Function with no Vaccination and Initial Infected = {cont_assign[var_dict['infected']]}", f"imgs/SIR_{args.max_iter}_{resolution}_{infected}_no_vaccination.png", show=True)

        ## with vaccination (vaccination vs beta with different gamma values)

        vaccination_values = np.arange(0, 1, resolution)

        for gamma in [5e-4, 5e-3, 5e-2, 5e-1]:

            Z = np.zeros((len(vaccination_values), len(beta_values)))

            bool_assign = {var_dict["g"]: False, var_dict["overwhelmed"]: False}
            cont_assign = {var_dict["infected"]: infected, var_dict["susceptible"]: 1 - infected, var_dict["gamma"]: gamma}

            for i, vaccination_val in enumerate(tqdm(vaccination_values)):
                for j, beta_val in enumerate(beta_values):
                    cont_assign[var_dict["beta"]] = beta_val
                    cont_assign[var_dict["vaccinated"]] = vaccination_val
                    Z[i][j] = mdp.context.evaluate(res["value_dd"][-1], bool_assign=bool_assign, cont_assign=cont_assign)

            Z = Z.T

            plot_and_save(Z, vaccination_values, beta_values, "Vaccination Rate", "Infection Rate", f"Value Function with Recovery Rate = {cont_assign[var_dict['gamma']]}", f"imgs/SIR_{args.max_iter}_{resolution}_{gamma}.png", show=True)

    elif domain == "Racecar":

        resolution = 0.5
        # the smaller the resolution, the more accurate the plot, but the slower the computation

        ax_noise = np.arange(-30, 30, resolution)
        ay_noise = np.arange(-5, 250, resolution)

        Z = np.zeros((len(ax_noise), len(ay_noise)))

        bool_assign = {var_dict["reach"]: False, var_dict["violation"]: False}
        cont_assign = {var_dict["ax"]: 0, var_dict["ay"]: 0, var_dict["x"]: 1, var_dict["y"]: 0, var_dict["vx"]: 0, var_dict["vy"]: 0.0}

        for i, t1 in enumerate(tqdm(ax_noise)):
            for j, t2 in enumerate(ay_noise):
                cont_assign[var_dict["ax_noise"]] = t1
                cont_assign[var_dict["ay_noise"]] = t2
                Z[i][j] = mdp.context.evaluate(res["value_dd"][-1], bool_assign=bool_assign, cont_assign=cont_assign)

        Z = Z.T

        plot_and_save(Z, ax_noise, ay_noise, "ax noise", "ay noise", f"Value Function for {args.max_iter} steps", f"./imgs/Racecar_{args.max_iter}_{resolution}.png", show=True)

    elif domain == "Reservoir":

        resolution = 0.1
        # the smaller the resolution, the more accurate the plot, but the slower the computation

        rlevel___t1 = np.arange(42.5, 57.5, resolution)
        rlevel___t2 = np.arange(42.5, 57.5, resolution)

        Z = np.zeros((len(rlevel___t1), len(rlevel___t2)))

        bool_assign = {var_dict["failure"]: False}
        cont_assign = {}

        for i, t1 in enumerate(tqdm(rlevel___t1)):
            for j, t2 in enumerate(rlevel___t2):
                cont_assign[var_dict["rlevel___t1"]] = t1
                cont_assign[var_dict["rlevel___t2"]] = t2
                Z[i][j] = mdp.context.evaluate(res["value_dd"][-1], bool_assign=bool_assign, cont_assign=cont_assign)

        plot_and_save(1-Z.T, rlevel___t1, rlevel___t2, "rlevel___t1", "rlevel___t2", f"Value Function for {args.max_iter} steps", f"imgs/Reservoir_{args.max_iter}_{resolution}.png", show=True)

    else:
        raise ValueError("Domain not evaluated in the thesis")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--domain', type=str, default='RobotLinear_1D',
                        help='The name of the RDDL environment.')
    parser.add_argument('--instance', type=str, default='0',
                        help='The instance number of the RDDL environment.')
    parser.add_argument('--policy_fpath', type=str,
                        help='The file path to the policy.')
    parser.add_argument('--max_iter', type=int, default=10,
                        help='The maximum number of iterations')
    parser.add_argument('--enable_early_convergence', action='store_true',
                        help='Whether to enable early convergence.')
    parser.add_argument('--is_linear', action='store_true',
                        help='Whether the MDP is linear or not.')
    parser.add_argument('--reduce_lp', action='store_true',
                        help='Whether to perform the reduce LP function.')
    parser.add_argument('--assert_concurrency', action='store_true',
                        help='Whether to assert concurrency or not')
    parser.add_argument('--save_graph', action='store_true',
                        help='Whether to save the XADD graph to a file.')
    args = parser.parse_args()

    run_vi(args)
