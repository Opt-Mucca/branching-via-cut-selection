import networkx as nx
import os
import argparse
import numpy as np
from pyscipopt import Model


def read_network_data(instance_data, link_model='undirected'):
    """
    Parent function for reading network data from SNDLib: http://sndlib.zib.de/home.action
    It reads the ASCII file formats, not the XML formats
    Note: This reader ignores hop limits and all explicit path information
    Args:
        instance_data (list): The readlines() result of reading in the ASCII network file
        link_model (str): What type of graph are we building (undirected, directed, bidirected)

    Returns:
        Network relevant data, split into node specific, link specific, and demand specific information.
    """

    # Identify the start and end of the node definitions
    start_nodes_i = None
    end_nodes_i = None
    for i, line in enumerate(instance_data):
        if line == 'NODES (\n':
            start_nodes_i = i
        if start_nodes_i is not None and end_nodes_i is None and line == ')\n':
            end_nodes_i = i
            break
    assert start_nodes_i is not None and end_nodes_i is not None, 'Failed reading network {}'.format(instance_data[1])
    # Read all node relevant information
    node_dict, node_mapping = read_node_data(instance_data[start_nodes_i + 1:end_nodes_i])

    # Identify the start and end of the link definitions
    start_links_i = None
    end_links_i = None
    for i, line in enumerate(instance_data):
        if line == 'LINKS (\n':
            start_links_i = i
        if start_links_i is not None and end_links_i is None and line == ')\n':
            end_links_i = i
            break
    assert start_links_i is not None and end_links_i is not None, 'Failed reading network {}'.format(instance_data[1])
    # Read all the link relevant information
    node_dict, node_mapping, link_dict, link_mapping, bidirected_link_mapping = read_link_data(
        node_dict, node_mapping, instance_data[start_links_i + 1:end_links_i], link_model=link_model)

    # Identify the start and end of the demand definitions
    start_demands_i = None
    end_demands_i = None
    for i, line in enumerate(instance_data):
        if line == 'DEMANDS (\n':
            start_demands_i = i
        if start_demands_i is not None and end_demands_i is None and line == ')\n':
            end_demands_i = i
            break
    assert start_demands_i is not None and end_demands_i is not None, 'Failed reading network {}'.format(
        instance_data[1])
    # Read all the demand relevant information
    demand_dict = read_demand_data(instance_data[start_demands_i + 1:end_demands_i])

    return node_dict, node_mapping, link_dict, link_mapping, bidirected_link_mapping, demand_dict


def read_node_data(node_data):
    """
    Function for reading all the node relevant data. It assumes that only the lines of the file that are
    within the node definition of the ASCII file format are given as an argument
    Args:
        node_data (list): List of node relevant lines from the ASCII network file format

    Returns:
        Node relevant information as a dictionary
    """
    node_dict = {}
    node_mapping = {}

    # Line format: '  node_name ( longitude latitude )

    for i, node in enumerate(node_data):
        # Then this is a comment or an empty line
        if node.strip(' ')[0] == '#' or node.strip(' ') == '':
            continue
        node = node.split('(')
        assert len(node) == 2
        node_name = node[0].strip(' ')
        node_x1 = float(node[1].split(' ')[1])
        node_x2 = float(node[1].split(' ')[2])
        node_dict = add_node_data(node_dict, node_name, i, node_x1, node_x2)
        node_mapping[i] = node_name

    return node_dict, node_mapping


def add_node_data(node_dict, node_name, node_i, node_x1, node_x2):
    """
    Function for adding information on a single node to the node information dictionary
    Args:
        node_dict (dict): Node relevant information as a dictionary
        node_name (str): Node name as defined in the ASCII network file
        node_i (int): The ID that of the node that networkX will use
        node_x1 (float): The x1 geographic coordinate (used for plotting)
        node_x2 (float): The x2 geographic coordinate (used for plotting)

    Returns:
        The updated node information dictionary
    """

    assert node_name not in node_dict, 'node {} already in dict {}'.format(node_name, node_dict)

    # Add the relevant information
    node_dict[node_name] = {}
    node_dict[node_name]['i'] = node_i
    node_dict[node_name]['x1'] = node_x1
    node_dict[node_name]['x2'] = node_x2

    return node_dict


def read_link_data(node_dict, node_mapping, link_data, link_model='undirected'):
    """
    Function for reading all the link relevant data. It assumes that only the lines of the file that are
    within the link definition of the ASCII file format are given as an argument
    Args:
        node_dict (dict): Dictionary containing all node relevant data
        node_mapping (dict): Dictionary containing mapping from node networkx ID to ID from ASCII file
        link_data (list): List of link relevant lines from the ASCII network file format
        link_model (str): Whether the link_model is 'undirected', 'directed', or 'bidirected'

    Returns:
        Link relevant information as a dict
    """
    link_dict = {}
    link_mapping = {node_name: {} for node_name in node_dict}
    n_dummy_links = 0
    bidirected_link_mapping = {}

    # Line format: '  link_name ( from_node to_node ) pre_installed_capacity pre_installed_capacity_cost
    # routing_cost setup_cost ( module_capacity1 module_cost1 module_capacity_2 module_cost2 ..... )'

    for link_i, link in enumerate(link_data):
        # Then this is a comment or an empty line
        if link.strip(' ')[0] == '#' or link.strip(' ') == '':
            continue
        link_split = link.split(' ')
        # Extract the name of the link. In the case of a bidirected model, create two links
        link_names = [link_split[2]] if link_model != 'bidirected' else [link_split[2] + '_bidirected_0',
                                                                         link_split[2] + '_bidirected_1']
        if link_model == 'bidirected':
            bidirected_link_mapping[link_names[0]] = link_names[1]
            bidirected_link_mapping[link_names[1]] = link_names[0]
        for link_name_i, link_name in enumerate(link_names):
            # Extract the static features of the link
            pre_installed_capacity = float(link_split[7])
            pre_installed_capacity_cost = float(link_split[8])
            routing_cost = float(link_split[9])
            setup_cost = float(link_split[10])
            capacities = []
            link_capacities = link.split('(')[2].split(')')[0].strip(' ').split(' ')
            assert len(link_capacities) == 1 or len(link_capacities) % 2 == 0, print(link, link_capacities)
            for j in range(len(link_capacities) // 2):
                capacities.append((float(link_capacities[2 * j]), float(link_capacities[2 * j + 1])))
            if len(capacities) == 0:
                capacities.append((0, 0))
            # Keep track of the from_node and to_node. In bidirected the from and to node are swapped for second link
            from_node = link_split[4] if link_name_i == 0 else link_split[5]
            to_node = link_split[5] if link_name_i == 0 else link_split[4]
            # In the following case a link already exists between these two nodes. So slightly transform the problem
            if (from_node in link_mapping and to_node in link_mapping[from_node]) or (
                    link_model == 'undirected' and to_node in link_mapping and from_node in link_mapping[to_node]):
                # Now create a fake node, add the link from_node to fake_node
                # Then a fake arc that connects the fake node and to_node
                fake_node_name = '{}_dummy_node_0'.format(link_name)
                fake_link_name = '{}_dummy_link_0'.format(link_name)
                while fake_node_name in node_dict and fake_link_name in link_dict:
                    fake_node_name_split = fake_node_name.split('_')
                    fake_i = int(fake_node_name_split[-1])
                    fake_node_name = '{}_dummy_node_{}'.format(link_name, fake_i)
                    fake_link_name = '{}_dummy_link_{}'.format(link_name, fake_i)
                node_dict = add_node_data(
                    node_dict, fake_node_name, len(node_dict),
                    ((node_dict[from_node]['x1'] + node_dict[to_node]['x1']) / 2) + np.random.uniform(0, 1e-2),
                    ((node_dict[from_node]['x2'] + node_dict[to_node]['x2']) / 2) + np.random.uniform(0, 1e-2))
                node_mapping[len(node_dict) - 1] = fake_node_name

                # First the link from from_node to fake_node
                link_dict = add_link_data(link_dict, link_name, len(link_dict), from_node,
                                          fake_node_name, pre_installed_capacity, pre_installed_capacity_cost,
                                          routing_cost, setup_cost, capacities)
                link_mapping[from_node][fake_node_name] = link_name
                n_dummy_links += 1
                fake_capacities = [(c[0], 0) for c in capacities]
                link_dict = add_link_data(link_dict, fake_link_name, len(link_dict),
                                          fake_node_name, to_node, pre_installed_capacity, 0, 0, 0,
                                          fake_capacities)
                assert fake_node_name not in link_mapping
                link_mapping[fake_node_name] = {}
                link_mapping[fake_node_name][to_node] = fake_link_name
            else:
                link_dict = add_link_data(link_dict, link_name, len(link_dict),
                                          from_node, to_node, pre_installed_capacity, pre_installed_capacity_cost,
                                          routing_cost, setup_cost, capacities)
                link_mapping[from_node][to_node] = link_name

    return node_dict, node_mapping, link_dict, link_mapping, bidirected_link_mapping


def add_link_data(link_dict, link_name, link_i, from_node, to_node, pre_installed_capacity, pre_installed_capacity_cost,
                  routing_cost, setup_cost, capacities):
    """
    Function for adding information on a link to the link information dictionary
    Args:
        link_dict (dict): Link information dictionary
        link_name (str): The name of the link
        link_i (int): The ID of the link that we will use in networkX
        from_node (str): The u if the link is considered as an arc (u,v)
        to_node (str): The v if the link is considered as an arc (u,v)
        pre_installed_capacity (float): The pre-installed capacity that already exists if we choose this link
        pre_installed_capacity_cost (float): Cost associated with the pre-installed capacity. This can be ignored
        routing_cost (float): The cost of routing a single unit of flow through this link
        setup_cost (float): The fixed cost of the link if it is selected
        capacities (list): List of tuples of (capacity, cost) where capacity can be added to the link for cost

    Returns:
        Updated link information dictionary
    """

    assert link_name not in link_dict, 'Link {} is already in link_dict {}'.format(link_name, link_dict)
    link_dict[link_name] = {}
    link_dict[link_name]['i'] = link_i
    link_dict[link_name]['from_node'] = from_node
    link_dict[link_name]['to_node'] = to_node
    link_dict[link_name]['pre_installed_capacity'] = pre_installed_capacity
    link_dict[link_name]['pre_installed_capacity_cost'] = pre_installed_capacity_cost
    link_dict[link_name]['routing_cost'] = routing_cost
    link_dict[link_name]['setup_cost'] = setup_cost
    link_dict[link_name]['capacities'] = capacities

    return link_dict


def read_demand_data(demand_data):
    """
    Function for reading all the demand relevant data. It assumes that only the lines of the file that are
    within the demand definition of the ASCII file format are given as an argument
    Args:
        demand_data (list): List of demand relevant lines from the ASCII network file format

    Returns:
        Demand relevant information as a dict
    """
    demand_dict = {}

    # Line format: '  demand_name ( from_node to_node ) routing_unit demand_value max_path_length'

    for i, demand in enumerate(demand_data):
        # Then this is a comment or an empty line
        if demand.strip(' ')[0] == '#' or demand.strip(' ') == '':
            continue
        demand = demand.split(' ')
        name = demand[2]
        demand_dict[name] = {}
        demand_dict[name]['i'] = i
        demand_dict[name]['from_node'] = demand[4]
        demand_dict[name]['to_node'] = demand[5]
        demand_dict[name]['routing_unit'] = float(demand[7])
        demand_dict[name]['demand_value'] = float(demand[8])
        # Ignore the max_path_length

    return demand_dict


def construct_graph(node_dict, link_dict, link_model='undirected'):
    """
    Function for creating the networkx graph model.
    Note: This function is purely used for visualisation help and for the edge_disjoint_paths function. The disjoint
        paths function creates the set of admissible paths for a given demand. Note that this is a simplification
        of the original model, which allowed any set of edge disjoint paths as a solution.
    Args:
        node_dict (dict): Node relevant information
        link_dict (dict): Link relevant information
        link_model (str): Whether the graph is 'undirected', 'directed', or 'bidirected'

    Returns:
        The networkx Graph model object
    """
    if link_model == 'undirected':
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()

    for node in node_dict:
        graph.add_node(node_dict[node]['i'], pos=(node_dict[node]['x1'], node_dict[node]['x2']), name=node)

    for link in link_dict:
        link = link_dict[link]
        graph.add_edge(node_dict[link['from_node']]['i'], node_dict[link['to_node']]['i'],
                       pre_installed_capacity=link['pre_installed_capacity'],
                       pre_installed_capacity_cost=link['pre_installed_capacity_cost'],
                       routing_cost=link['routing_cost'], setup_cost=link['setup_cost'])

    return graph


def construct_scip_model(graph, node_dict, node_mapping, link_dict, link_mapping, bidirected_link_mapping, demand_dict,
                         demand_model='undirected', link_model='undirected', link_capacity_model='modular',
                         fixed_charge_model='with', routing_model='integer', survival_model='one_plus_one'):
    """
    The function for creating the PySCIPOpt Model. It uses the networkx graph to create sets of edge_disjoint_paths
    for all demands. The node, link, and demand_dict are then used to construct the variables, constraints,
    and objectives.
    The formulations are taken from:
    @article{orlowski2010sndlib,
      title={SNDlib 1.0—Survivable network design library},
      author={Orlowski, Sebastian and Wess{\"a}ly, Roland and Pi{\'o}ro, Michal and Tomaszewski, Artur},
      journal={Networks: An International Journal},
      volume={55},
      number={3},
      pages={276--286},
      year={2010},
      publisher={Wiley Online Library}
    }
    All *_model kwargs change the formulation of the model.
    Args:
        graph (networkx Graph): The networkx graph Model
        node_dict (dict): Node relevant information dictionary
        node_mapping (dict): Mapping from networkx node IDs to the SNDLib ID
        link_dict (dict): Link relevant information dictionary
        link_mapping (dict): Mapping from networkx link IDs to the SNDLib ID
        bidirected_link_mapping (dict): Mapping from a bidirected link ID to opposite directional link
        demand_dict (dict): Demand relevant information dictionary
        demand_model (str): 'undirected' or 'directed'. Whether demands must flow along directed paths
        link_model (str): 'undirected', 'directed', 'bidirected'. The underlying network topology type
        link_capacity_model (str): 'modular' or 'explicit'. Modular admits integer combinations of capacities.
                                   'explicit' admits exactly one unit of a single capacity per link.
                                   'single_modular' admits only the smallest capacity module to be installed
                                   'linear' admits fractional combinations of capacities
        fixed_charge_model (str): 'with' or 'without'. Whether the setup cost of a link is accounted for
        routing_model (str): 'integer' or 'single_path'. Integer requires all paths to integer
                              multiples of rounding unit flow.
                             'single_path' requires exactly two paths to both admit exactly the required demand
        survival_model (str): 'one_plus_one' or 'none'. 'one_plus_one' adds set of survivability constraints
                              that the model can still route when any single link is destroyed

    Returns:
        The SCIP Model
    """

    # Generate the SCIP Model
    scip = Model()
    # ------------------------- VARIABLES -------------------------
    # The variables we use are per link (the capacity we purchase), and per path (the amount we route)
    # There are demand variables in the case of integer routing models and path binaries for single_path routing models

    # First the link specific variables
    link_inclusion_vars = {}
    link_negated_inclusion_vars = {}
    link_sum_capacities = {}
    link_capacity_vars = {link_name: {} for link_name in link_dict}
    vtype = 'C' if link_capacity_model == 'linear' else 'I'
    for link_name in link_dict:
        link = link_dict[link_name]
        link_inclusion_vars[link_name] = scip.addVar(name='link_{}'.format(link_name), vtype='B')
        link_negated_inclusion_vars[link_name] = scip.addVar(name='link_{}_negated'.format(link_name), vtype='B')
        link_sum_capacities[link_name] = scip.addVar(name='link_{}_sum_caps'.format(link_name), vtype='C')
        for i, capacity in enumerate(link['capacities']):
            link_capacity_vars[link_name][i] = scip.addVar(name='link_{}_capacity_{}'.format(link_name, i), vtype=vtype,
                                                           lb=0)

    # Now the demand specific variables (including path variables)
    path_dict = {}
    path_flow_vars = {demand_name: {} for demand_name in demand_dict}
    path_inclusion_vars = {demand_name: {} for demand_name in demand_dict}
    demand_vars = {}
    for demand_name in demand_dict:
        demand = demand_dict[demand_name]
        if routing_model == 'integer':
            demand_vars[demand_name] = scip.addVar(name='demand_{}'.format(demand_name), vtype='I', lb=1)
        from_node_i = node_dict[demand['from_node']]['i']
        to_node_i = node_dict[demand['to_node']]['i']
        path_dict[demand_name] = list(nx.edge_disjoint_paths(graph, from_node_i, to_node_i))
        for i, path in enumerate(path_dict[demand_name]):
            vtype = 'C' if routing_model == 'continuous' else 'I'
            path_flow_vars[demand_name][i] = scip.addVar(name='demand_{}_path_flow_{}'.format(demand_name, i),
                                                         vtype=vtype, lb=0)
            if routing_model == 'single_path':
                path_inclusion_vars[demand_name][i] = scip.addVar(name='demand_{}_path_inclusion_{}'.format(
                    demand_name, i), vtype='B')

    # ------------- CONSTRAINTS ----------------------

    # First add the constraints that ensure demand is met
    for demand_name in demand_dict:
        demand = demand_dict[demand_name]
        paths = path_dict[demand_name]
        # The sum of the amounts along each edge-disjoint path must equal 2x the demand
        rhs = demand['demand_value']
        if survival_model == 'one_plus_one':
            rhs *= 2
        if routing_model == 'integer':
            rhs = demand_vars[demand_name] * rhs
        scip.addCons(sum(path_flow_vars[demand_name][i] for i in range(len(paths))) == rhs,
                     name='demand_{}_routing'.format(demand_name))
        # Now add constraints for routing_model='single_path' (In this case the flow on the path must be 0 or rhs / 2)
        if routing_model == 'single_path':
            rhs = demand['demand_value']
            for i in range(len(paths)):
                scip.addCons(path_inclusion_vars[demand_name][i] * rhs == path_flow_vars[demand_name][i],
                             name='demand_{}_single_path_{}'.format(demand_name, i))

    # Now link capacities. In the case of explicit or single modular capacities add constraints.
    # Then get capacities into a single expression per link.
    link_capacity_constraint_expressions = {}
    link_flow_constraint_expressions = {}
    for link_name in link_dict:
        # TODO: Should it be mandatory to include arcs with pre-installed capacity?
        link = link_dict[link_name]

        if link_capacity_model == 'explicit':
            scip.addCons(sum(link_capacity_vars[link_name][i] for i in range(len(link['capacities']))) <= 1,
                         name='explicit_link_capacity_{}'.format(link_name))
        if link_capacity_model == 'single_modular' and len(link['capacities']) >= 2:
            min_capacity = None
            c_i = 0  # Not needed, but shuts my IDE up.
            for c_i, c in enumerate(link['capacities']):
                if min_capacity is None:
                    min_capacity = c_i
                if min_capacity is not None and c[0] < min_capacity:
                    min_capacity = c_i
            unusable_link_capacity_ids = [i for i in range(len(link['capacities'])) if i != c_i]
            scip.addCons(sum(link_capacity_vars[link_name][i] for i in unusable_link_capacity_ids) == 0,
                         name='single_modular_link_capacity_{}'.format(link_name))

        link_capacity_expression = link_inclusion_vars[link_name] * link['pre_installed_capacity']
        for i, capacity in enumerate(link['capacities']):
            link_capacity_expression += link_capacity_vars[link_name][i] * capacity[0]
        link_capacity_constraint_expressions[(link['from_node'], link['to_node'])] = link_capacity_expression
        link_flow_constraint_expressions[(link['from_node'], link['to_node'])] = 0
        if link_model == 'undirected':
            assert (link['to_node'], link['from_node']) not in link_capacity_constraint_expressions, link_name
            assert (link['to_node'], link['from_node']) not in link_flow_constraint_expressions, link_name
            link_capacity_constraint_expressions[(link['to_node'], link['from_node'])] = link_capacity_expression
            link_flow_constraint_expressions[(link['to_node'], link['from_node'])] = 0

    # Get the flows in a nice single expression per link (remember flows are per path)
    for demand_name in demand_dict:
        for path_i, path in enumerate(path_dict[demand_name]):
            for i in range(len(path) - 1):
                from_node = node_mapping[path[i]]
                to_node = node_mapping[path[i + 1]]
                link_flow_constraint_expressions[(from_node, to_node)] += \
                    demand_dict[demand_name]['routing_unit'] * path_flow_vars[demand_name][path_i]

    # Now add constraints that make sure capacity only exists if link is installed
    # Then add the most important constraints that flow is less than capacity
    for link_name in link_dict:
        link = link_dict[link_name]
        from_node = link['from_node']
        to_node = link['to_node']
        lhs = link_flow_constraint_expressions[(from_node, to_node)]
        rhs = link_capacity_constraint_expressions[(from_node, to_node)]
        scip.addCons(link_sum_capacities[link_name] == rhs, name='link_cap_sum_{}'.format(link_name))
        scip.addCons(link_inclusion_vars[link_name] == 1 - link_negated_inclusion_vars[link_name],
                     name='negated_link_{}'.format(link_name))
        scip.addConsSOS1([link_sum_capacities[link_name], link_negated_inclusion_vars[link_name]],
                         name='sos_cons_link_{}'.format(link_name))
        if link_model != 'bidirected' or link_name not in bidirected_link_mapping:
            # The constraint is that all flows over all paths that use this link, is less than the capacity of the link
            scip.addCons(lhs <= rhs, name='link_{}_capacity'.format(link_name))
        else:
            other_link_name = bidirected_link_mapping[link_name]
            other_from_node = link_dict[other_link_name]['from_node']
            other_to_node = link_dict[other_link_name]['to_node']
            other_rhs = link_capacity_constraint_expressions[(other_from_node, other_to_node)]
            # The constraint here is different from the original formulation. FLow is less than sum of capacity here
            scip.addCons(lhs <= rhs + other_rhs, name='link_{}_capacity'.format(link_name))

    # Now the survivability constraint
    if survival_model == 'one_plus_one':
        for demand_name in demand_dict:
            demand = demand_dict[demand_name]
            paths = path_dict[demand_name]
            lhs = demand['demand_value']
            if routing_model == 'integer':
                lhs = demand_vars[demand_name] * lhs
            for path_i in range(len(paths)):
                surviving_path_ids = [i for i in range(len(paths)) if i != path_i]
                scip.addCons(lhs <= sum(path_flow_vars[demand_name][i] for i in surviving_path_ids),
                             name='survival_demand_{}_path_{}'.format(demand_name, path_i))

    # --------------------------- OBJECTIVE ------------------------------------

    # Initialise the objective
    objective = 0
    for link_name in link_dict:
        link = link_dict[link_name]
        # Add pre-installed capacity costs
        objective += link_inclusion_vars[link_name] * link['pre_installed_capacity_cost']
        for capacity_i in range(len(link['capacities'])):
            # Add the cost of each module of installed capacity
            objective += link_capacity_vars[link_name][capacity_i] * link['capacities'][capacity_i][1]
        from_node = link['from_node']
        to_node = link['to_node']
        # TODO: For the bidirected case the paper says to take the max flow over a 'link', not the sum
        # Add the cost of each module of routed flow
        objective += link['routing_cost'] * link_flow_constraint_expressions[(from_node, to_node)]

        # Add fixed costs if link had set up costs
        if fixed_charge_model == 'with':
            objective += link['setup_cost'] * link_inclusion_vars[link_name]

    # Set the objective
    scip.setObjective(objective, sense='minimize')

    return scip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sndlib_dir', type=str)
    parser.add_argument('mps_dir', type=str)
    args = parser.parse_args()
    assert os.path.isdir(args.sndlib_dir)
    assert os.path.isdir(args.mps_dir)

    mps_dir = args.mps_dir
    sndlib_dir = args.sndlib_dir
    instance_names = os.listdir(sndlib_dir)

    for instance_name in instance_names:
        instance_path = os.path.join(sndlib_dir, instance_name, instance_name + '.txt')
        assert os.path.isfile(instance_path)
        model_dir = os.path.join(sndlib_dir, instance_name, 'models')
        assert os.path.isdir(model_dir)

        model_files = os.listdir(model_dir)

        with open(instance_path, 'r') as s:
            instance_path_data = s.readlines()

        for d_model in [('undirected', 'U'), ('directed', 'D')]:
            for l_model in [('undirected', 'U'), ('directed', 'D'), ('bidirected', 'B')]:
                nd, nm, ld, lm, bdlm, dd = read_network_data(instance_path_data, link_model=l_model[0])
                for l_c_model in [('explicit', 'E'), ('modular', 'M'), ('single_modular', 'S')]:
                    for f_c_model in [('with', 'Y'), ('without', 'N')]:
                        for r_model in [('integer', 'I'), ('single_path', 'S')]:
                            for s_model in [('one_plus_one', 'P'), ('none', 'N')]:
                                model_string = d_model[1] + '-' + l_model[1] + '-' + l_c_model[1] + '-' + \
                                               f_c_model[1] + '-' + r_model[1] + '-A-N-{}.txt'.format(s_model[1])
                                if os.path.isfile(os.path.join(model_dir, model_string)):
                                    print(instance_name, model_string)
                                    nx_graph = construct_graph(nd, ld, link_model=l_model[0])
                                    m = construct_scip_model(nx_graph, nd, nm, ld, lm, bdlm, dd,
                                                             demand_model=d_model[0],
                                                             link_model=l_model[0],
                                                             link_capacity_model=l_c_model[0],
                                                             fixed_charge_model=f_c_model[0],
                                                             routing_model=r_model[0],
                                                             survival_model=s_model[0])
                                    # m.optimize()
                                    mps_file = instance_name + '-' + model_string[:-4] + '.mps'
                                    m.writeProblem(os.path.join(mps_dir, mps_file))
                                    # quit()
