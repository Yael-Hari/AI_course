from torch.distributed.pipeline.sync.skip import pop

from utils import orientations, vector_add, powerset
import ast
from typing import Tuple
import itertools
import networkx as nx
from copy import deepcopy
import numpy as np


ids = ["316375872", "206014482"]


class OptimalTaxiAgent:
    def __init__(self, initial: dict):
        self.initial = initial
        # we shouldnt add anything to th states! because we get the state from user in "act" func
        self.states_graph = nx.DiGraph()
        self.index_to_state = dict()
        self.state_to_index = dict()

        self.set_all_possible_states_graph()


    def act(self, state):
        raise NotImplemented

    def value_iteration_algo(self):
        pass

    def set_all_possible_next_states_and_probs(self, state):
        """
        building the networkx digraph: its actually a tree
            nodes:
                - have the states
                - have the multiplier:
            edges:
                - have the weights: -100 for
        """
        # get all actions like in HW 1 + terminate + reset
        actions = self.actions(state)

        # get all states resulting from the all actions --> deterministic
        deterministic_act_state_prob_tuples = []
        for act in actions:
            result_state = self.result(state, act)
            probs_to_not_change_goal = np.prod([
                1 - pass_dict['prob_change_goal'] for pass_dict in state['passengers'].values()
            ])

            deterministic_act_state_prob_tuples.append((act, result_state, probs_to_not_change_goal))

        # get all states resulting from the all the actions
        # but now the passengers can change their destination --> stochastic
        stochastic_action_result_state_tuples = []
        all_passengers_subsets = powerset(state['passengers'].keys())
        for pass_subset in all_passengers_subsets:
            # only the subset are changing their destinations
            # add (action, next_state) tuples to stochastic_action_result_state_tuples
            # also need to add to state what is the probability to get there
            for act, state, prob in deterministic_act_state_prob_tuples:
                ...     # TODO

    def get_gas_station_list(self, map, h, w):
        l = []
        for i in range(h):
            for j in range(w):
                if map[i][j] == "G":
                    l.apend((i, j))
        return l

    def generate_locations(self, state: dict) -> dict:
        # get new locations by:
        # current location + one step in legal orientation (EAST, NORTH, WEST, SOUTH)
        possible_locations_by_taxi = dict()
        for taxi_name, taxi_dict in state["taxis"].items():
            curr_location = taxi_dict["location"]
            possible_locations = [
                vector_add(curr_location, orient) for orient in orientations
            ]
            possible_locations_by_taxi[taxi_name] = possible_locations

        return possible_locations_by_taxi

    def get_legal_moves_on_map(self, state: dict) -> dict:
        legal_locations_by_taxi = {}
        possible_locations_by_taxi = self.generate_locations(state)
        for taxi_name, taxi_dict in state["taxis"].items():
            # 1. check fuel > 0
            legal_locations = []
            if taxi_dict["fuel"] > 0:
                map_size_height = state["map_size_height"]
                map_size_width = state["map_size_width"]
                map_matrix = state["map"]

                possible_locations = possible_locations_by_taxi[taxi_name]
                for new_location in possible_locations:
                    x, y = new_location
                    # 2. check that the taxi doesn't get out of the map
                    # 3. check that the taxi is on a passable tile
                    if (0 <= x < map_size_height) and (0 <= y < map_size_width):
                        if map_matrix[x][y] != "I":
                            legal_locations.append(new_location)
            legal_locations_by_taxi[taxi_name] = legal_locations
        return legal_locations_by_taxi

    def get_legal_refuel(self, state: dict) -> dict:
        # Refueling can be performed only at gas stations
        legal_refuels_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            map_matrix = state["map"]
            x, y = taxi_dict["location"]  # current location of taxi
            # check that the location on map is "G"
            try:
                legal_refuel = (map_matrix[x][y] == "G")  # bool
            except Exception as e:
                print()
            legal_refuels_by_taxi[taxi_name] = legal_refuel
        return legal_refuels_by_taxi

    def get_legal_pick_up(self, state: dict) -> dict:
        # Pick up passengers if they are on the same tile as the taxi.
        legal_pickups_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            capacity = taxi_dict["capacity"]
            taxi_loc = taxi_dict['location']
            passengers_in_taxi_list = taxi_dict["passengers_list"]
            legal_pickups = []
            # The number of passengers in the taxi has to be < taxi’s capacity.
            if capacity > 0:
                for passenger_name, passenger_dict in state["passengers"].items():
                    # check that location of taxi is the same as location of the passenger
                    if (taxi_dict["location"] == passenger_dict["location"]) & (
                            passenger_dict["location"] != passenger_dict["destination"]
                    ):
                        legal_pickups.append(passenger_name)
            legal_pickups_by_taxi[taxi_name] = legal_pickups
        return legal_pickups_by_taxi

    def get_legal_drop_off(self, state: dict) -> dict:
        # The passenger can only be dropped off on his destination tile
        # and will refuse to leave the vehicle otherwise.
        legal_drop_offs_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            legal_drop_offs = []
            # go over the passengers that's on the curr taxi
            for passenger_name in taxi_dict["passengers_list"]:
                passenger_dict = state["passengers"][passenger_name]
                # check that location of taxi is the same as destination of the passenger
                if taxi_dict["location"] == passenger_dict["destination"]:
                    legal_drop_offs.append(passenger_name)
            legal_drop_offs_by_taxi[taxi_name] = legal_drop_offs
        return legal_drop_offs_by_taxi

    def actions(self, state: dict) -> Tuple[Tuple[Tuple]]:
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        # -----------------------------------------------------------------
        # Atomic Actions: ["move", "pick_up", "drop_off", "refuel", "wait"]
        # explicit syntax:
        # (“move”, “taxi_name”, (x, y))
        # (“pick up”, “taxi_name”, “passenger_name”
        # (“drop off”, “taxi_name”, “passenger_name”)
        # ("refuel", "taxi_name")
        # ("wait", "taxi_name")

        # Full Action - a tuple with action for each taxi
        # Example: ((“move”, “taxi 1”, (1, 2)),
        #           (“wait”, “taxi 2”),
        #           (“pick up”, “very_fancy_taxi”, “Yossi”))
        # -----------------------------------------------------------------

        # For each taxi get Possible Atomic Actions

        legal_locations_by_taxi = self.get_legal_moves_on_map(
            state
        )  # DICT[taxi_name: list of (x,y) locations]
        legal_pickups_by_taxi = self.get_legal_pick_up(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_drop_offs_by_taxi = self.get_legal_drop_off(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_refuels_by_taxi = self.get_legal_refuel(
            state
        )  # DICT[taxi_name: True / False]

        # -----------------------------------------------------------------
        # Get Atomic Actions with right syntax
        atomic_actions_lists = []
        for taxi_name in state["taxis"].keys():
            atomic_actions = [("wait", taxi_name)]
            for location in legal_locations_by_taxi[taxi_name]:
                atomic_actions.append(("move", taxi_name, location))
            for passenger_name in legal_pickups_by_taxi[taxi_name]:
                atomic_actions.append(("pick up", taxi_name, passenger_name))
            for passenger_name in legal_drop_offs_by_taxi[taxi_name]:
                atomic_actions.append(("drop off", taxi_name, passenger_name))
            if legal_refuels_by_taxi[taxi_name]:
                atomic_actions.append(("refuel", taxi_name))
            atomic_actions_lists.append(atomic_actions)

        # -----------------------------------------------------------------
        # Get Actions - all permutations of atomic actions
        actions = list(itertools.product(*atomic_actions_lists))
        all_wait_action = tuple(
            [("wait", taxi_name) for taxi_name in state["taxis"].keys()]
        )
        assert all_wait_action in actions
        actions.remove(all_wait_action)

        # -----------------------------------------------------------------
        # For each action - Check That Taxis Don't Clash with each other
        #   == not going to the same location (therefore cannot pickup the same passenger)
        n_taxis = state["n_taxis"]
        if n_taxis > 1:
            legal_actions = []
            for action in actions:
                taxis_next_locations = []
                for atomic_action in action:  # TODO: NOTE changed from atomic_actions_lists to action
                    action_type = atomic_action[0]
                    taxi_name = atomic_action[1]
                    taxi_curr_location = state["taxis"][taxi_name]["location"]
                    if action_type == "move":
                        taxi_next_location = atomic_action[2]
                    else:
                        taxi_next_location = taxi_curr_location
                    taxis_next_locations.append(taxi_next_location)
                # check if there are 2 taxis in the same location
                legal_action = len(set(taxis_next_locations)) == n_taxis
                if legal_action:
                    legal_actions.append(action)
        else:  # n_taxis == 1 --> no clashing between taxis
            legal_actions = actions

        # -----------------------------------------------------------------
        # The result should be a tuple (or other iterable) of actions
        # or a string: 'reset', 'terminate'
        # as defined in the problem description file
        legal_actions.append('reset')
        legal_actions.append('terminate')

        return tuple(legal_actions)

    def result(self, state: dict, action: Tuple[Tuple]) -> dict:
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        result_state = deepcopy(state)

        for action_tuple in action:
            result_state = self._execute_action_tuple(result_state, action_tuple)

        return result_state

    def _execute_action_tuple(self, state: dict, action_tuple: Tuple) -> dict:
        """
        input: state dict, and an action tuple like: (“move”, “taxi_name”, (x, y))
        output: the state dict after preforming the action
        """
        actions_possible = MOVE, PICK_UP, DROP_OFF, REFUEL, WAIT = (
            "move",
            "pick up",
            "drop off",
            "refuel",
            "wait",
        )

        action_type = action_tuple[0]
        taxi_name = action_tuple[1]

        result_state = state.copy()

        # check input is legal
        assert (
            action_type in actions_possible
        ), f"{action_type} is not a possible action!"

        if action_type == MOVE:  # (“move”, “taxi_name”, (x, y))
            assert (
                len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            # taxi updates:
            #   fuel -= 1
            result_state["taxis"][taxi_name]["fuel"] -= 1
            #   location
            future_location = action_tuple[2]
            result_state["taxis"][taxi_name]["location"] = future_location

        elif action_type == PICK_UP:  # (“pick up”, “taxi_name”, “passenger_name”)
            assert (
                len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            passenger_name = action_tuple[2]

            # Taxi updates:
            #   taxi capacity -= 1
            result_state["taxis"][taxi_name]["capacity"] -= 1
            #   add passenger name to passengers_list of taxi
            result_state["taxis"][taxi_name]["passengers_list"].append(passenger_name)
            # Problem updates:
            #   n_picked_undelivered += 1
            result_state["n_picked_undelivered"] += 1
            #   n_unpicked -= 1
            result_state["n_unpicked"] -= 1
            # Passenger updates:
            #   update "in_taxi" of passenger to name of taxi
            result_state["passengers"][passenger_name]["in_taxi"] = taxi_name

        elif action_type == DROP_OFF:  # (“drop off”, “taxi_name”, “passenger_name”)
            assert (
                len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            passenger_name = action_tuple[2]
            # Taxi updates:
            #   taxi capacity += 1
            result_state["taxis"][taxi_name]["capacity"] += 1
            #   remove passenger name from passengers_list of taxi
            result_state["taxis"][taxi_name]["passengers_list"].remove(passenger_name)
            # Problem updates:
            #   n_picked_undelivered -= 1
            result_state["n_picked_undelivered"] -= 1
            #   n_delivered += 1
            result_state["n_delivered"] += 1
            # Passenger updates:
            #   passenger location = taxi location
            result_state["passengers"][passenger_name]["location"] = result_state["passengers"][passenger_name]['destination']
            #   update "in_taxi" of passenger to False
            result_state["passengers"][passenger_name]["in_taxi"] = False

        elif action_type == REFUEL:  # ("refuel", "taxi_name")
            assert (
                len(action_tuple) == 2
            ), f"len of action_tuple should be 2: {action_tuple}"
            # taxi updates:
            #   fuel = max_fuel
            result_state["taxis"][taxi_name]["fuel"] = result_state["taxis"][taxi_name][
                "max_fuel"
            ]

        elif action_type == WAIT:  # ("wait", "taxi_name")
            assert (
                len(action_tuple) == 2
            ), f"len of action_tuple should be 2: {action_tuple}"
            pass

        return result_state

class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented


def dict_to_str(d: dict) -> str:
    d_str = str(d)
    return d_str


def str_to_dict(s: str) -> dict:
    j_dict = ast.literal_eval(s)
    return j_dict


if __name__ == '__main__':
    state = \
        {
            "optimal": True,
            "map": [['P', 'P', 'P'],
                    ['P', 'G', 'P'],
                    ['P', 'P', 'P']],
            "taxis": {'taxi 1': {"location": (0, 0), "fuel": 10, "capacity": 1}},
            "passengers": {'Dana': {"location": (2, 2),
                                    "destination": (0, 0),
                                    "possible_goals": ((0, 0), (2, 2)),
                                    "prob_change_goal": 0.1}},
            "turns to go": 100
        }
    agent = OptimalTaxiAgent(initial=state)
    agent.get_all_possible_next_states(state)