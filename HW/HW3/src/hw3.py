import itertools
import math
import random

from sample_agent import Agent as RivalAgent
from Simulator import Simulator

IDS = ["316375872", "206014482"]


class Agent:
    def __init__(self, initial_state, player_number):
        """
        ! Should finish within 60 seconds.
        """
        random.seed(42)
        self.ids = IDS
        self.player_number = player_number
        self.my_taxis = []
        self.simulator = Simulator(initial_state)
        self.max_turns_to_go = initial_state["turns to go"]
        self.curr_turns_to_go = self.max_turns_to_go
        for taxi_name, taxi in initial_state["taxis"].items():
            if taxi["player"] == player_number:
                self.my_taxis.append(taxi_name)
        self.rival_agent = RivalAgent(initial_state, player_number)

    # ~~~~~~~~~~~~~~~~~~~~~~ UCT steps

    def selection(self, node):
        """
        must select a node to expand
        """

        while node.fully_expanded() and node.children:
            nodes_sorted_desc = sorted(
                node.children, key=lambda x: self.UCT_score_function(x), reverse=True
            )  # reverse True for sort descending
            legal_action = False
            index_of_highest_score = 0
            while not legal_action:
                node = nodes_sorted_desc[index_of_highest_score]
                # get the action that got us from parent to this child
                action = node.actions_history[-1]
                try:  # go to this child and update simulator
                    self.play_action_in_simulator(node, action)
                    # action updates
                    legal_action = True
                    # node.update_untried_actions(action)
                except ValueError as e:
                    index_of_highest_score += 1
                    raise e

        return node

    def expansion(self, node_to_expand):
        """
        must update the tree with a new node
        """
        if not node_to_expand.fully_expanded():
            if node_to_expand.untried_actions:
                # set node's all possible actions if needed
                state = self.simulator.get_state()
                untried_actions = self.get_untried_actions(node=node_to_expand, state=state)
                # rand action
                action = self.choose_action_by_heuristic(untried_actions)
                self.play_action_in_simulator(node_to_expand, action)
                # create child node
                child_state = self.simulator.get_state()
                child_actions_history = tuple(
                    list(node_to_expand.actions_history) + [action]
                )
                child_possible_actions = self.get_all_possible_actions(child_state)
                child = node_to_expand.add_child(
                    child_actions_history, child_possible_actions
                )
            else:
                raise RuntimeError("node already tried all possible actions")
        else:
            raise RuntimeError("node is already fully expanded!")
        return child

    def simulation(self, node):
        """
        must perform a simulation from a new node
        """
        while self.curr_turns_to_go:
            child = self.expansion(node_to_expand=node)
            node = child

        return self.simulator.score[f"player {self.player_number}"]   # reward

    def backpropagation(self, reward, node):
        """
        must update the UCT tree with the new information.
        updating all the way up from the given node --> his parent --> his parent
            --> ... --> root
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    # ~~~~~~~~~~~~~~~~~~~~~~

    def UCT_score_function(self, node):
        if node.n_visits == 0:
            score = float("inf")
        else:
            score = node.mean_reward + math.sqrt(
                2 * math.log(node.parent.n_visits) / node.n_visits
            )
        return score

    def act(self, state):
        """
        input: current state of the game.
        output: best action.
        ! Should finish within 5 seconds
        """
        best_action = self.get_best_action_by_UCT(state, n_iterations=10)
        return best_action

    def get_best_action_by_UCT(self, state, n_iterations):
        """
        creating tree and run UCT.
        in each iteration:
            1. inializes the simulator
            2. select a node to exapnd
            3. simulate a full game against the rival
            4. update the mean rewards of the nodes in the tree
        """
        # init root node
        root = Node(actions_history=tuple(), possible_actions=self.get_all_possible_actions(state))
        for _ in range(n_iterations):
            # init iteration
            node = root
            self.simulator = Simulator(state)
            self.curr_turns_to_go = self.max_turns_to_go
            # run
            node_to_expand = self.selection(node)
            new_node = self.expansion(node_to_expand)
            reward = self.simulation(new_node)
            self.backpropagation(reward, new_node)

        return self.get_best_action(root, state)

    def get_best_action(self, root, state):
        # get best action after check it is legal
        # sort nodes by UCT_score_function
        nodes_sorted_desc = sorted(
            root.children, key=lambda x: self.UCT_score_function(x), reverse=True
        )  # reverse True for sort descending
        # check if this action is legal from the input state
        self.simulator = Simulator(state)
        legal_action = False
        index_of_best_node = 0
        while not legal_action:
            best_node = nodes_sorted_desc[index_of_best_node]
            # get the action that got us from parent to this child
            best_action = best_node.actions_history[-1]
            legal_action = self.simulator.check_if_action_legal(
                best_action, self.player_number
            )
            index_of_best_node += 1
        return best_action

    def play_action_in_simulator(self, node, next_action):
        self.simulator.act(next_action, self.player_number)
        node.update_untried_actions(next_action)
        self.curr_turns_to_go -= 1
        next_state = self.simulator.get_state()
        self.rival_agent.act(next_state)  # TODO: update also rival action??

    def get_all_possible_actions(self, state):
        """
        get all possible actions from given state.
        NOTE: the code of the for loop is the same like in sample_agent.py
        """
        actions = {}
        self.simulator.set_state(state)
        for taxi in self.my_taxis:
            actions[taxi] = set()
            neighboring_tiles = self.simulator.neighbors(
                state["taxis"][taxi]["location"]
            )
            for tile in neighboring_tiles:
                actions[taxi].add(("move", taxi, tile))
            if state["taxis"][taxi]["capacity"] > 0:
                for passenger in state["passengers"].keys():
                    if (
                            state["passengers"][passenger]["location"]
                            == state["taxis"][taxi]["location"]
                    ):
                        actions[taxi].add(("pick up", taxi, passenger))
            for passenger in state["passengers"].keys():
                if (
                        state["passengers"][passenger]["destination"]
                        == state["taxis"][taxi]["location"]
                        and state["passengers"][passenger]["location"] == taxi
                ):
                    actions[taxi].add(("drop off", taxi, passenger))
            actions[taxi].add(("wait", taxi))

        # get whole actions
        actions = list(itertools.product(*actions.values()))
        # filter out illegal actions
        actions = [
            action
            for action in actions
            if self.simulator.check_if_action_legal(action, self.player_number)
        ]
        return actions

    def get_untried_actions(self, node, state):
        node.possible_actions = self.get_all_possible_actions(state)
        node.untried_actions = node.possible_actions
        return node.untried_actions

    def choose_action_by_heuristic(self, actions):
        while True:
            whole_action = []
            num_taxis = len(self.my_taxis)
            actions_lists_by_taxis = [
                [act[i] for act in actions] for i in range(num_taxis)
            ]
            for atomic_actions_of_taxi in actions_lists_by_taxis:
                for action in atomic_actions_of_taxi:
                    if action[0] == "drop off":
                        whole_action.append(action)
                        break
                    if action[0] == "pick up":
                        whole_action.append(action)
                        break
                else:
                    whole_action.append(random.choice(list(atomic_actions_of_taxi)))
            whole_action = tuple(whole_action)
            if self.simulator.check_if_action_legal(whole_action, self.player_number):
                return whole_action


class UCTAgent:
    def __init__(self, initial_state, player_number):
        """
        ! Should finish within 60 seconds.
        """
        random.seed(42)
        self.ids = IDS
        self.player_number = player_number
        self.my_taxis = []
        self.simulator = Simulator(initial_state)
        self.max_turns_to_go = initial_state["turns to go"]
        self.curr_turns_to_go = self.max_turns_to_go
        for taxi_name, taxi in initial_state["taxis"].items():
            if taxi["player"] == player_number:
                self.my_taxis.append(taxi_name)
        self.rival_agent = RivalAgent(initial_state, player_number)

    # ~~~~~~~~~~~~~~~~~~~~~~ UCT steps

    def selection(self, node):
        """
        must select a node to expand
        """

        while node.fully_expanded() and node.children:
            nodes_sorted_desc = sorted(
                node.children, key=lambda x: self.UCT_score_function(x), reverse=True
            )  # reverse True for sort descending
            legal_action = False
            index_of_highest_score = 0
            while not legal_action:
                node = nodes_sorted_desc[index_of_highest_score]
                # get the action that got us from parent to this child
                action = node.actions_history[-1]
                try:  # go to this child and update simulator
                    self.play_action_in_simulator(node, action)
                    # action updates
                    legal_action = True
                    # node.update_untried_actions(action)
                except ValueError as e:
                    index_of_highest_score += 1
                    raise e

        return node

    def expansion(self, node_to_expand):
        """
        must update the tree with a new node
        """
        if not node_to_expand.fully_expanded():
            if node_to_expand.untried_actions:
                # set node's all possible actions if needed
                state = self.simulator.get_state()
                untried_actions = self.get_untried_actions(node=node_to_expand, state=state)
                # rand action
                action = random.choice(untried_actions)
                self.play_action_in_simulator(node_to_expand, action)
                # create child node
                child_state = self.simulator.get_state()
                child_actions_history = tuple(
                    list(node_to_expand.actions_history) + [action]
                )
                child_possible_actions = self.get_all_possible_actions(child_state)
                child = node_to_expand.add_child(
                    child_actions_history, child_possible_actions
                )
            else:
                raise RuntimeError("node already tried all possible actions")
        else:
            raise RuntimeError("node is already fully expanded!")
        return child

    def simulation(self, node):
        """
        must perform a simulation from a new node
        """
        while self.curr_turns_to_go:
            # next_action = random.choice(node.untried_actions)
            # self.play_action_in_simulator(node, next_action=next_action)
            child = self.expansion(node_to_expand=node)
            node = child

        return self.simulator.score[f"player {self.player_number}"]   # reward

    def backpropagation(self, reward, node):
        """
        must update the UCT tree with the new information.
        updating all the way up from the given node --> his parent --> his parent
            --> ... --> root
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    # ~~~~~~~~~~~~~~~~~~~~~~

    def UCT_score_function(self, node):
        if node.n_visits == 0:
            score = float("inf")
        else:
            score = node.mean_reward + math.sqrt(
                2 * math.log(node.parent.n_visits) / node.n_visits
            )
        return score

    def act(self, state):
        """
        input: current state of the game.
        output: best action.
        ! Should finish within 5 seconds
        """
        best_action = self.get_best_action_by_UCT(state, n_iterations=10)
        return best_action

    def get_best_action_by_UCT(self, state, n_iterations):
        """
        creating tree and run UCT.
        in each iteration:
            1. inializes the simulator
            2. select a node to exapnd
            3. simulate a full game against the rival
            4. update the mean rewards of the nodes in the tree
        """
        # init root node
        root = Node(actions_history=tuple(), possible_actions=self.get_all_possible_actions(state))
        for _ in range(n_iterations):
            # init iteration
            node = root
            self.simulator = Simulator(state)
            self.curr_turns_to_go = self.max_turns_to_go
            # run
            node_to_expand = self.selection(node)
            new_node = self.expansion(node_to_expand)
            reward = self.simulation(new_node)
            self.backpropagation(reward, new_node)

        return self.get_best_action(root, state)

    def get_best_action(self, root, state):
        # get best action after check it is legal
        # sort nodes by UCT_score_function
        nodes_sorted_desc = sorted(
            root.children, key=lambda x: self.UCT_score_function(x), reverse=True
        )  # reverse True for sort descending
        # check if this action is legal from the input state
        self.simulator = Simulator(state)
        legal_action = False
        index_of_best_node = 0
        while not legal_action:
            best_node = nodes_sorted_desc[index_of_best_node]
            # get the action that got us from parent to this child
            best_action = best_node.actions_history[-1]
            legal_action = self.simulator.check_if_action_legal(
                best_action, self.player_number
            )
            index_of_best_node += 1
        return best_action

    def play_action_in_simulator(self, node, next_action):
        self.simulator.act(next_action, self.player_number)
        node.update_untried_actions(next_action)
        self.curr_turns_to_go -= 1
        next_state = self.simulator.get_state()
        self.rival_agent.act(next_state)  # TODO: update also rival action??

    def get_all_possible_actions(self, state):
        """
        get all possible actions from given state.
        NOTE: the code of the for loop is the same like in sample_agent.py
        """
        actions = {}
        self.simulator.set_state(state)
        for taxi in self.my_taxis:
            actions[taxi] = set()
            neighboring_tiles = self.simulator.neighbors(
                state["taxis"][taxi]["location"]
            )
            for tile in neighboring_tiles:
                actions[taxi].add(("move", taxi, tile))
            if state["taxis"][taxi]["capacity"] > 0:
                for passenger in state["passengers"].keys():
                    if (
                            state["passengers"][passenger]["location"]
                            == state["taxis"][taxi]["location"]
                    ):
                        actions[taxi].add(("pick up", taxi, passenger))
            for passenger in state["passengers"].keys():
                if (
                        state["passengers"][passenger]["destination"]
                        == state["taxis"][taxi]["location"]
                        and state["passengers"][passenger]["location"] == taxi
                ):
                    actions[taxi].add(("drop off", taxi, passenger))
            actions[taxi].add(("wait", taxi))

        # get whole actions
        actions = list(itertools.product(*actions.values()))
        # filter out illegal actions
        actions = [
            action
            for action in actions
            if self.simulator.check_if_action_legal(action, self.player_number)
        ]
        return actions

    def get_untried_actions(self, node, state):
        node.possible_actions = self.get_all_possible_actions(state)
        node.untried_actions = node.possible_actions
        return node.untried_actions


class Node:
    def __init__(
        self, actions_history: tuple, possible_actions, parent=None
    ):
        """
        @ Tuple actions_history: tuple of actions that got us to this node
            example: (act1, act3, act4)
        """
        self.parent = parent
        self.children = []

        self.possible_actions = possible_actions
        self.actions_history = actions_history
        self.untried_actions = possible_actions

        self.n_visits = 0
        self.mean_reward = 0

    def add_child(self, child_actions_history, child_possible_actions):
        child = Node(actions_history=child_actions_history, possible_actions=child_possible_actions, parent=self)
        self.children.append(child)
        return child

    def update(self, reward):
        """
        update mean reward = empirical mean of rewards from arm i
        """
        # NOTE - need to update n_visits after updating the new mean reward
        curr_mean = self.mean_reward
        self.mean_reward = curr_mean + (1 / (self.n_visits + 1)) * (reward - curr_mean)
        self.n_visits += 1

    def fully_expanded(self):
        # check if all children have been visited
        return len(self.children) == len(self.possible_actions)

    def __repr__(self):
        return f"Node; {self.n_visits=}; {self.mean_reward=}; children: {len(self.children)}"

    def update_untried_actions(self, action):
        if action in self.untried_actions:   # TODO is this ok ???
            self.untried_actions.remove(action)
