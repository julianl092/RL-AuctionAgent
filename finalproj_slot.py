
import sys
import logging
import numpy as np
from gsp import GSP
from util import argmax_index
from math import floor, ceil
from random import random, randint
from collections import Counter

class Finalproj_slot:
    """Balanced bidding agent"""
    def __init__(self, id, value, budget):
        self.id = id
        self.value = value

        self.num_rounds = 48
        self.num_slots = 2

        self.budget_buckets = 10
        self.bids = [[[40] for i in range(self.num_rounds) ] for i in range(self.num_slots + 1)]
        
        self.clicks = np.zeros((self.num_slots + 1, self.num_rounds))
        self.clicks.fill(50)
        
        self.success_percentile = 80
        
        self.alpha = 0.05
        self.gamma = 1

        self.epsilon = 1
        self.eps_decay_rate = 0.975
        self.min_epsilon = 0.05

        self.prev_target = None
        self.budget_original = budget
        self.budget = budget

        self.broke = []
        self.target_success = []
        self.targets = []
        
        self.Q_table = np.zeros((self.budget_buckets + 1, self.num_rounds, self.num_slots + 1))

    def initial_bid(self, reserve):
        # Update epsilon 
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.eps_decay_rate

        # Reset budget
        self.budget = self.budget_original
        
        # Determine best slot to target in current state using Q-table
        best_action = np.random.choice(np.flatnonzero(self.Q_table[self.budget_buckets][0] == self.Q_table[self.budget_buckets][0].max()))

        # With probability epsilon, discard the best action and choose a random slot to target
        if random() < 0.5:
            if random() < (1 / self.num_slots):
                new_action = self.num_slots
            else:
                new_action = randint(0, self.num_slots - 1)
        else:
            new_action = best_action

        self.prev_target = new_action

        # Based on bids from past days in this same time period, construct bid to target selected slot.
        if new_action == self.num_slots:
            return 0 
        else:
            try:
                # Based on history, calculate the minimum bid required to secure the targeted slot
                min_bid = np.percentile(self.bids[new_action + 1][0], self.success_percentile)

                # Construct a balanced bid based from the past minimum bids
                if min_bid >= self.value:
                    bid = reserve
                else:
                    if new_action > 0:
                        pos_k = self.clicks[0][new_action - 1]
                        pos_j = self.clicks[1][new_action]
                        bid = self.value - (pos_j / pos_k) * (self.value - min_bid)
                    else:
                        bid = min_bid
                return bid
            except:
                return reserve

    def slot_info(self, t, history, reserve):
        """Compute the following for each slot, assuming that everyone else
        keeps their bids constant from the previous rounds.

        Returns list of tuples [(slot_id, min_bid, max_bid)], where
        min_bid is the bid needed to tie the other-agent bid for that slot
        in the last round.  If slot_id = 0, max_bid is 2* min_bid.
        Otherwise, it's the next highest min_bid (so bidding between min_bid
        and max_bid would result in ending up in that slot)
        """
        prev_round = history.round(t-1)
        other_bids = [a_id_b for a_id_b in prev_round.bids if a_id_b[0] != self.id]

        clicks = prev_round.clicks
        def compute(s):
            (min, max) = GSP.bid_range_for_slot(s, clicks, reserve, other_bids)
            if max == None:
                max = 2 * min
            return (s, min, max)
            
        info = list(map(compute, list(range(len(clicks)))))
        return info


    def expected_utils(self, t, history, reserve):
        """
        Figure out the expected utility of bidding such that we win each
        slot, assuming that everyone else keeps their bids constant from
        the previous round.

        returns a list of utilities per slot.
        """
        utilities = []
        prev_round = history.round(t - 1)
        other_bids = [a_id_b for a_id_b in prev_round.bids if a_id_b[0] != self.id]
       
        other_bids.sort(key = lambda x: x[1], reverse=True)
        prev_round.clicks.sort(reverse=True)
        
        for i in range(len(prev_round.clicks)):
            # case where slots >= bids
            if i >= len(other_bids):
                pos = prev_round.clicks[i]
                utilities.append(pos * (self.value - reserve))
            else:
                pos = prev_round.clicks[i]
                a_bid = other_bids[i][1]
                utilities.append(pos * (self.value - a_bid))

        return utilities

    def target_slot(self, t, history, reserve):
        """Figure out the best slot to target, assuming that everyone else
        keeps their bids constant from the previous rounds.

        Returns (slot_id, min_bid, max_bid), where min_bid is the bid needed to tie
        the other-agent bid for that slot in the last round.  If slot_id = 0,
        max_bid is min_bid * 2
        """
        i = argmax_index(self.expected_utils(t, history, reserve))
        info = self.slot_info(t, history, reserve)
        return info[i]

    def bid(self, t, history, reserve):
        
        prev_round = history.round(t-1)

        ## Add previous round's bids and clicks for each slot to our records
        prev_round.bids.sort(key = lambda x: x[1], reverse=True)
        for rank, bid in enumerate(prev_round.bids):
            self.bids[rank][t - 1].append(bid[1])

        for rank, clicks in enumerate(prev_round.clicks):
            self.clicks[rank][t - 1] = clicks

        # Find previous slot occupied & reward gained
        try:
            prev_slot = prev_round.occupants.index(self.id)
            payment = prev_round.slot_payments[prev_slot]
            reward = prev_round.clicks[prev_slot] * (self.value - prev_round.per_click_payments[prev_slot])
        except:
            prev_slot = self.num_slots
            reward = 0
            payment = 0

        self.target_success.append(self.prev_target == prev_slot)

        # Calculate discretized previous budget & new budget
        prev_budget = max(0, ceil(self.budget_buckets * self.budget / self.budget_original))
        self.budget -= payment
        new_budget = max(0, ceil(self.budget_buckets * self.budget / self.budget_original))

        # Update Q-value for previous round's state (budget, time) & action in Q-table
        self.Q_table[prev_budget][t - 1][prev_slot] *= 1 - self.alpha
        self.Q_table[prev_budget][t - 1][prev_slot] += self.alpha * (reward + self.gamma * np.max(self.Q_table[new_budget][t]))

        if prev_budget == 1 and new_budget == 0:
            self.broke.append(t)
        

        # Determine best slot to target in current state using Q-table
        best_action = np.random.choice(np.flatnonzero(self.Q_table[new_budget][t] == self.Q_table[new_budget][t].max()))
    
        # With probability epsilon, discard the best slot and target a random slot for exploration
        if random() < self.epsilon:
            if random() < (1 / self.num_slots):
                new_action = self.num_slots
            else:
                new_action = randint(0, self.num_slots - 1)
        else:
            new_action = best_action

        self.prev_target = new_action
        self.targets.append(new_action)

        # Based on bids from past days in this same time period, construct bid to target selected slot.
        if new_action == self.num_slots:
            return 0 
        else:
            min_bid = np.percentile(self.bids[new_action + 1][t], self.success_percentile)

            if min_bid >= self.value:
                bid = reserve
            else:
                if new_action > 0:
                    pos_k = self.clicks[new_action - 1][t]
                    pos_j = self.clicks[new_action][t]
                    bid = self.value - (pos_j / pos_k) * (self.value - min_bid)
                else:
                    bid = min_bid
            return bid
        
    def __repr__(self):
        return "%s(id=%d, value=%d)" % (
            self.__class__.__name__, self.id, self.value)


