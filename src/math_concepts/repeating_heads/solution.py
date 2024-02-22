def repeating_heads(n, x):
    bet_size = 100
    heads_chance = 1 / 2

    trial_win_chance = heads_chance**n
    trial_lose_chance = 1 - trial_win_chance

    repeated_trial_lose_chance = trial_lose_chance**x
    repeated_trial_win_chance = 1 - repeated_trial_lose_chance

    break_even_payout = bet_size / repeated_trial_win_chance
    return [repeated_trial_win_chance * 100, break_even_payout]


print(repeating_heads(1, 2))
