# This function returns the expected score betweena  player and their opponent
#     rating_1: rating of player
#     rating_2: rating of opponent
#     elo_scale: How many points above the other one player needs to be
#        to have a 10x chance of winning the game
function expected_score(rating_1, rating_2; elo_scale = 400)
    return 1/(1+10^((rating_2 - rating_1)/elo_scale))
end

# This function rates a single player and returns the new rating
#     rating_1: rating of player
#     rating_2: rating of opponent
#     outcome: the actual outcome of a given match
#     elo_scale: How many points above the other one player needs to be
#        to have a 10x chance of winning the game
#     outcome_scale: the amount by which each game affects the player's rating
function rate(rating_1, rating_2, outcome; elo_scale=400,
              outcome_scale=15)

    return rating_1 + outcome_scale*(outcome - expected_score(rating_1,
                                                              rating_2;
                                                              elo_scale))

end

