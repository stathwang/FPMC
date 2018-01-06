library(plyr)
library(data.table)
library(Matrix)
library(doMC)
library(pryr)
library(hashmap)

registerDoMC(detectCores()-1)

# FPMC: Factorized Personalized Markov Chains

# Train a model
fpmc <- function(dat2, k_cf = 32, k_mc = 32, 
                 reg = 0, init_sigma = 1,
                 learning_rate = 0.5, learning_rate_decay = 1, 
                 maxiters = 100, maxtime = Inf,
                 sampling_bias = 100, adaptive_sampling = TRUE, 
                 seed = 123) {
  
  starttime <- Sys.time()
  
  # n_users: total number of users
  # n_prods: total number of products
  n_users <- uniqueN(dat2$usr)
  n_prods <- uniqueN(dat2$prod)
  
  # Initialize the model parameters
  set.seed(seed)
  V_user_prod <- init_sigma * matrix(rnorm(n_users * k_cf), nrow = n_users, ncol = k_cf)
  V_prod_user <- init_sigma * matrix(rnorm(n_prods * k_cf), nrow = n_prods, ncol = k_cf)
  V_prev_next <- init_sigma * matrix(rnorm(n_prods * k_mc), nrow = n_prods, ncol = k_mc)
  V_next_prev <- init_sigma * matrix(rnorm(n_prods * k_mc), nrow = n_prods, ncol = k_mc)
  
  # Training starts here...
  iters <- 1
  avg_cost <- vector('numeric', maxiters)
  current_absolute_cost <- vector('numeric', maxiters)
  current_delta <- vector('numeric', maxiters)
  
  while (iters <= maxiters && Sys.time() - starttime < 60*maxtime) {
    
    # Pick a random training sample
    if (adaptive_sampling == TRUE && iters %% as.integer(n_prods * log(n_prods)) == 1) {
      CF_rank <- apply(V_prod_user, 2, order)
      MC_rank <- apply(V_next_prev, 2, order)
      ranks <- cBind(CF_rank, MC_rank)
      
      CF_var <- apply(V_prod_user, 2, var)
      MC_var <- apply(V_next_prev, 2, var)
      vars <- c(CF_var, MC_var)
    }
    
    user <- sample(all_users, 1)
    user_id <- usr2idx[[user]]
    user_prod_ids <- dat2[usr == user, prod_idx]
    
    rand <- sample(1:(length(user_prod_ids)-1), 1)
    prev_prod <- user_prod_ids[rand]
    true_next <- user_prod_ids[rand + 1]
    
    if (adaptive_sampling == TRUE) {
      while (TRUE) {
        rank <- rexp(1, 1/sampling_bias)
        while (ceiling(rank) >= n_prods) {
          rank <- rexp(1, 1/sampling_bias)
        }
        factor_signs <- sign(c(V_user_prod[user_id,], V_prev_next[prev_prod,]))         # a vector of (k_cf + k_mc) dimensions
        factor_prob <- abs(c(V_user_prod[user_id,], V_prev_next[prev_prod,])) * vars    # a vector of (k_cf + k_mc) dimensions
        f <- sample(k_cf + k_mc, 1, p = factor_prob/sum(factor_prob))                   # a number between 1 and (k_cf + k_mc) inclusive
        mult <- ceiling(rank) * factor_signs[f]
        if (mult < 0) mult <- mult + n_prods + 1
        false_next <- ranks[mult, f]
        if (false_next != true_next) break
      }
    } else {
      false_next <- sample(1:(n_prods-1), 1)
      if (false_next >= true_next) false_next <- false_next + 1
    }
    
    # Update model parameters using stochastic gradient descent one training sample at a time
    # Compute error
    x_true <- sum(V_user_prod[user_id,] * V_prod_user[true_next,]) + 
      sum(V_prev_next[prev_prod,] * V_next_prev[true_next,])
    x_false <- sum(V_user_prod[user_id,] * V_prod_user[false_next,]) + 
      sum(V_prev_next[prev_prod,] * V_next_prev[false_next,])
    absolute_error <- x_false - x_true
    delta <- 1 - 1 / (1 + exp(min(10, max(-10, absolute_error))))
    
    # Update CF
    V_user_prod_mem <- V_user_prod[user_id,]
    V_user_prod[user_id,] <- V_user_prod[user_id,] + learning_rate * (delta * (V_prod_user[true_next,] - V_prod_user[false_next,]) - reg * V_user_prod[user_id,])
    V_prod_user[true_next,] <- V_prod_user[true_next,] + learning_rate * (delta * V_user_prod_mem - reg * V_prod_user[true_next,])
    V_prod_user[false_next,] <- V_prod_user[false_next,] + learning_rate * (-delta * V_user_prod_mem - reg * V_prod_user[false_next,])
    
    # Update MC
    V_prev_next_mem <- V_prev_next[prev_prod,]
    V_prev_next[prev_prod,] <- V_prev_next[prev_prod,] + learning_rate * (delta * (V_next_prev[true_next,] - V_next_prev[false_next]) - reg * V_prev_next[prev_prod,])
    V_next_prev[true_next,] <- V_next_prev[true_next,] + learning_rate * (delta * V_prev_next_mem - reg * V_next_prev[true_next,])
    V_next_prev[false_next,] <- V_next_prev[false_next,] + learning_rate * (-delta * V_prev_next_mem - reg * V_next_prev[false_next,])
    
    current_absolute_cost[iters] <- absolute_error
    current_delta[iters] <- delta
    avg_cost[iters] <- sum(c(current_absolute_cost[1:iters], absolute_error)) / iters
    
    cat('User: ', user, '\n')
    cat('User ID: ', user_id, '\n')
    cat('True Product ID: ', true_next, '\n')
    cat('False Product ID: ', false_next, '\n')
    cat('Iteration: ', iters, '\n')
    cat('Average Error: ', avg_cost[iters], '\n')
    cat('Time Progressed: ', Sys.time() - starttime, '\n')
    cat('\n\n')
    
    iters <- iters + 1
  }
  
  return(list(avg_cost = avg_cost,
              current_absolute_cost = current_absolute_cost,
              current_delta = current_delta,
              V_user_prod = V_user_prod, 
              V_prod_user = V_prod_user, 
              V_prev_next = V_prev_next, 
              V_next_prev = V_next_prev,
              k_cf = k_cf,
              k_mc = k_mc))
}


# Top-k recommendation
top_k <- function(dat2, users, V_user_prod, V_prod_user, V_prev_next, V_next_prev, 
                  top_k = 10, excluded_prods = NULL, parallel = FALSE) {
  dat_trunc <- dat2[usr %in% users]
  dat_trunc <- split(dat_trunc, by = 'usr')
  recommend <- llply(dat_trunc, function(x) {
    user <- unique(x[,usr])
    uid <- unique(x[,usr_idx])
    viewed_prod_ids <- x[,prod_idx]
    last_prod_id <- x[,last(prod_idx)]
    
    # Final recommendation is a simple sum of user-prod matrix 
    # and markov transition matrix in same k-dimension
    
    # drop=FALSE to maintain matrix dimension
    # out is a 1 x length(prod_id) matrix
    scores <- V_user_prod[uid,,drop=FALSE] %*% t(V_prod_user) + 
      V_prev_next[last_prod_id,,drop=FALSE] %*% t(V_next_prev)
    
    # scores[,viewed_prod_ids] <- -Inf
    if (length(excluded_prods) != 0) {
      excluded_prod_ids <- prod2idx[[excluded_prods]]
      out[,excluded_prod_ids] <- -Inf
    }
    
    ranked_prod_ids <- order(scores, decreasing = TRUE)[1:top_k]
    output <- data.table(usr = user,
                         usr_idx = uid,
                         prod = idx2prod[[ranked_prod_ids]],
                         prod_idx = ranked_prod_ids,
                         score = scores[ranked_prod_ids])
    return(output)
  }, .parallel = parallel)
  return(rbindlist(recommend))
}

