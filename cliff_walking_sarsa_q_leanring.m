
close all;
clear all;

disp('**** Running SARSA and Q-Learning  ****'); disp(' ');

global HEIGHT; HEIGHT = 4;                  %Grid height
global WIDTH; WIDTH = 12;                   %Grid width
global START_STATE; START_STATE = [4, 1];   %Reset position of game
global GOAL_STATE; GOAL_STATE = [4, 12];    %Goal position

global LEFT; global RIGHT; global UP; global DOWN;  %Actions
LEFT = 1; RIGHT = 2; UP = 3; DOWN = 4;
NACTIONS = 4;

ALPHA = 0.5;                                %Step size - for Q-Learning and Sarsa
GAMMA = 1;                                  %Discount factor - for Q-Learning and Sarsa
Nepisodes = 500;                            %Number of episodes
Nruns = 50;                                 %Number of runs
epsilon_vals = [0.5, 0.3, 0.2, 0.1, 0.01];  %Prob of exploration - for eplison-greedy method
c_vals = [5, 2, 1, 0.5, 0.1];               %Confidence parameter - for UCB method
plot_colours = ['r', 'g', 'b', 'c', 'm'];   %Plot colour for each parameter value


%%% SARSA with e-greedy %%%
figure(1), title('SARSA with e-greedy policy'), xlabel('Episodes'), ylabel('Sum of rewards during episode'), legend(), ylim([-1000 0]), hold on
for i = 1:length(epsilon_vals)
    avg_sum_rewards = zeros(1, Nepisodes);
    for j = 1:Nruns
        Q = zeros(WIDTH*HEIGHT, NACTIONS);
        for k = 1:Nepisodes
            [sum_rewards, Q] = sarsa_egreedy(Q, ALPHA, GAMMA, epsilon_vals(i));
            avg_sum_rewards(k) = avg_sum_rewards(k) + sum_rewards;
        end
    end
    avg_sum_rewards = avg_sum_rewards/Nruns;   
    figure(1), plot(avg_sum_rewards, plot_colours(i), 'DisplayName', strcat('e=',num2str(epsilon_vals(i))))
    if(epsilon_vals(i) == 0.01) sarsa_egreedy_avg_sum_rwds = avg_sum_rewards; end   %picking an epsilon for comparing with Q-learning
end

%%% SARSA with UCB %%%
figure(2), title('SARSA with UCB policy'), xlabel('Episodes'), ylabel('Sum of rewards during episode'), legend(), ylim([-250 0]), hold on
for i = 1:length(c_vals)
    avg_sum_rewards = zeros(1, Nepisodes);
    for j = 1:Nruns
        Q = zeros(WIDTH*HEIGHT, NACTIONS);
        N = zeros(WIDTH*HEIGHT, NACTIONS);
        t = 1;
        for k = 1:Nepisodes
            [sum_rewards, Q, N, t] = sarsa_ucb(Q, ALPHA, GAMMA, c_vals(i), t, N);
            avg_sum_rewards(k) = avg_sum_rewards(k) + sum_rewards;
        end
    end
    avg_sum_rewards = avg_sum_rewards/Nruns;   
    figure(2), plot(avg_sum_rewards, plot_colours(i), 'DisplayName', strcat('c=',num2str(c_vals(i))))
    if(c_vals(i) == 0.1) sarsa_ucb_avg_sum_rwds = avg_sum_rewards; end   %picking a c for comparing with Q-learning
end

%%% Q-learning with e-greedy %%%
figure(3), title('Q-learning with e-greedy policy'), xlabel('Episodes'), ylabel('Sum of rewards during episode'), legend(), ylim([-2000 0]), hold on
for i = 1:length(epsilon_vals)
    avg_sum_rewards = zeros(1, Nepisodes);
    for j = 1:Nruns
        Q = zeros(WIDTH*HEIGHT, NACTIONS);
        for k = 1:Nepisodes
            [sum_rewards, Q] = qlearning_egreedy(Q, ALPHA, GAMMA, epsilon_vals(i));
            avg_sum_rewards(k) = avg_sum_rewards(k) + sum_rewards;
        end
    end
    avg_sum_rewards = avg_sum_rewards/Nruns;   
    figure(3), plot(avg_sum_rewards, plot_colours(i), 'DisplayName', strcat('e=',num2str(epsilon_vals(i))))
    if(epsilon_vals(i) == 0.01) qlearning_egreedy_avg_sum_rwds = avg_sum_rewards; end   %picking an epsilon for comparing with SARSA
end

%%% Q-learning with UCB %%%
figure(4), title('Q-learning with UCB policy'), xlabel('Episodes'), ylabel('Sum of rewards during episode'), legend(), ylim([-250 0]), hold on
for i = 1:length(c_vals)
    avg_sum_rewards = zeros(1, Nepisodes);
    for j = 1:Nruns
        Q = zeros(WIDTH*HEIGHT, NACTIONS);
        N = zeros(WIDTH*HEIGHT, NACTIONS);
        t = 1;
        for k = 1:Nepisodes
            [sum_rewards, Q, N, t] = qlearning_ucb(Q, ALPHA, GAMMA, c_vals(i), t, N);
            avg_sum_rewards(k) = avg_sum_rewards(k) + sum_rewards;
        end
    end
    avg_sum_rewards = avg_sum_rewards/Nruns;   
    figure(4), plot(avg_sum_rewards, plot_colours(i), 'DisplayName', strcat('c=',num2str(c_vals(i))))
    if(c_vals(i) == 0.1) qlearning_ucb_avg_sum_rwds = avg_sum_rewards; end   %picking a c for comparing with SARSA
end

%%% SARSA vs Q-learning comparision (e-greedy policy) %%%
figure(5), title('SARSA vs Q-learning (e-greedy policy, e=0.01)'), xlabel('Episodes'), ylabel('Sum of rewards during episode'), legend(), ylim([-2000 0]), hold on, 
plot(sarsa_egreedy_avg_sum_rwds, 'r', 'DisplayName', 'SARSA'),
plot(qlearning_egreedy_avg_sum_rwds, 'g', 'DisplayName', 'Q-learning')

%%% SARSA vs Q-learning comparision (UCB policy) %%%
figure(6), title('SARSA vs Q-learning (UCB policy, c=0.1)'), xlabel('Episodes'), ylabel('Sum of rewards during episode'), legend(), ylim([-250 0]), hold on, 
plot(sarsa_ucb_avg_sum_rwds, 'b', 'DisplayName', 'SARSA'),
plot(qlearning_ucb_avg_sum_rwds, 'm', 'DisplayName', 'Q-learning')


%****************** Functions ******************%

%%% SARSA with e-greedy policy function %%%
function [sum_rewards, Q_new] = sarsa_egreedy(Q, alpha, gamma, epsilon)

global START_STATE; global GOAL_STATE;
global HEIGHT;

State = START_STATE;
sum_rewards = 0;
A = e_greedy(epsilon, State, Q);
while(~isequal(State, GOAL_STATE))
    [State_next, R] = take_step(State, A);
    A_next = e_greedy(epsilon, State_next, Q);
    sum_rewards = sum_rewards + R;
    %update Q(S, A)
    Q((State(2)-1)*HEIGHT+State(1),A) = Q((State(2)-1)*HEIGHT+State(1),A) + ...
              alpha*(R + gamma*Q((State_next(2)-1)*HEIGHT+State_next(1),A_next) - Q((State(2)-1)*HEIGHT+State(1),A));
    State = State_next;
    A = A_next;
end
Q_new = Q;
end

%%% SARSA with UCB policy function %%%
function [sum_rewards, Q_new, N_new, t_final] = sarsa_ucb(Q, alpha, gamma, c, t_init, N)

global START_STATE; global GOAL_STATE;
global HEIGHT;

State = START_STATE;
sum_rewards = 0;
[A, N] = UCB(c, t_init, N, State, Q);
t = t_init + 1;
while(~isequal(State, GOAL_STATE))
    [State_next, R] = take_step(State, A);
    [A_next, N] = UCB(c, t, N, State_next, Q);
    sum_rewards = sum_rewards + R;
    %update Q(S, A)
    Q((State(2)-1)*HEIGHT+State(1),A) = Q((State(2)-1)*HEIGHT+State(1),A) + ...
              alpha*(R + gamma*Q((State_next(2)-1)*HEIGHT+State_next(1),A_next) - Q((State(2)-1)*HEIGHT+State(1),A));
    State = State_next;
    A = A_next;
    t = t + 1;
end
Q_new = Q;
N_new = N;
t_final = t;
end

%%% Q-learning with e-greedy policy function %%%
function [sum_rewards, Q_new] = qlearning_egreedy(Q, alpha, gamma, epsilon)

global START_STATE; global GOAL_STATE;
global HEIGHT;

State = START_STATE;
sum_rewards = 0;
while(~isequal(State, GOAL_STATE))
    A = e_greedy(epsilon, State, Q);
    [State_next, R] = take_step(State, A);
    sum_rewards = sum_rewards + R;
    %update Q(S, A)
    Q((State(2)-1)*HEIGHT+State(1),A) = Q((State(2)-1)*HEIGHT+State(1),A) + ...
              alpha*(R + gamma*max(Q((State_next(2)-1)*HEIGHT+State_next(1),:)) - Q((State(2)-1)*HEIGHT+State(1),A));
    State = State_next;
end
Q_new = Q;
end

%%% Q-learning with UCB policy function %%%
function [sum_rewards, Q_new, N_new, t_final] = qlearning_ucb(Q, alpha, gamma, c, t_init, N)

global START_STATE; global GOAL_STATE;
global HEIGHT;

State = START_STATE;
sum_rewards = 0;
t = t_init;
while(~isequal(State, GOAL_STATE))
    [A, N] = UCB(c, t, N, State, Q);
    [State_next, R] = take_step(State, A);
    sum_rewards = sum_rewards + R;
    %update Q(S, A)
    Q((State(2)-1)*HEIGHT+State(1),A) = Q((State(2)-1)*HEIGHT+State(1),A) + ...
              alpha*(R + gamma*max(Q((State_next(2)-1)*HEIGHT+State_next(1),:)) - Q((State(2)-1)*HEIGHT+State(1),A));
    State = State_next;
    t = t + 1;
end
Q_new = Q;
N_new = N;
t_final = t;
end

%%% e-greedy policy function %%%
function A = e_greedy(epsilon, State, Q)

global HEIGHT;

Q_state_all_actions = Q((State(2)-1)*HEIGHT+State(1),:);
max_est_reward = max(Q_state_all_actions);
greedy_actions = find(Q_state_all_actions == max_est_reward);
p = randi(length(greedy_actions)); %randomly pick an action if there is a tie
A_greedy = greedy_actions(p);
A_explore = randi(length(Q_state_all_actions));      %randomly pick an action to explore

x = rand;
if(x < epsilon), A = A_explore;
else,            A = A_greedy;
end
end

%%% UCB policy function %%%
function [A, N_new] = UCB(c, t, N, State, Q)

global HEIGHT;

Q_state_all_actions = Q((State(2)-1)*HEIGHT+State(1),:);
N_state_all_action = N((State(2)-1)*HEIGHT+State(1),:);
upper_bound = Q_state_all_actions + c*sqrt(log(t)./(N_state_all_action + 1E-4)); %Adding small offset 1E-4 to prevent div-by-zero
max_est_reward = max(upper_bound);
best_actions = find(upper_bound == max_est_reward);
p = randi(length(best_actions)); %randomly pick an action if there is a tie
A = best_actions(p);

N_new = N;
N_state_all_action(A) = N_state_all_action(A) + 1; 
N_new((State(2)-1)*HEIGHT+State(1),:) = N_state_all_action;
end

%%% Takes an action in the current state %%%
function [State_next, R] = take_step(State, A)

global LEFT; global RIGHT; global UP; global DOWN;
global HEIGHT; global WIDTH;
global START_STATE;

r = State(1); c = State(2);
if(A == LEFT)
    State_next = [r, max(c-1, 1)];
elseif(A == RIGHT)
    State_next = [r, min(c+1, WIDTH)];
elseif(A == UP)
    State_next = [max(r-1, 1), c];
else
    State_next = [min(r+1, HEIGHT), c];
end

R = -1;
if((r==3 && c>=2 && c<=11 && A==DOWN) || (r==4 && c==1 && A==RIGHT))
    R = -100;
    State_next = START_STATE;
end
end

