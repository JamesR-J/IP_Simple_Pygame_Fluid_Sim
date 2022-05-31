import os
import time
import torch
import pickle
import statistics

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch_maddpg_arguments import parse_args
from torch_maddpg_replay_buffer import ReplayBuffer
from torch_maddpg_model import actor_agent, critic_agent
from multi_agent_main_game import Game


print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())


def get_trainers(env, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]

    if arglist.restore == True:  # restore the model
        for idx in arglist.restore_idxs:
            trainers_cur[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(agent_idx))
            trainers_tar[idx] = torch.load(arglist.old_model_name + 't_{}'.format(agent_idx))

    for i in range(env.n):
        actors_cur[i] = actor_agent(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_cur[i] = critic_agent(sum(obs_shape_n), sum(action_shape_n), arglist).to(
            arglist.device)
        actors_tar[i] = actor_agent(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_tar[i] = critic_agent(sum(obs_shape_n), sum(action_shape_n), arglist).to(
            arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def update_trainers(agents_cur, agents_tar, tao):
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar


def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, \
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    # update all trainers
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...' + ' ' * 100)

        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue  # jump to the next model update

            # sample the experience
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample(arglist.batch_size, agent_idx)

            # use the date to update the critic
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                    for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q
            q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_
            tar_value = q_ * arglist.gamma * done_n + rew  # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()

            # use the data to update the actor
            model_out, policy_c_new = actor_c( \
                obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the action of this agent
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

            opt_a.zero_grad()
            (1e-3 * loss_pse + loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # save the model to the path_dir
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                arglist.scenario_name, time_now, game_step))
            if not os.path.exists(model_file_dir):  # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar


def train(arglist):
    """
    init the env, agent and train the agents
    """
    # env = Game('No Render')
    env = Game('Render')

    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    action_shape_n = [env.action_space[i].shape[0] for i in range(env.n)]
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, obs_shape_n, action_shape_n, arglist)
    memory = ReplayBuffer(arglist.memory_size)

    obs_size = []
    action_size = []
    game_step = 0
    update_cnt = 0
    agent_info = [[[]]]  # placeholder for benchmarking info
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    obs_n = env.reset()

    reward_list = []

    start = time.time()

    for episode_gone in range(arglist.max_episode):
        if game_step > 1 and game_step % 100 == 0:
            mean_agents_r = [round(np.mean(agent_rewards[idx][-200:-1]), 2) for idx in range(env.n)]
            mean_ep_r = round(np.mean(episode_rewards[-200:-1]), 3)
            print(" " * 43 + 'episode reward:{} agents mean reward:{}'.format(mean_ep_r, mean_agents_r), end='\r')
        print('=Training: steps:{} episode:{}'.format(game_step, episode_gone), end='\r')

        if env.render_images:
            if episode_gone % 10 == 0:
                env.take_pics = True

        div_list = []
        div_avrge2 = 0

        for episode_cnt in range(arglist.per_episode_max_len):

            env.collisions()
            if env.render_images:
                env.render(div_avrge2)
            env.update_div()

            div_list.append(env.div_norm)

            # only calcs div for latest 1000 div calcs
            div_avrge2 = statistics.mean(div_list[-1250:])

            reward_list.append(div_avrge2)

            if env.render_images:
                env.save_pics(episode_gone + 1, episode_cnt + 1)

            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() for
                        agent, obs in zip(actors_cur, obs_n)]
            # below is good for testing
            # action_n = [[uniform(0, 0.001), uniform(0, 0.0001)] for _ in actors_cur]

            # interact with env
            new_obs_n, rew_n, done_n, info_n = env.step(action_n, env.last_div, div_avrge2)

            # save the experience
            memory.add(obs_n, np.concatenate(action_n), rew_n, new_obs_n, done_n)
            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train the agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(arglist, game_step, update_cnt,
                                                                                        memory, obs_size, action_size,
                                                                                        actors_cur, actors_tar,
                                                                                        critics_cur, critics_tar,
                                                                                        optimizers_a, optimizers_c)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)

            env.last_div = div_avrge2

            if done or terminal:
                obs_n = env.reset()
                agent_info.append([[]])
                episode_rewards.append(0)
                for a_r in agent_rewards:
                    a_r.append(0)
                continue

    total_time = (time.time() - start)

    return game_step, reward_list, total_time


if __name__ == '__main__':
    arglist = parse_args()
    steps, reward, time = train(arglist)

    # edit to adjust output pickle file name
    filename = 'test_1'

    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(reward, f)

    graphname = 'div_versus_timestep_' + filename + '.png'

    with open(filename + '_time_taken.pkl', 'wb') as g:
        pickle.dump(time, g)

    plt.plot(range(steps), reward, 'c')
    plt.xlabel('Time Steps')
    plt.ylabel('Time Averaged Divergence')
    plt.ylim(0,)
    plt.show()

