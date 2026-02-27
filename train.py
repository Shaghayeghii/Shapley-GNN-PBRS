from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json, math, time
from utils import remove_empty_slots
from utils import convert_list_to_dict
from user import User

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='')
    args = parser.parse_args()
    params = vars(args)

    CONSTANTS_FILE_PATH = 'constants.json'
    if len(params['constants_path']) > 0:
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH

    with open(constants_file) as f:
        constants = json.load(f)

    file_path_dict = constants['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    DICT_FILE_PATH = file_path_dict['dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']

    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    WARMUP_MEM = run_dict['warmup_mem']
    NUM_EP_TRAIN = run_dict['num_ep_run']
    TRAIN_FREQ = run_dict['train_freq']
    MAX_ROUND_NUM = run_dict['max_round_num']
    SUCCESS_RATE_THRESHOLD = run_dict['success_rate_threshold']

    database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')
    remove_empty_slots(database)
    db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')
    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)
    else:
        user = User(constants)
    emc = ErrorModelController(db_dict, constants)
    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)

    def run_round(state, warmup=False):
        agent_action_index, agent_action = dqn_agent.get_action(state, use_rule=warmup)
        state_tracker.update_state_agent(agent_action)
        user_action, reward, done, success = user.step(agent_action, state)
        if not done:
            emc.infuse_error(user_action)
        state_tracker.update_state_user(user_action)
        next_state = state_tracker.get_state(done)
        dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)
        return next_state, reward, done, success, user_action, agent_action  # return both actions

    def warmup_run():
        print('Warmup Started...')
        total_step = 0
        while total_step != WARMUP_MEM and not dqn_agent.is_memory_full():
            episode_reset()
            done = False
            state = state_tracker.get_state()
            while not done:
                next_state, _, done, _, _, _ = run_round(state, warmup=True)
                total_step += 1
                state = next_state
        print('...Warmup Ended')

    def train_run():
        print('Training Started...')
        episode = 0
        period_reward_total = 0
        period_success_total = 0
        period_turns_total = 0
        total_latency = 0.0
        success_rate_best = 0.0
        total_inform = 0
        total_correct_matches = 0
        total_user_actions = 0

        while episode < NUM_EP_TRAIN:
            episode_reset()
            episode += 1
            done = False
            state = state_tracker.get_state()
            turn_count = 0
            while not done:
                start = time.time()
                next_state, reward, done, success, user_action, agent_action = run_round(state)
                end = time.time()
                latency = end - start
                total_latency += latency
                period_reward_total += reward
                turn_count += 1
                state = next_state

                # Update Inform F1 and Match Rate calculation
                total_user_actions += 1
                if user_action == agent_action:
                    total_correct_matches += 1
                if user_action != "" and agent_action != "" and user_action == agent_action:
                    total_inform += 1

            period_success_total += success
            period_turns_total += turn_count

            if episode % TRAIN_FREQ == 0:
                success_rate = period_success_total / TRAIN_FREQ
                avg_reward = period_reward_total / TRAIN_FREQ
                avg_turns = period_turns_total / TRAIN_FREQ
                avg_latency = (total_latency / period_turns_total) * 1000  # convert to ms
                inform_f1 = (2 * total_inform) / (total_user_actions + total_inform) if total_user_actions != 0 else 0
                match_rate = total_correct_matches / total_user_actions if total_user_actions != 0 else 0

                print('Episode: {} BEST SUCCESS RATE: {:.2f} Avg Reward: {:.2f} Avg Turns: {:.2f} Avg Latency: {:.2f} ms Inform F1: {:.2f} Match Rate: {:.2f}'.format(
                    episode, success_rate, avg_reward, avg_turns, avg_latency, inform_f1, match_rate))

                if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
                    dqn_agent.empty_memory()
                if success_rate > success_rate_best:
                    print('Episode: {} NEW BEST SUCCESS RATE: {:.2f} Avg Reward: {:.2f} Avg Turns: {:.2f}'.format(
                        episode, success_rate, avg_reward, avg_turns))
                    success_rate_best = success_rate
                    dqn_agent.save_weights()

                period_success_total = 0
                period_reward_total = 0
                period_turns_total = 0
                total_latency = 0.0
                total_inform = 0
                total_correct_matches = 0
                total_user_actions = 0

                dqn_agent.copy()
                dqn_agent.train()
        print('...Training Ended')

    def episode_reset():
        state_tracker.reset()
        user_action = user.reset()
        emc.infuse_error(user_action)
        state_tracker.update_state_user(user_action)
        dqn_agent.reset()

    try:
        start_time = time.process_time()
        warmup_run()
        train_run()
    except KeyboardInterrupt:
        end_time = time.process_time()
        print(f"\nTraining interrupted. CPU time used: {end_time - start_time:.2f} seconds.")
