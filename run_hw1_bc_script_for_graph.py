from hw1.hw1 import my_app
import hydra, json
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config_hw1")
def my_main(cfg: DictConfig):
    print("CONFIG : ", cfg)
    x =20
    step = 200
    eval_reward = []
    all_steps = []
    for i in range(x):
        print("Batch Size: ", cfg.alg.train_batch_size)
        results = my_app(cfg)
        eval_reward.append(results["eval_reward_Average"])
        all_steps.append(cfg.alg.train_batch_size)
        cfg.alg.train_batch_size += step
    # Use results to graph eval_reward vs all_steps
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(eval_reward, label='Eval reward Average')
    plt.xticks(range(len(all_steps)), all_steps)
    plt.xlabel('Train batch size', fontsize=15)    
    plt.ylabel('eval_reward_Average', fontsize=15)
    plt.title('Reward vs Train batch size', fontsize=15)
    plt.legend(loc='bottom left')
    plt.grid(True)
    plt.savefig("hopt_avg.png")
    plt.show()
    plt.close()

    return results


if __name__ == "__main__":
    import os
    results = my_main()