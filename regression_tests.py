#!/usr/bin/env python
import unittest
import sys

# Note: those tests work only on CUDA. CPU values are different. Also, on some GPUs the values can be different.


class Main(unittest.TestCase):
    def test_train_cartpole_td3_tdg(self):
        from main import ex
        config = dict(
            env_name='GYMMB_CartPole-v1',
            agent_alg='td3',
            tdg_error_weight=5,
            dump_dir=None,
            print_config=False,
            seed=123,
            n_total_steps=200,
            n_warm_up_steps=50,
            model_training_n_batches=10,
            policy_training_n_iters=5,
            policy_actors=32,
            eval_freq=100,
            n_eval_episodes_per_policy=1,
            neptune_project=None
        )
        ret, abs_action = ex.run('train', config_updates=config).result
        self.assertAlmostEqual(ret, 42.)
        self.assertAlmostEqual(abs_action, 0.39846721291542053)

    def test_train_cartpole_td3(self):
        from main import ex
        config = dict(
            env_name='GYMMB_CartPole-v1',
            agent_alg='td3',
            tdg_error_weight=0,
            dump_dir=None,
            print_config=False,
            seed=123,
            n_total_steps=200,
            n_warm_up_steps=50,
            model_training_n_batches=10,
            policy_training_n_iters=5,
            policy_actors=32,
            eval_freq=100,
            n_eval_episodes_per_policy=1,
            neptune_project=None
        )
        ret, abs_action = ex.run('train', config_updates=config).result
        self.assertAlmostEqual(ret, 54.0)
        self.assertAlmostEqual(abs_action, 0.02501722611486911)

    def test_train_test_ddpg_tdg(self):
        from main import ex
        config = dict(
            env_name='GYMMB_Test-v0',
            agent_alg='ddpg',
            tdg_error_weight=5,
            dump_dir=None,
            print_config=False,
            seed=123,
            n_total_steps=200,
            n_warm_up_steps=50,
            model_training_n_batches=50,
            policy_training_n_iters=5,
            policy_actors=32,
            eval_freq=100,
            n_eval_episodes_per_policy=1,
            neptune_project=None
        )
        ret, abs_action = ex.run('train', config_updates=config).result
        self.assertAlmostEqual(ret, -47.70172882080078)
        self.assertAlmostEqual(abs_action, 0.9462572932243347)

    def test_train_test_ddpg(self):
        from main import ex
        config = dict(
            env_name='GYMMB_Test-v0',
            agent_alg='ddpg',
            tdg_error_weight=0,
            dump_dir=None,
            print_config=False,
            seed=123,
            n_total_steps=200,
            n_warm_up_steps=50,
            model_training_n_batches=50,
            policy_training_n_iters=5,
            policy_actors=32,
            eval_freq=100,
            n_eval_episodes_per_policy=1,
            neptune_project=None
        )
        ret, abs_action = ex.run('train', config_updates=config).result
        self.assertAlmostEqual(ret, -0.02898537367582321)
        self.assertAlmostEqual(abs_action, 0.06166519597172737)


if __name__ == '__main__':
    unittest.main(argv=sys.argv)
