from models import TriageAction
from server.customer_support_env import CustomerSupportEnv
class EasyGrader:
    def grade(self, obs=None):
        """
        Launch an 'easy' episode, solve it perfectly, return canonical score.
        """
        env = CustomerSupportEnv()
        episode = env.reset("easy", seed=42)
        total_steps = 0
        while not episode.done:
            gt = env._ground_truths[env._step_index]
            # Take the perfect ground-truth action
            obs = env.step(
                TriageAction(
                    route_category=gt.correct_route,
                    urgency_assessment=gt.correct_urgency,
                    resolution_difficulty=gt.correct_difficulty,
                    priority_score=sum(gt.optimal_priority_range) / 2,
                )
            )
            episode = obs
            total_steps += 1
        # Return canonical score
        return float(episode.metadata.get("task_score", 0.01))

class MediumGrader:
    def grade(self, obs=None):
        env = CustomerSupportEnv()
        episode = env.reset("medium", seed=42)
        total_steps = 0
        while not episode.done:
            gt = env._ground_truths[env._step_index]
            obs = env.step(
                TriageAction(
                    route_category=gt.correct_route,
                    urgency_assessment=gt.correct_urgency,
                    resolution_difficulty=gt.correct_difficulty,
                    priority_score=sum(gt.optimal_priority_range) / 2,
                )
            )
            episode = obs
            total_steps += 1
        return float(episode.metadata.get("task_score", 0.01))

class HardGrader:
    def grade(self, obs=None):
        env = CustomerSupportEnv()
        episode = env.reset("hard", seed=42)
        total_steps = 0
        while not episode.done:
            gt = env._ground_truths[env._step_index]
            obs = env.step(
                TriageAction(
                    route_category=gt.correct_route,
                    urgency_assessment=gt.correct_urgency,
                    resolution_difficulty=gt.correct_difficulty,
                    priority_score=sum(gt.optimal_priority_range) / 2,
                )
            )
            episode = obs
            total_steps += 1
        return float(episode.metadata.get("task_score", 0.01))