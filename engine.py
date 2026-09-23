import pandas as pd

class ProgressionEngine:
    def __init__(self):
        self.logs_path = 'workout_logs.csv'
        self.plan_path = 'workout_plan.csv'
        self.issues_path = 'issue_log.csv'

    def calculate_e1rm(self, weight, reps, rir):
        """Calculates true strength capacity based on RIR."""
        return weight * (1 + 0.0333 * (reps + rir))

    def sweep_session(self, target_date):
        """Phase 1: Sweep the session, compare to targets, and route."""
        logs = pd.read_csv(self.logs_path)
        plan = pd.read_csv(self.plan_path)
        
        # Get only the sets performed on the target_date
        session_logs = logs[logs['date'] == target_date].copy()
        session_logs['problems'] = session_logs['problems'].fillna("")
        
        fast_path_updates = []
        ai_routing_queue = []

        # Group sets by exercise
        grouped = session_logs.groupby(['exercise', 'muscle_group'])

        for (exercise, muscle_group), group in grouped:
            # 1. Check for explicit user problems
            problems = " | ".join(filter(None, set(group['problems'])))
            has_problem = len(problems.strip()) > 0

            # 2. Fetch the target for this exercise
            plan_row = plan[plan['exercise'] == exercise]
            if not plan_row.empty:
                target_w = plan_row.iloc[0]['target_weight_kg']
                target_r = plan_row.iloc[0]['target_reps']
                target_rir = plan_row.iloc[0]['target_rir']
                target_e1rm = self.calculate_e1rm(target_w, target_r, target_rir)
            else:
                target_e1rm = 0  # No target exists yet (new exercise)

            # 3. Calculate actual performance
            group['e1rm'] = group.apply(lambda row: self.calculate_e1rm(row['weight'], row['reps'], row['rir']), axis=1)
            actual_max_e1rm = group['e1rm'].max()
            
            # Find the best set to base the next progression on
            best_set = group.loc[group['e1rm'].idxmax()]

            # Did true strength drop below the target? (using a 2% variance buffer)
            missed_target = actual_max_e1rm < (target_e1rm * 0.98) 

            # ROUTING LOGIC
            if has_problem or missed_target:
                # Trigger Slow Path
                ai_routing_queue.append({
                    'exercise': exercise,
                    'muscle_group': muscle_group,
                    'problem_note': problems,
                    'status': 'Stall/Regression' if missed_target else 'User Note',
                    'actual_e1rm': float(actual_max_e1rm),
                    'target_e1rm': float(target_e1rm),
                    'best_weight': float(best_set['weight']),
                    'best_reps': int(best_set['reps']),
                    'best_rir': int(best_set['rir'])
                })
            else:
                # Trigger Fast Path (Deterministic Increment)
                theoretical_reps = best_set['reps'] + best_set['rir']
                if theoretical_reps >= 12:
                    new_w = best_set['weight'] + 2.5
                    new_r = 8
                else:
                    new_w = best_set['weight']
                    new_r = best_set['reps'] + 1

                fast_path_updates.append({
                    'exercise': exercise,
                    'target_weight_kg': new_w,
                    'target_reps': new_r,
                    'target_rir': 1, 
                    'ai_instructions': "" # Clear AI instructions on success
                })

        return fast_path_updates, ai_routing_queue