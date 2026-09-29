import pandas as pd

import pandas as pd

class ProgressionEngine:
    def __init__(self):
        self.logs_path = 'workout_logs.csv'
        self.plan_path = 'workout_plans.csv' # Replaced active_target!
        self.issues_path = 'issue_log.csv'

    def sweep_session(self, target_date):
        """Phase 1: Sweep the session, compare every individual set to its target, and route."""
        logs = pd.read_csv(self.logs_path)
        plan = pd.read_csv(self.plan_path)
        
        # Get only the sets performed on the target_date
        session_logs = logs[logs['date'] == target_date].copy()
        session_logs['problems'] = session_logs['problems'].fillna("")
        
        # Automatically assign a 'set_index' (1, 2, 3...) based on the order logged in the CSV
        session_logs['set_index'] = session_logs.groupby('exercise').cumcount() + 1

        fast_path_updates = []
        ai_routing_queue = []

        # We now evaluate EVERY SINGLE SET completely independently!
        # We now evaluate EVERY SINGLE SET completely independently!
        for _, log_row in session_logs.iterrows():
            exercise = log_row['exercise']
            set_index = log_row['set_index']
            problems = str(log_row['problems']).strip()
            has_problem = len(problems) > 0
            muscle_group = log_row['muscle_group']

            # 1. 🐛 FIX: Get all planned rows for this exercise and sort them by global order
            plan_sets = plan[plan['exercise'] == exercise].sort_values('order')
            
            # If the user logged more sets than planned, skip the extra ones
            if len(plan_sets) < set_index:
                continue 

            # Select the exact set matching the local set index (0-indexed list, so set_index - 1)
            plan_row = plan_sets.iloc[set_index - 1]
            
            # Extract target variables
            target_w = plan_row['current_target_weight(kg)']
            target_r = plan_row['current_target_reps']
            upper_reps = plan_row['upper_reps']
            lower_reps = plan_row['lower_reps']
            increment = plan_row['weight_increment(kg)']
            
            # 🐛 FIX 2: Grab the true global order so the fast_path can update the correct CSV row later!
            true_global_order = int(plan_row['order']) 

            # Extract actual performance
            actual_w = log_row['weight']
            actual_r = log_row['reps']

            # 2. Did they miss the target? 
            missed_target = (actual_w < target_w) or (actual_w == target_w and actual_r < target_r)

            # 3. ROUTING LOGIC
            if has_problem or missed_target:
                # 🛑 SLOW PATH 
                ai_routing_queue.append({
                    'exercise': exercise,
                    'muscle_group': muscle_group, 
                    'set_index': set_index,
                    'problem_note': problems,
                    'status': 'Stall/Regression' if missed_target else 'User Note',
                    'actual_weight': float(actual_w),
                    'actual_reps': int(actual_r),
                    'target_weight': float(target_w),
                    'target_reps': int(target_r)
                })
            else:
                # ⚡ FAST PATH 
                if actual_r >= upper_reps:
                    new_w = actual_w + increment
                    new_r = lower_reps 
                else:
                    new_w = actual_w
                    new_r = actual_r + 1

                fast_path_updates.append({
                    'exercise': exercise,
                    'order': true_global_order, # <-- FIX: Now passes the exact CSV row order!
                    'new_target_weight_kg': new_w,
                    'new_target_reps': new_r
                })

        return fast_path_updates, ai_routing_queue