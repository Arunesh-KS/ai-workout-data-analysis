import pandas as pd

class WorkoutEngine:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.dropna(subset=['Exercise', 'Weight (kg)', 'Reps'])
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d', errors='coerce')
    def get_exercise_history(self, exercise_name, up_to_date=None):
        ex_df = self.df[self.df['Exercise'].str.lower() == exercise_name.lower()].copy()
        if ex_df.empty:
            return None
            
        # --- NEW: Time Machine Filter ---
        if up_to_date:
            # Filters out any sessions that occurred after the target date
            ex_df = ex_df[ex_df['Date'] <= pd.to_datetime(up_to_date)]
            if ex_df.empty:
                return None
        # --------------------------------
            
        ex_df['Notes'] = ex_df['Notes'].fillna("")
        
        session_summary = ex_df.groupby('Date').agg({
            'Weight (kg)': list,  
            'Reps': list,          
            'RIR': list,           
            'Notes': lambda x: " | ".join(filter(None, set(x))) 
        }).reset_index()
        
        session_summary = session_summary.sort_values(by='Date')
        return session_summary  
    # def get_exercise_history(self, exercise_name):
    #     ex_df = self.df[self.df['Exercise'].str.lower() == exercise_name.lower()].copy()
    #     if ex_df.empty:
    #         return None
            
    #     ex_df['Notes'] = ex_df['Notes'].fillna("")
        
    #     session_summary = ex_df.groupby('Date').agg({
    #         'Weight (kg)': list,  # FIXED: Now this is a list of weights per set
    #         'Reps': list,          
    #         'RIR': list,           
    #         'Notes': lambda x: " | ".join(filter(None, set(x))) 
    #     }).reset_index()
        
    #     session_summary = session_summary.sort_values(by='Date')
    #     return session_summary

    def calculate_default_progression(self, session_summary):
        last_session = session_summary.iloc[-1]
        
        weight_list = last_session['Weight (kg)']
        reps_list = last_session['Reps']
        notes = last_session['Notes']
        
        # 1. Find the heaviest weight lifted that day
        max_weight = max(weight_list)
        
        # 2. Find the reps achieved ONLY at that max weight
        reps_at_max_weight = [r for w, r in zip(weight_list, reps_list) if w == max_weight]
        best_reps = max(reps_at_max_weight)
        
        # 3. Apply Double Progression to the working weight
        if best_reps >= 12:
            default_target = {
                "action": "Increment Weight",
                "weight_kg": max_weight + 2.5,
                "reps": "8"
            }
        else:
            default_target = {
                "action": "Increment Reps",
                "weight_kg": max_weight,
                "reps": str(best_reps + 1) 
            }
            
        return default_target, notes