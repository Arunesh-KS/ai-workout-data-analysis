import pandas as pd
import json
import os
import time
import csv

from google import genai
from google.genai import types

# Import your cleanly separated logic
from engine import ProgressionEngine
from pydantic import TypeAdapter
from schema import AgentResponse  # Schema without InvestigationState


# ==========================================
# CUSTOM JSON ENCODER
# ==========================================

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


# ==========================================
# DATABASE HELPER FUNCTIONS
# ==========================================

def get_user_exercise_catalog(filepath='workout_logs.csv'):
    """Extracts a list of all exercises this specific user has ever performed."""
    try:
        logs = pd.read_csv(filepath)
        return logs['exercise'].dropna().unique().tolist()
    except FileNotFoundError:
        return []


def get_issue_history(muscle_group, filepath='issue_log.csv'):
    try:
        issues = pd.read_csv(filepath)
    except FileNotFoundError:
        return {
            "active_issues": [],
            "past_rectified_issues": []
        }

    mg_issues = issues[issues['muscle_group'] == muscle_group].fillna("")
    active = mg_issues[mg_issues['status'] == 'Active']
    rectified = mg_issues[mg_issues['status'] == 'Rectified']

    return {
        "active_issues": active.to_dict(orient='records'),
        "past_rectified_issues": rectified.to_dict(orient='records')
    }


def get_healthy_sets(target_date, muscle_group, logs_path, flagged_queue_items):
    """Returns all sets for a muscle group that successfully hit their targets."""
    try:
        logs = pd.read_csv(logs_path)
    except FileNotFoundError:
        return []

    session = logs[
        (logs['date'] == target_date) &
        (logs['muscle_group'] == muscle_group)
    ].copy()

    if session.empty:
        return []

    session['set_index'] = session.groupby('exercise').cumcount() + 1

    flagged_tuples = {
        (item['exercise'], item['set_index'])
        for item in flagged_queue_items
    }

    healthy_sets = []

    for _, row in session.iterrows():
        if (row['exercise'], row['set_index']) not in flagged_tuples:
            healthy_sets.append({
                'exercise': row['exercise'],
                'set_index': row['set_index'],
                'weight': float(row['weight']),
                'reps': int(row['reps']),
                'rir': int(row['rir'])
            })

    return healthy_sets


def fetch_exercise_history(
    exercise_name: str,
    filepath='workout_logs.csv',
    session_limit=3
):
    """Fetches all set logs for an exercise across the last N distinct workout sessions."""
    try:
        logs = pd.read_csv(filepath)

        ex_logs = logs[logs['exercise'] == exercise_name]

        if ex_logs.empty:
            return f"No recent data found for {exercise_name}."

        # Get unique dates sorted chronologically,
        # then take the last N sessions.
        unique_dates = sorted(
            ex_logs['date'].dropna().unique()
        )

        last_dates = unique_dates[-session_limit:]

        recent_logs = ex_logs[
            ex_logs['date'].isin(last_dates)
        ]

        return recent_logs.to_dict(orient='records')

    except Exception as e:
        return f"Database Error: {e}"


def fetch_plan_history(query_type: str, filepath='workout_plans.csv'):
    """Fetches the current active plan or the most recent previous plan."""

    try:
        plans = pd.read_csv(filepath)

        plan_cols = [
            'order',
            'exercise',
            'lower_reps',
            'upper_reps',
            'rir',
            'weight_increment(kg)',
            'current_target_weight(kg)',
            'current_target_reps'
        ]

        if query_type == "current":

            current_plan = plans[
                plans['end_date'].isna()
            ]

            if current_plan.empty:
                return "No active plan found."

            structured_plan = (
                current_plan
                .groupby('day_name')
                .apply(
                    lambda x: x.sort_values('order')[plan_cols].to_dict('records')
                )
                .to_dict()
            )

            return {
                "status": "current",
                "start_date": current_plan['start_date'].iloc[0],
                "schedule": structured_plan
            }

        elif query_type == "previous":

            past_plans = plans[
                plans['end_date'].notna()
            ]

            if past_plans.empty:
                return "No previous plans found."

            last_plan_id = (
                past_plans
                .sort_values('end_date', ascending=False)
                .iloc[0]['plan_id']
            )

            last_plan_data = past_plans[
                past_plans['plan_id'] == last_plan_id
            ]

            structured_plan = (
                last_plan_data
                .groupby('day_name')
                .apply(
                    lambda x: x.sort_values('order')[plan_cols].to_dict('records')
                )
                .to_dict()
            )

            return {
                "status": "previous",
                "plan_id": last_plan_id,
                "start_date": last_plan_data['start_date'].iloc[0],
                "end_date": last_plan_data['end_date'].iloc[0],
                "schedule": structured_plan
            }

    except Exception as e:
        return f"Database Error: {e}"


def fetch_session_context(target_date: str) -> dict:
    """Retrieves the global pre-flight context for a specific workout date."""

    try:
        with open('session_history.csv', mode='r') as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row['date'] == target_date:
                    return row

    except FileNotFoundError:
        print("⚠️ session_history.csv not found.")

    return {
        "sleep": "unknown",
        "fatigue": "unknown",
        "physical_activity": "unknown",
        "nutrition": "unknown",
        "stress": "unknown",
        "soreness_pain": "unknown",
        "other_notes": "No data recorded for this session."
    }


# ==========================================
# 3. THE AGENTIC LOOP
# ==========================================

def call_ai_coach(initial_user_message, ai_payload, client):

    print("\n[INITIALIZING GEMINI AGENT (MESSAGE APPEND)...]")

    agent_adapter = TypeAdapter(AgentResponse)

    schema_instructions = json.dumps(
        agent_adapter.json_schema(),
        indent=2
    )

    valid_exercises = get_user_exercise_catalog()

    # ==========================================
    # SYSTEM PROMPT
    # ==========================================

    system_prompt = f"""
You are an expert strength coach investigating a user's stalled progress,
performance decline, or reported training problem.

Your job is NOT to immediately prescribe a solution. Your job is to
investigate the available evidence, form plausible hypotheses, gather
the most useful missing information, and only then decide whether an
intervention is justified.

--------------------------------------------------
AVAILABLE COMMANDS
--------------------------------------------------

1. "QUERY_DATABASE":
Look up recent performance history for a specific exercise.

The history for flagged exercises and the current workout plan are
ALREADY provided in:
- pre_fetched_exercise_history
- pre_fetched_current_plan

Do NOT use QUERY_DATABASE for those exercises again.
Use this command only for additional relevant exercises.

2. "ASK_USER":
Ask the user for highly specific information.

3. "QUERY_PLAN":
Look up the user's previous workout plan when relevant.
The current plan is already provided, so do not query it.

4. "FINALIZE_DIAGNOSIS":
Finish the investigation and prescribe target adjustments
or maintain current targets.

--------------------------------------------------
INVESTIGATION PRINCIPLES
--------------------------------------------------

1. RETRIEVE BEFORE ASKING

If information may already exist in the database or workout-plan
history, retrieve it before asking the user.

Do not ask the user for information that can be obtained through a tool.

The user's baseline lifestyle factors for this workout are already provided
under global_session_context.

Do NOT ask about:
- sleep
- nutrition
- general fatigue
- overall stress
- general soreness

Ask only for specific mechanical details, recent events, or lifestyle
information not already provided.

Before using QUERY_DATABASE, check pre_fetched_exercise_history.
If the exercise is already there, use that data instead.

2. DO NOT FORCE A DIAGNOSIS

If evidence is insufficient, continue investigating.

Distinguish between:
- strongly supported explanation
- plausible explanation
- unresolved possibility

3. INVESTIGATE PATTERNS

Determine whether the problem is isolated or affects several related exercises.

You may query multiple independent exercises in a single turn.

4. CONSIDER TEMPORARY VS PERSISTENT PROBLEMS

A single bad session does not automatically justify changing targets.
Distinguish acute events from chronic trends.

5. PLAN SET / ORDER HANDLING

In the workout plan:
- order = global execution order
- set_index = local set number for that exercise

Different sets of the same exercise may have different targets,
weights, reps, and RIR.

Always reference the correct set_index when making adjustments.

--------------------------------------------------
DATABASE DIRECTORY
--------------------------------------------------

When using QUERY_DATABASE, use the exact exercise name from:

{valid_exercises}

Before using QUERY_DATABASE for a flagged exercise,
check pre_fetched_exercise_history first.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Respond in strict JSON matching this schema:

{schema_instructions}

--------------------------------------------------
MULTI-QUERY BATCHING
--------------------------------------------------

You may include multiple independent actions in "actions".

Examples:
- multiple QUERY_DATABASE actions
- QUERY_DATABASE + QUERY_PLAN
- QUERY_DATABASE + ASK_USER when the question does not depend on
  the database result

Do not batch actions when one depends on another action's result.
"""


    # ==========================================
    # MESSAGE HISTORY
    # ==========================================

    messages = [
        {
            "role": "user",
            "content": initial_user_message
        }
    ]

    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    interaction_count = 0
    total_cached_tokens = 0

    while True:

        try:

            # Throttle
            time.sleep(2)

            # ==========================================
            # CONVERT MESSAGE HISTORY TO GEMINI FORMAT
            # ==========================================

            gemini_contents = []

            for message in messages:

                gemini_role = (
                    "model"
                    if message["role"] == "assistant"
                    else "user"
                )

                gemini_contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[
                            types.Part.from_text(
                                text=message["content"]
                            )
                        ]
                    )
                )

            # ==========================================
            # GEMINI API CALL
            # ==========================================

            response = client.models.generate_content(

                model="gemini-3.5-flash-lite",

                contents=gemini_contents,

                config=types.GenerateContentConfig(

                    system_instruction=system_prompt,

                    temperature=0.5,

                    response_mime_type="application/json"
                )
            )

            # ==========================================
            # TOKEN USAGE
            # ==========================================

            usage = response.usage_metadata

            print(f"usage: {usage}")

            prompt_tokens = (
                getattr(
                    usage,
                    "prompt_token_count",
                    0
                ) or 0
            )

            output_tokens = (
                getattr(
                    usage,
                    "candidates_token_count",
                    0
                ) or 0
            )

            total_tokens_turn = (
                getattr(
                    usage,
                    "total_token_count",
                    0
                ) or 0
            )

            # Gemini usage metadata uses
            # cached_content_token_count for cached input.
            cached_this_turn = (
                getattr(
                    usage,
                    "cached_content_token_count",
                    0
                ) or 0
            )

            total_tokens += total_tokens_turn
            total_input_tokens += prompt_tokens
            total_output_tokens += output_tokens
            total_cached_tokens += cached_this_turn
            interaction_count += 1

            print(
                f"📊 Tokens: "
                f"{prompt_tokens} In | "
                f"{output_tokens} Out | "
                f"{total_tokens_turn} Total | "
                f"Cached this turn: {cached_this_turn} | "
                f"Cached cumulative: {total_cached_tokens}"
            )

            # ==========================================
            # GET RESPONSE TEXT
            # ==========================================

            response_content = response.text

            # ==========================================
            # VALIDATE JSON THROUGH PYDANTIC
            # ==========================================

            parsed_envelope = agent_adapter.validate_json(
                response_content
            )

            actions_list = parsed_envelope.actions

            system_results = []

            # ==========================================
            # EXECUTE ACTIONS
            # ==========================================

            for action in actions_list:

                # --------------------------------------
                # QUERY DATABASE
                # --------------------------------------

                if action.action_type == "QUERY_DATABASE":

                    print(
                        f"🔍 AI is looking up history for: "
                        f"{action.exercise_to_query}"
                    )

                    data = fetch_exercise_history(
                        action.exercise_to_query
                    )

                    system_results.append(
                        f"Database ({action.exercise_to_query}): "
                        f"{json.dumps(data, cls=CustomJSONEncoder)}"
                    )

                # --------------------------------------
                # QUERY PLAN
                # --------------------------------------

                elif action.action_type == "QUERY_PLAN":

                    print(
                        f"📋 AI is looking up plan details: "
                        f"{action.query_type.upper()}"
                    )

                    data = fetch_plan_history(
                        action.query_type
                    )

                    system_results.append(
                        f"Plan ({action.query_type}): "
                        f"{json.dumps(data, cls=CustomJSONEncoder)}"
                    )

                # --------------------------------------
                # ASK USER
                # --------------------------------------

                elif action.action_type == "ASK_USER":

                    print(
                        f"🧠 AI Reasoning: "
                        f"{action.rationale}"
                    )

                    print(
                        f"🗣️ COACH: "
                        f"{action.question}"
                    )

                    user_answer = input(
                        "👉 YOUR ANSWER: "
                    )

                    system_results.append(
                        f"USER ANSWER: {user_answer}"
                    )

                    break

                # --------------------------------------
                # FINALIZE
                # --------------------------------------

                elif action.action_type == "FINALIZE_DIAGNOSIS":

                    print(
                        "total tokens used in this session: ",
                        total_tokens
                    )

                    print(
                        f"Total Input Tokens: "
                        f"{total_input_tokens}"
                    )

                    print(
                        f"Total Output Tokens: "
                        f"{total_output_tokens}"
                    )

                    print(
                        f"Total Cached Input Tokens: "
                        f"{total_cached_tokens}"
                    )

                    print(
                        f"Number of User Interactions: "
                        f"{interaction_count}"
                    )

                    return action.model_dump()

            # ==========================================
            # APPEND HISTORY
            # ==========================================

            if system_results:

                combined_results = "\n---\n".join(
                    system_results
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": response_content
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content":
                            "SYSTEM RESULTS / USER UPDATE:\n"
                            + combined_results
                    }
                )

                print(
                    "📜 [History Appended to Message Stack]"
                )

        except Exception as e:

            print(
                f"Agent Loop Error: {e}"
            )

            return None


# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def run_pipeline(target_date, api_key):

    client = genai.Client(
        api_key=api_key
    )

    engine = ProgressionEngine()

    print(
        f"\n--- RUNNING SESSION ANALYSIS FOR "
        f"{target_date} ---"
    )

    fast_path_updates, ai_routing_queue = (
        engine.sweep_session(target_date)
    )

    if ai_routing_queue:

        print(
            "\n⚠️ AI INTERVENTION REQUIRED:"
        )

        grouped_issues = {}

        for item in ai_routing_queue:

            mg = item['muscle_group']

            if mg not in grouped_issues:
                grouped_issues[mg] = []

            grouped_issues[mg].append(item)

        daily_context = fetch_session_context(
            target_date
        )

        for mg, flags in grouped_issues.items():

            print(
                f"\nGathering context for muscle group: "
                f"[{mg}]..."
            )

            history = get_issue_history(
                mg
            )

            # ==========================================
            # PREFETCH FLAGGED EXERCISE HISTORY
            # ==========================================

            pre_fetched_histories = {}

            flagged_exercise_names = []

            for flag in flags:

                ex_name = flag['exercise']

                if ex_name not in flagged_exercise_names:

                    flagged_exercise_names.append(
                        ex_name
                    )

                    pre_fetched_histories[
                        ex_name
                    ] = fetch_exercise_history(
                        ex_name,
                        session_limit=3
                    )

            # ==========================================
            # PREFETCH CURRENT PLAN
            # ==========================================

            current_plan_data = (
                fetch_plan_history(
                    "current"
                )
            )

            # ==========================================
            # BUILD PAYLOAD
            # ==========================================

            ai_payload = {

                "muscle_group": mg,

                "flagged_sets": flags,

                "healthy_sets_today":
                    get_healthy_sets(
                        target_date,
                        mg,
                        'workout_logs.csv',
                        flags
                    ),

                "pending_fast_path_updates":
                    fast_path_updates,

                "global_session_context":
                    daily_context,

                "pre_fetched_current_plan":
                    current_plan_data,

                "pre_fetched_exercise_history":
                    pre_fetched_histories,

                "active_issues":
                    history["active_issues"],

                "past_rectified_issues":
                    history["past_rectified_issues"]
            }

            # ==========================================
            # INITIAL USER MESSAGE
            # ==========================================

            initial_user_message = f"""
<session_context date="{target_date}" muscle_group="{mg}">

<data_inventory_manifest>

CRITICAL NOTICE:

The complete history and current plan data for the following
flagged exercises are ALREADY loaded below.

You DO NOT need to call QUERY_DATABASE or QUERY_PLAN for them:

{chr(10).join(
    f"- {name}"
    for name in flagged_exercise_names
)}

You may use QUERY_DATABASE only if the user introduces
new information that requires checking another exercise.

</data_inventory_manifest>

<global_lifestyle_context>

{json.dumps(
    daily_context,
    indent=2
)}

</global_lifestyle_context>

<flagged_sets>

{json.dumps(
    flags,
    indent=2
)}

</flagged_sets>

<pre_fetched_current_plan>

{json.dumps(
    current_plan_data,
    indent=2
)}

</pre_fetched_current_plan>

<pre_fetched_exercise_history>

{json.dumps(
    pre_fetched_histories,
    indent=2
)}

</pre_fetched_exercise_history>

<active_issues>

{json.dumps(
    history["active_issues"],
    indent=2
)}

</active_issues>

</session_context>
"""

            ai_prescription = call_ai_coach(
                initial_user_message,
                ai_payload,
                client
            )

            # ==========================================
            # PRINT FINAL PRESCRIPTION
            # ==========================================

            if ai_prescription:

                print(
                    "\n✅ FINAL AI PRESCRIPTION:"
                )

                print(
                    f"  Diagnosis: "
                    f"{ai_prescription['analysis']}"
                )

                for adj in ai_prescription[
                    'adjustments'
                ]:

                    set_idx = adj.get(
                        'set_index',
                        'N/A'
                    )

                    print(
                        f"  - "
                        f"{adj['exercise']} "
                        f"(Set {set_idx}) "
                        f"Target Updated: "
                        f"{adj['new_target_weight_kg']}"
                        f"kg x "
                        f"{adj['new_target_reps']} "
                        f"@ RIR "
                        f"{adj['new_target_rir']}"
                    )

                    print(
                        f"  - Cue: "
                        f"{adj['ai_instructions']}"
                    )

                print(
                    "\n📝 PROPOSED ISSUE LOG UPDATES:"
                )

                for log_update in ai_prescription.get(
                    'issue_log_updates',
                    []
                ):

                    generated_advice = (
                        log_update.get(
                            'ai_advice',
                            'No advice recorded.'
                        )
                    )

                    print(
                        f'  - APPEND TO issue_log.csv: '
                        f'{target_date} | '
                        f'{log_update.get("exercise")} | '
                        f'{mg} | '
                        f'{log_update.get("issue_type", "Problem")} | '
                        f'{log_update.get("description", "")} | '
                        f'Active | '
                        f'"{generated_advice}"'
                    )

    else:

        if fast_path_updates:

            print(
                "\n✅ FAST PATH "
                "(DETERMINISTIC INCREMENTS):"
            )

            for update in fast_path_updates:

                print(
                    f"  - "
                    f"{update['exercise']} "
                    f"(Set {update['order']}): "
                    f"target updated to "
                    f"{update['new_target_weight_kg']}kg x "
                    f"{update['new_target_reps']}"
                )


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:

        print(
            "Error: GEMINI_API_KEY environment variable "
            "not found."
        )

        print(
            'Run this in your terminal first: '
            '$env:GEMINI_API_KEY="your_key_here"'
        )

    else:

        run_pipeline(
            '2026-01-25',
            api_key
        )

#very poor,very high,high,poor,high,high,Poor recovery and noticeable soreness before training.